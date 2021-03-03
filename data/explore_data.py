import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")


def data_analysis(**cfg):

    cat_df, image_df = create_csv(cfg["datadir"])

    if not os.path.exists("./output/data_analysis/plots/"):
        os.makedirs("./output/data_analysis/plots/")

    # Plot 1: Count of Training images by category
    cat_df.set_index('category')['n_train'].plot.bar(
        color='c', figsize=(20, 6))
    plt.xticks(rotation=0)
    plt.ylabel('Count')
    plt.title('Training Images by Category')

    plt.savefig("./output/data_analysis/plots/train_category.png", dpi=300)
    plt.close()

    # Plot 2: Distribution Plot - Average Sizes
    img_dsc = image_df.groupby('category').mean()
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        img_dsc['height'], label='Average Height')
    sns.kdeplot(
        img_dsc['width'], label='Average Width')
    plt.xlabel('Pixels')
    plt.ylabel('Density')
    plt.title('Average Size Distribution')
    plt.savefig("./output/data_analysis/plots/average_size.png", dpi=300)
    plt.close()

    print("Both plots created and saved in ./output/data_analysis/plots/")


def create_csv(datadir):

    traindir = datadir + "/train/"
    validdir = datadir + "/val/"
    testdir = datadir + "/test/"

    if not os.path.exists("./output/data_analysis/"):
        os.makedirs("./output/data_analysis/")

    # Iterate through each category
    if os.path.exists("./output/data_analysis/tables/"):
        cat_df = pd.read_csv("./output/data_analysis/tables/cat_df.csv")
        image_df = pd.read_csv("./output/data_analysis/tables/image_df.csv")
    else:
        # Create a folder to save csv files
        os.makedirs("./output/data_analysis/tables/")

        # Initialize lists
        categories = []
        img_categories = []
        n_train = []
        n_valid = []
        n_test = []
        hs = []
        ws = []

        # Read all images and save dataframes to disk
        for d in os.listdir(traindir):
            if not d.startswith('.'):
                categories.append(d)
                # Number of each image
                train_imgs = os.listdir(traindir + d)
                valid_imgs = os.listdir(validdir + d)
                test_imgs = os.listdir(testdir + d)
                n_train.append(len(train_imgs))
                n_valid.append(len(valid_imgs))
                n_test.append(len(test_imgs))
                # Find stats for train images
                for i in train_imgs:
                    if not i.startswith('.'):
                        img_categories.append(d)
                        img = Image.open(traindir + d + '/' + i)
                        img_array = np.array(img)
                        # Shape
                        hs.append(img_array.shape[0])
                        ws.append(img_array.shape[1])

        # Dataframe of categories
        cat_df = pd.DataFrame({'category': categories,
                               'n_train': n_train,
                               'n_valid': n_valid, 'n_test': n_test}). \
            sort_values('category')
        # Dataframe of training images
        image_df = pd.DataFrame({
            'category': img_categories,
            'height': hs,
            'width': ws
        })
        cat_df.to_csv("./output/data_analysis/tables/cat_df.csv")
        image_df.to_csv("./output/data_analysis/tables/image_df.csv")

    return cat_df, image_df
