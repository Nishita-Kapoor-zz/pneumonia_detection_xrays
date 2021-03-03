# Predicting Normal or Pneumonia Chest X-rays using Deep Learning
This is my first attempt at a Machine Learning project. My goal was to get familiar with hands-on machine learning
using Pytorch and implementing various ML algorithms.

#### Abstract:
Pneumonia is a life-threatening infectious disease affecting one or both lungs in humans commonly caused by bacteria 
called Streptococcus pneumoniae. Chest X-Rays which are used to diagnose pneumonia, need expert radiotherapists for 
evaluation. Thus, developing an automatic system for detecting pneumonia would be beneficial for treating the disease 
without any delay particularly in remote areas. 

This work, appraises the functionality of pre-trained CNN models followed by different classifiers for
the classification of abnormal and normal chest X-Rays.

## Table of Contents

- [Installation](#installation)  
- [Files and Directories](#files-and-directories)
- [Usage](#usage)  
- [Data](#dataset)    
- [Models](#models)    
- [Results](#results)    
- [License](#license)
- [Acknowledgements](#acknowledgements)     
- [Footer](#footer)
      
### Installation
Clone project:
```
git clone https://github.com/Nishita-Kapoor/pneumonia_detection_xrays.git
```
Once cloned, install packages:
```
pip install -r requirements.txt
```
Next download data from Kaggle following this [link](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
Note - downloading data from Kaggle requires Kaggle account and API.

### Files and Directories


### Usage
-folder structure, files
-How to Run

#### Config file

### Generated files


### Dataset
The dataset is organized into 3 folders (train, test, val) and contains subfolders for
each image category (Pneumonia/Normal). There are 5,856 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Dataset Name: Chest X-Ray Images (Pneumonia)
Dataset Link: [Kaggle Chest Xray(Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
```
Number of Class         : 2
Number/Size of Images   : Total      : 5856 (1.15 Gigabyte (GB))
                          Training   : 5216 
                          Validation : 16  
                          Testing    : 624  
```
#### Sample Input:

![Normal xray](images/sample_images.png)

#### Training images by category: 

![traindata](output/data_analysis/plots/train_category.png)

Note: The training set is an imbalanced dataset for Normal & PNEUMONIA (about 1:3)

#### Image pre-processing:
To prepare the images for the network, they were resized to 224 x 224 and normalized by 
subtracting a mean value and dividing by a standard deviation. The validation and testing data was not augmented 
but only resized and normalized. The normalization values are standardized for Imagenet.

```
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),  
```


 ### Models
The project uses models built using transfer learning with PyTorch. The models supported are [VGG16](https://arxiv.org/pdf/1409.1556.pdf) and 
[ResNet-50](https://arxiv.org/pdf/1512.03385.pdf). 


### Results
- Hyperparameter tuning experiments
- Metrics (Acc, F1, Confusion matrix)
- Training Curves (plots of train, val) for both models
    
### License
Please see [LICENSE](./LICENSE)
    
### Acknowledgements
I do not claim ownership for this program. This was used as a learning experience; 
this code is from the Kaggle notebook found [here](https://www.kaggle.com/dnik007/pneumonia-detection-using-pytorch/comments)

### Footer
Please feel free to contribute/use the repo as per need. In case of any questions,
you can reach me at <nishita.s66@gmail.com>.