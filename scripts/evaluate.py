import torch
import torch.nn as nn
from data.dataloaders import create_dataloaders
import numpy as np
from PIL import Image
from utils.utils import load_checkpoint, check_gpu
import os
import pandas as pd
from torchvision import transforms

def predict(**cfg):

    # Create a test loader
    loader = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the image as a Pytorch Tensor
    image = loader(Image.open(cfg["predict"]["image_path"]).convert("RGB")).float().unsqueeze(0)

    # Load Model
    model, _ = load_checkpoint(**cfg)

    # Check GPU availability
    train_gpu, _ = check_gpu()
    if train_gpu:
        image = image.cuda()

    # Get model prediction
    output = model(image)
    _, pred = torch.max(output, 1)

    # Check prediction - Normal or Pneumonia
    if pred == 0:
        print("Prediction: Normal")
    elif pred == 1:
        print("Prediction: Pneumonia! Please see a doctor ASAP!.")


def evaluate(**cfg):

    data_d, dataloaders = create_dataloaders(cfg['datadir'], batch_size=cfg['evaluate']['batch_size'])
    model, optimizer = load_checkpoint(**cfg)
    train_on_gpu, multi_gpu = check_gpu()

    # Set default split as test
    if cfg["evaluate"]["data_split"] is not None:
        data_splits = list(cfg["evaluate"]["data_split"])
    else:
        data_splits = ["test"]

    # Create evaluate folder
    save_path = "./output/evaluate/" + cfg["run_name"] + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    results_dict = {}
    # Iterate in all data_splits
    for data_split in data_splits:
        
        evaluation_loader = dataloaders[data_split]
        evaluation_loss = 0.0
        evaluation_acc = 0.0

        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        classes = [0, 1]
        criterion = nn.NLLLoss()
        model.eval()
        i = 1

        # iterate over dataset
        for data, target in evaluation_loader:
            i = i + 1
            if len(target) != 8:
                continue
    
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update evaluation loss
            evaluation_loss += loss.item() * data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            evaluation_acc += accuracy.item() * data.size(0)

            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

            # calculate evaluation accuracy for each object class
            for i in range(cfg['evaluate']['batch_size']):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
        # average evaluation loss and accuracy
        evaluation_loss = evaluation_loss / len(evaluation_loader.dataset)
        print(str(data_split).capitalize() + ' Loss: {:.6f}\n'.format(evaluation_loss))
        evaluation_acc = evaluation_acc / len(evaluation_loader.dataset)
        results_dict[data_split] = [evaluation_loss, evaluation_acc]

        for i in range(2):
            if class_total[i] > 0:
                print(str(data_split).capitalize() + ' Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print(str(data_split).capitalize() + ' Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    
        print('\n' + str(data_split).capitalize() + ' Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))

    results_df = pd.DataFrame(results_dict, index=['Loss', 'Accuracy'])
    results_df.to_csv(save_path + "results.csv")

