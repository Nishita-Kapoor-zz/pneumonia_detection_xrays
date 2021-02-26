import torch
import torch.nn as nn
from data.dataloaders import create_dataloaders
import numpy as np
from PIL import Image
from utils.utils import load_checkpoint


def predict(**cfg):
    image_path = cfg["img_path"]
    image = Image.open(image_path)

    model = load_checkpoint(**cfg)
    output = model(image)
    if output == 0:
        print("This is a normal image")
    elif output == 1:
        print("This is a pneumonia image")


def evaluate(**cfg):

    data, dataloaders = create_dataloaders(cfg['datadir'], batch_size=8)
    model, optimizer, train_on_gpu, multi_gpu = load_checkpoint(**cfg)

    test_loader = dataloaders['test']
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    classes = [0, 1]
    criterion = nn.NLLLoss()
    model.eval()
    i = 1
    # iterate over test data
    len(test_loader)
    for data_d, target in test_loader:
        i = i + 1
        if len(target) != 8:
            continue

        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data_d, target = data_d.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data_d)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * 8    # Change later to batch_size- data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        #     print(target)

        for i in range(8):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(2):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
