import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import pandas as pd
from torch import cuda
from torchsummary import summary
import torch.nn as nn
import torch


def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def imshow_tensor(image, ax=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image


def save_and_plot_results(history, save_path):

    # Create a pandas Dataframe
    history = pd.DataFrame(
        history,
        columns=[
            'train_loss', 'valid_loss', 'train_acc',
            'valid_acc'
        ])

    # Save history as a csv file
    history.to_csv(save_path + "summary.csv")

    # Loss Plot
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'valid_loss']:
        plt.plot(
            history[c], label=c)
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.savefig(save_path + "loss.png", dpi=400)

    # Accuracy Plot
    plt.figure(figsize=(8, 6))
    for c in ['train_acc', 'valid_acc']:
        plt.plot(
            100 * history[c], label=c)
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.savefig(save_path + "accuracy.png", dpi=400)

    print("Training finished successfully and all results saved in " + save_path)


def save_checkpoint(model, path):
    """Save a PyTorch model checkpoint

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """
    model_name = path.split('/')[-2].split("-")[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Path must have the correct model name"

    train_on_gpu, multi_gpu = check_gpu()

    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
    }

    # Extract the final classifier and the state dictionary
    if model_name == 'vgg16':
        # Check to see if model was parallelized
        if multi_gpu:
            checkpoint['classifier'] = model.module.classifier
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['classifier'] = model.classifier
            checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'resnet50':
        if multi_gpu:
            checkpoint['fc'] = model.module.fc
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['fc'] = model.fc
            checkpoint['state_dict'] = model.state_dict()

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)


def load_checkpoint(**cfg):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """
    train_on_gpu, multi_gpu = check_gpu()

    save_path = "./output/train/" + cfg["run_name"] + "/"
    checkpoint_path = save_path + "checkpoint.pth"

    # Get the model name
    model_name = cfg["run_name"].split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Path must have the correct model name"

    # Load in checkpoint
    checkpoint = torch.load(checkpoint_path)

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    # Move to gpu
    if multi_gpu:
        model = nn.DataParallel(model)

    if train_on_gpu:
        model = model.to('cuda')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #if multi_gpu:
    #    summary(model.module, input_size=(3, 224, 224), batch_size=cfg['evaluate']['batch_size'])
    #else:
    #    summary(model, input_size=(3, 224, 224), batch_size=cfg['evaluate']['batch_size'])

    return model, optimizer


def check_gpu():
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')

    # Number of gpus
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False
    return train_on_gpu, multi_gpu


def f1score(target, pred, is_training=False):
    '''Calculate F1 score. Can work with gpu tensors

        The original implmentation is written by Michal Haltuf on Kaggle.

        Returns
        -------
        torch.Tensor
            `ndim` == 1. 0 <= val <= 1

    '''
    assert target.ndim == 1
    assert pred.ndim == 1 or pred.ndim == 2

    if pred.ndim == 2:
        pred = pred.argmax(dim=1)

    tp = (target * pred).sum().to(torch.float32)
    tn = ((1 - target) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - target) * pred).sum().to(torch.float32)
    fn = (target * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1



