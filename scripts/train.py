import numpy as np
from timeit import default_timer as timer
import torch
import pandas as pd
from models.models import get_pretrained_model
from torch import cuda, optim
import torch.nn as nn
from torchsummary import summary
from data.dataloaders import create_dataloaders
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Training the model
def train(**cfg):
    """
    Train a PyTorch Model

     Params
     --------
         cfg: Config File with desired params

     Returns
     --------
         model (PyTorch model): trained cnn with best weights
         history (DataFrame): history of train and validation loss and accuracy
     """

    save_path = "./output/train/" + cfg["run_name"] + "/"
    checkpoint = save_path + "checkpoint.pth"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data, dataloaders = create_dataloaders(cfg['datadir'], batch_size=cfg["train"]['batch_size'])

    max_epochs_stop = 5
    print_every = 2

    model = get_pretrained_model(model_name=cfg["model"])

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
    # Move to gpu and parallelize
    if train_on_gpu:
        model = model.to('cuda')
    if multi_gpu:
        model = nn.DataParallel(model)
    if multi_gpu:
        summary(
            model.module,
            input_size=(3, 224, 224),
            batch_size=cfg["train"]["batch_size"],
            device='cuda')
    else:
        summary(
            model, input_size=(3, 224, 224), batch_size=cfg["train"]["batch_size"], device='cuda')

    model.class_to_idx = data['train'].class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    valid_max_acc = 0
    history = []
    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')
    overall_start = timer()
    # Main loop
    for epoch in tqdm(range(cfg["train"]["n_epochs"])):
        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0
        valid_acc = 0
        # Set to training
        model.train()
        start = timer()
        # Training loop
        for ii, (data, target) in enumerate(dataloaders["train"]):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)
            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)
            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
                correct_tensor.cpu().numpy())
            # calculate test accuracy for each object class
            '''for i in range(batch_size):       
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1'''
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)
            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(dataloaders["train"]):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')
        # After training loops ends, start validation
        else:
            model.epochs += 1
            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()
                # Validation loop
                for data, target in dataloaders["val"]:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()
                    # Forward pass
                    output = model(data)
                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)
                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)
                # Calculate average losses
                train_loss = train_loss / len(dataloaders["train"].dataset)
                valid_loss = valid_loss / len(dataloaders["val"].dataset)
                # Calculate average accuracy
                train_acc = train_acc / len(dataloaders["train"].dataset)
                valid_acc = valid_acc / len(dataloaders["val"].dataset)
                history.append([train_loss, valid_loss, train_acc, valid_acc])
                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )
                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), checkpoint)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch
                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.'
                        )
                        # Load the best state dict
                        model.load_state_dict(torch.load(checkpoint))
                        # Attach the optimizer
                        model.optimizer = optimizer
                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])

                        save_and_plot_results(history, save_path)

                        return model

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])

    save_and_plot_results(history, save_path)

    return model

def save_and_plot_results(history, save_path):

    # Save history as a csv file
    history.to_csv(save_path + "summary.csv")

    # Loss Plot
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'valid_loss']:
        plt.plot(
            history[c], label=c)
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
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.savefig(save_path + "accuracy.png", dpi=400)

    print("Training finished successfully and all results saved in " + save_path)
