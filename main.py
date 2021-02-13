# Main file to run
# Author: Nishita Kapoor
# Functionality:

# Package Imports
import yaml
import argparse
from data.explore_data import create_plots
from torchvision import datasets
from torch.utils.data import DataLoader, sampler
from data.dataloaders import image_transforms
from models.models import get_pretrained_model
from torch import cuda, optim
import torch.nn as nn
from torchsummary import summary
import numpy as np
import torch
import pandas as pd
from timeit import default_timer as timer

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path of config file, e.g. configs/config.yaml", default="configs/config.yaml",
                        type=str)
    args = parser.parse_args()

    # Read config file
    with open(str(args.config), "r") as ymlfile:
        cfg = yaml.load(ymlfile)
        print(cfg)

    # Step 1: EDA
    if cfg["plot"]:
        create_plots(**cfg)

    # Step 2: Pre-processing
    data = {
        'train':
            datasets.ImageFolder(root=cfg["datadir"] + "/train/", transform=image_transforms['train']),
        'val':
            datasets.ImageFolder(root=cfg["datadir"] + "/val/", transform=image_transforms['val']),
        'test':
            datasets.ImageFolder(root=cfg["datadir"] + "/test/", transform=image_transforms['test'])
    }

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=cfg["batch_size"], shuffle=True),
        'val': DataLoader(data['val'], batch_size=cfg["batch_size"], shuffle=True),
        'test': DataLoader(data['test'], batch_size=cfg["batch_size"], shuffle=True)
    }

    # Step 3: Training the model
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
            batch_size=cfg["batch_size"],
            device='cuda')
    else:
        summary(
            model, input_size=(3, 224, 224), batch_size=cfg["batch_size"], device='cuda')

    model.class_to_idx = data['train'].class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    # Training the model
    def train(model,
              criterion,
              optimizer,
              train_loader,
              valid_loader,
              save_file_name,
              max_epochs_stop=3,
              n_epochs=20,
              print_every=2):
        """Train a PyTorch Model

        Params
        --------
            model (PyTorch model): cnn to train
            criterion (PyTorch loss): objective to minimize
            optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
            train_loader (PyTorch dataloader): training dataloader to iterate through
            valid_loader (PyTorch dataloader): validation dataloader used for early stopping
            save_file_name (str ending in '.pt'): file path to save the model state dict
            max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
            n_epochs (int): maximum number of training epochs
            print_every (int): frequency of epochs to print training stats

        Returns
        --------
            model (PyTorch model): trained cnn with best weights
            history (DataFrame): history of train and validation loss and accuracy
        """

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
        for epoch in range(n_epochs):

            # keep track of training and validation loss each epoch
            train_loss = 0.0
            valid_loss = 0.0

            train_acc = 0
            valid_acc = 0

            # Set to training
            model.train()
            start = timer()

            # Training loop
            for ii, (data, target) in enumerate(train_loader):
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
                    f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                    end='\r')

            # After training loops ends, start validation
            else:
                model.epochs += 1

                # Don't need to keep track of gradients
                with torch.no_grad():
                    # Set to evaluation mode
                    model.eval()

                    # Validation loop
                    for data, target in valid_loader:
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
                    train_loss = train_loss / len(train_loader.dataset)
                    valid_loss = valid_loss / len(valid_loader.dataset)

                    # Calculate average accuracy
                    train_acc = train_acc / len(train_loader.dataset)
                    valid_acc = valid_acc / len(valid_loader.dataset)

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
                        torch.save(model.state_dict(), save_file_name)
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
                            model.load_state_dict(torch.load(save_file_name))
                            # Attach the optimizer
                            model.optimizer = optimizer

                            # Format history
                            history = pd.DataFrame(
                                history,
                                columns=[
                                    'train_loss', 'valid_loss', 'train_acc',
                                    'valid_acc'
                                ])
                            return model, history

        # Attach the optimizer
        model.optimizer = optimizer
        # Record overall time and print out stats
        total_time = timer() - overall_start
        print(
            f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
        )
        print(
            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
        )
        # Format history
        history = pd.DataFrame(
            history,
            columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
        return model, history

    model, history = train(
        model,
        criterion,
        optimizer,
        dataloaders['train'],
        dataloaders['val'],
        save_file_name=cfg["save_file_name"],
        max_epochs_stop=5,
        n_epochs=10,
        print_every=2)

    # Step 4: Prediction

    # Step 5: Visualization


# How to run:
# python main.py --config configs/config.yaml
if __name__ == '__main__':
    main()