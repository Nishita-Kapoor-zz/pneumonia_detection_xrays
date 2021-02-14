# Main file to run
# Author: Nishita Kapoor
# Functionality:

# Package Imports
import yaml
import argparse
from data.explore_data import create_plots
from scripts.train import train
from torchvision import datasets
from torch.utils.data import DataLoader, sampler
from data.dataloaders import image_transforms
from models.models import get_pretrained_model
from torch import cuda, optim
import torch.nn as nn
from torchsummary import summary



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

    # Step 3: Training the model
    if cfg["train"]:
        model = train(**cfg)
    elif cfg["evaluate"]:
        pass

    # Step 4: Prediction

    # Step 5: Visualization


# How to run:
# python main.py --config configs/config.yaml
if __name__ == '__main__':
    main()