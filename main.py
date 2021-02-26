# Main file to run
# Author: Nishita Kapoor

# Package Imports
import yaml
import argparse
from data.explore_data import data_analysis
from scripts.train import train
from scripts.evaluate import evaluate, predict
import os


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path of config file, e.g. configs/config.yaml", default="configs/final/config.yaml",
                        type=str)
    args = parser.parse_args()

    # Read config file
    with open(str(args.config), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        print(cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]

    task_map = {"EDA": data_analysis, "train": train,
                "evaluate": evaluate, "predict": predict}

    for task in cfg["tasks"]:
        task_map[task](**cfg)

# How to run:
# python main.py --config configs/config.yaml
if __name__ == '__main__':
    main()
