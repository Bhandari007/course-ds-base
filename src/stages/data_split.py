import yaml
from typing import Text
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
import argparse

def data_split(config_path:Text)->None:
    # Split Dataset
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    dataset = pd.read_csv(config["data"]["processed_data_path"])
    train_dataset, test_dataset = train_test_split(dataset, test_size=config["data_split"]["test_size"], random_state=config["base"]["random_seed"])
    train_dataset.to_csv(config["data_split"]["train_set_path"])
    test_dataset.to_csv(config["data_split"]["test_set_path"])
    logger.info(f"Dataset Splitted and saved at {config["data_split"]["train_set_path"]} and {config["data_split"]["test_set_path"]}")

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=True)
    args = arg_parser.parse_args()
    data_split(args.config)





    