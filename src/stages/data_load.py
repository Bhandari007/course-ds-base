from typing import Text
import yaml
import pandas as pd
from sklearn.datasets import load_iris
from loguru import logger
import argparse

def load_data(config_path:Text)->None:
    # Read config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    # Read Dataset
    data = load_iris(as_frame=True)
    dataset = data.frame
    dataset.columns = [colname.strip(' (cm)').replace(' ', '_') for colname in dataset.columns.tolist()]
    # Save Dataset
    dataset.to_csv(config["data"]["raw_data_path"], index=False)

    logger.info(f"Dataset Saved at {config["data"]["raw_data_path"]}")

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=True)
    args = arg_parser.parse_args()
    load_data(args.config)
    

 
    

