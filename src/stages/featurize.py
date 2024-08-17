from typing import Text
import yaml
import pandas as pd
from loguru import logger
import argparse

def featurize_dataset(config_path:Text)->None:
    # Read config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    dataset = pd.read_csv(config["data"]["raw_data_path"])
    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length'] / dataset['petal_width']

    dataset = dataset[[
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
#     'sepal_length_in_square', 'sepal_width_in_square', 'petal_length_in_square', 'petal_width_in_square',
    'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
    'target'
    ]]

    dataset.to_csv(config["data"]["processed_data_path"])
    logger.info(f"Processed data saved at {config["data"]["processed_data_path"]}")

if __name__== "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--config", dest="config", required=True)
    args = arg_parse.parse_args()
    featurize_dataset(args.config)



