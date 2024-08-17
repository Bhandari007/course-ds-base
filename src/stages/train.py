import yaml
from typing import Text
import pandas as pd
from sklearn.linear_model import LogisticRegression
from loguru import logger
import joblib
import argparse

def train(config_path:Text)-> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    

    train_dataset = pd.read_csv(config["data_split"]["train_set_path"])
    
    y_train = train_dataset.loc[:, 'target'].values.astype('int32')
    X_train = train_dataset.drop('target', axis=1).values.astype('float32')

    # Create an instance of Logistic Regression Classifier CV and fit the data

    logreg = LogisticRegression(C=config["model"]["hyper_params"]["C"], 
                                solver=config["model"]["hyper_params"]["solver"], 
                                multi_class=config["model"]["multi_class"], 
                                max_iter=config["model"]["max_iter"])
    logger.info("Training Model")

    logreg.fit(X_train, y_train)

    joblib.dump(logreg, config["model"]["model_path"])
    logger.info(f"Model saved at {config["model"]["model_path"]}")

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=True)
    args = arg_parser.parse_args()
    train(args.config)

