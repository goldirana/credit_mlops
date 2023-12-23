import os, joblib, pickle
# os.chdir("/Users/goldyrana/work/ATU/Sem1/predictive/t_project1/credit_card_approval_prediction")
from sklearn.model_selection import train_test_split
import polars as pl 
import pandas as pd
import numpy as np
import os, sys, yaml, argparse
from src.logger import logging


def read_config(config: str):
    with open(config) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


class custom_pd:
    def __init__(self):
        pass
    
    @staticmethod
    def drop_duplicates_records(df: pd.DataFrame):
        df = df.drop_duplicates(inplace = False)
        return df

    @staticmethod
    def lower_name(data: pd.DataFrame):
        try:
            data.columns = [column.lower() for column in data.columns]
        except Exception as e:
            logging.warning(f"Lowering the column names failed... [{e}]")
        return data

    @staticmethod
    def create_label(data: pd.DataFrame, column: str, mapper: dict):
        """Function is called to create labels for the data
        """
        data[column] = data[column].map(mapper)
        return data
            
    @staticmethod
    def save_data(data: pd.DataFrame, directory:str, print_message= None):
        try:
            data.to_csv(directory, index = False)
            print(print_message)
            logging.info(f"Data saved to sucessfully to {directory}")
        except Exception as e:
            logging.ERROR(f"Error in saving the data to {directory}... [{e}]")

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: list):
        """Function is called to drop the columns
        """
        try:
            df = df.drop(columns, axis = 1, inplace = False)
            return df
        except Exception as e:
            logging.info(f"Columns {columns} not found in axis = 1 ... [{e}]")

class encoders:
    def __init__(self):
        pass
            
    @staticmethod
    def fit_encoder(data, columns, destination_dir: str, encoder: object, encoder_name = None) -> None:
        """Function is called to apply the encoder on the data
        and save to destination directory using joblib extenstion file"""
        encoder = encoder
        print(type(encoder))
        fitted_encoder = {}
        for column in data.columns:
            if data[column].isnull().sum() > 0:
                logging.info(f"Null values found in the column {column}")
            try:
                fitted_encoder.update({column: encoder.fit(np.array(data[column]).reshape(-1, 1))})
            except Exception as e:
                logging.ERROR(f"Failed to apply {encoder_name} encoder on {column}: {[e]}")
        joblib.dump(fitted_encoder, destination_dir) # change the extenstion here
     
    @staticmethod   
    def read_encoder(directory: str) -> dict:
        try:
            if directory.split(".")[-1] == "pkl":
                return pickle.load(directory)
            elif directory.split(".")[-1] == "joblib":
                return joblib.load(directory)
            else: # In future, add new extenstion here 
                raise Exception("Read object Failed; File format not supported")
        except Exception as e:
            logging.info(f"Error in reading the fitted object from {directory}... [{e}]")
    
    @staticmethod
    def transform_encoder(data: pd.DataFrame, fitted_encoder: dict, encoder_name = None)-> pd.DataFrame:
        """Function is called to apply the encoder on the data
        """
        try:
            for column in fitted_encoder.keys():
                try:
                    data[column] = fitted_encoder[column].transform(np.array(data[column]).reshape(-1, 1))
                except Exception as e:
                    logging.info(f"Failed to transform encoder on {column}... {[e]}")               
            logging.info(f"Encoder apply {encoder_name} sucessfully...")
            return data
        except Exception as e:
            logging.info(f"Error in applying the label encoder... [{e}]")