import os, joblib, pickle
# os.chdir("/Users/goldyrana/work/ATU/Sem1/predictive/t_project1/credit_card_approval_prediction")
from sklearn.model_selection import train_test_split
import polars as pl 
import pandas as pd
import numpy as np
import os, sys, yaml, argparse
from src.logger import logging
from typing import Type
from sklearn.preprocessing import LabelEncoder
import importlib

class dummy_class:
    pass

def read_config(config: str):
    with open(config) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def read_pkl(directory: str):
    with open(directory,"rb") as file:
        data = pickle.load(file)
    return data

def write_pkl(data: object, directory: str):
    with open(directory, "wb") as file:
        pickle.dump(data, file)

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

    @staticmethod
    def create_dummy(data: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            try:
                dummy_data = pd.get_dummies(data, columns = [col], drop_first = True)
                data = pd.concat([data, dummy_data], axis = 1)
            except Exception as e:
                logging.error(f"Error in creating dummy variables... [{e}]")
        return data

def sanity_check_for_ml_model(data):
    """This functiopn is called to check the sanity of the data
    - check the null values
    - check the data type should be int or float for model training
    """
    
    if data.isnull().sum().sum() == 0:
        logging.info("[Sanity check Passed]: No null values found in the data")
    else:
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                logging.error(f"[Sanity check Failed]: Null values found in the column {col}")
            
    for col in data.columns:
        if data[col].dtype not in [int, float]:
            logging.error(f"[Sanity check Failed]: Data type {data[col].dtype} identified")
            try:
                data[col] = data[col].astype(float)
                logging.info("but Data can be type changed to float")
            except:
                pass

class encoders:
    def __init__(self):
        pass
            
    @staticmethod
    def fit_encoder(data, columns, destination_dir: str, encoder: object, encoder_name = None) -> None:
        """Function is called to apply the encoder on the data
        and save to destination directory using joblib extenstion file"""
    
        print(type(encoder))
        fitted_encoder = {}
        for column in columns:
            encoder = encoder
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

def import_module_str(module_name: str):
    """Function is called to import the module from string
    """
    module_name, class_name = module_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class

    

     
# from sklearn.preprocessing import LabelEncoder
# import numpy as np

# class CustomLabelEncoder(LabelEncoder):
#     def __init__(self, handle_unknown='error', *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.handle_unknown = handle_unknown

#     def fit(self, y):
#         super().fit(y)
#         self.classes_ = np.append(self.classes_, '__unknown__')

#     def transform(self, y):
#         if self.handle_unknown == 'error' and any(label not in self.classes_ for label in y):
#             raise ValueError("Unknown labels present in input data")

#         y = np.array(y)
#         y_unknown = np.isin(y, self.classes_)
#         y_encoded = super().transform(np.where(y_unknown, y, '__unknown__'))
#         return y_encoded

#     def inverse_transform(self, y):
#         y = np.array(y)
#         y_inverse = super().inverse_transform(y)
#         return np.where(y == self.classes_.size - 1, 'unknown', y_inverse)

# Example usage
# labels = ['cat', 'dog', 'fish']
# encoder = CustomLabelEncoder(handle_unknown='error')
# encoder.fit(labels)

# # Transform with unknown label
# new_labels = ['cat', 'dog', 'bird']
# encoded_labels = encoder.transform(new_labels)
# print("Encoded Labels:", encoded_labels)

# # Inverse transform
# decoded_labels = encoder.inverse_transform(encoded_labels)
# print("Decoded Labels:", decoded_labels)
