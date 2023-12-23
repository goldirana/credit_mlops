"""This files files the missing values using imputer
and 
"""

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import custom_pd, read_config, encoders, read_pkl, write_pkl
from src.utils import sanity_check_for_ml_model
from src.logger import logging
import os, sys, argparse, yaml
import pandas as pd
import numpy as np
import joblib, pickle
from dataclasses import dataclass


config_file = read_config("params.yaml")

@dataclass
class meta_data:
    # data source
    interim_data_path = config_file["data_source"]["interim_data"]
    null_data = config_file["data_source"]["null_data"]
    not_null_data = config_file["data_source"]["not_null_data"] 
    train_path = config_file["data_source"]["train_data_path"]
    test_path = config_file["data_source"]["test_data_path"]
    prediction_path = config_file["data_source"]["prediction_data_path"]
    
    # base info
    target_col = config_file["base"]["target_col"]
    test_size = config_file["base"]["test_size"]
    random_state = config_file["base"]["random_state"]
    
    # encoder info
    fit_encoder = config_file["encoder_path"]["fit_encoder"]
    label_encoder_path = config_file["encoder_path"]["label_encoder"]
    standard_scaler_path = config_file["encoder_path"]["standard_scaler_encoder"] 
    
    dummy_variable = config_file["dummy_variable"]
    
    # model_path = config_file["model_path"]["model_path"]
    # model_name = config_file["model_path"]["model_name"]
    # model_version = config_file["model_path"]["model_version"]
    # model_version_note = config_file["model_path"]["model_version_note"]
    # model_version_path = config_file["model_path"]["model_version_path"]
    # model_version_name = config_file["model_path"]["model_version_name"]
    # model_version_note_name = config_file["model_path"]["model_version_note_nam..e"]
    # model_version_pat    

def fill_na(data: pd.DataFrame) -> pd.DataFrame:
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype == object:
                data[col] = data[col].fillna(data[col].mode()[0])
            elif data[col].dtype == int or data[col].dtype == float:
                data[col] = data[col].fillna(data[col].median())
            else:
                logging.error(f"Data type {data[col].dtype} not identified")
    return data 

label_encoder_col = ['flag_own_realty',
       'name_income_type', 'name_education_type',
        'occupation_type']

standard_scaler_col = ['amt_income_total','days_birth', 'days_employed']

def fit_label_encoder(data: pd.DataFrame, columns: list, saved_dir: str) -> None:
    fitted_encoder = {}
    try:
        for col in columns:
            le = LabelEncoder()
            fitted_encoder.update({col: le.fit(data[col])})
        logging.info("Label encoder fitted sucessfully...")
        write_pkl(fitted_encoder, saved_dir)
    except Exception as e:
        logging.error(f"Error in fitting the label encoder... [{e}]")
    

def fit_standard_scaler(data: pd.DataFrame, columns: list, saved_dir: str) -> None:
    fitted_encoder = {}
    try:
        for col in columns:
            sc = StandardScaler()
            fitted_encoder.update({col: sc.fit(np.array(data[col]).reshape(-1, 1))})
        logging.info("Standard scaler fitted sucessfully...")
        write_pkl(fitted_encoder, saved_dir)
    except Exception as e:
        logging.error(f"Error in fitting the standard scaler... [{e}]")

def transform_encoder(data: pd.DataFrame, columns: list, fitted_encoder: dict) -> pd.DataFrame:
    for col in fitted_encoder.keys():
        try:
            data[col] = fitted_encoder[col].transform(np.array(data[col]).reshape(-1, 1))
        except Exception as e:
            logging.error(f"Error in transforming the encoder... [{e}]")
    return data

if __name__ == "__main__":
    
    # fill missing values
    not_null_data = fill_na(data = pd.read_csv(meta_data.not_null_data, sep = ",", encoding = 'utf-8'))
    null_data = fill_na(data= pd.read_csv(meta_data.null_data, sep = ",", encoding = 'utf-8'))

    if meta_data.fit_encoder == True:
        print(not_null_data.info())
        print(not_null_data.isnull().sum())
        fit_label_encoder(data = not_null_data, columns= label_encoder_col, saved_dir= meta_data.label_encoder_path)
        fit_standard_scaler(data = not_null_data, columns= standard_scaler_col, saved_dir= meta_data.standard_scaler_path)
  
        # read the encoder
        le = read_pkl(meta_data.label_encoder_path)
        sc = read_pkl(meta_data.standard_scaler_path)
        
        # transform the data using encoder
        null_data = transform_encoder(null_data, columns= label_encoder_col, fitted_encoder = le)
        null_data = transform_encoder(null_data,columns= standard_scaler_col, fitted_encoder = sc)
        
        not_null_data = transform_encoder(not_null_data,columns= label_encoder_col, fitted_encoder = le)
        not_null_data = transform_encoder(not_null_data,columns= standard_scaler_col, fitted_encoder = sc)
        
        logging.info("Encoder applied sucessfully...")
    else:
        logging.info("Bypassing the encoder...")
        
    """save the data: 
        - not_null -> split into train and test -> data/processed
        - null_data -> data/prediction
    """
    # split the data
    train, test = train_test_split(not_null_data, test_size = meta_data.test_size, random_state = meta_data.random_state)
    # save the data
    sanity_check_for_ml_model(train)
    custom_pd.save_data(train, meta_data.train_path, print_message = "Train data saved sucessfully...")
    sanity_check_for_ml_model(test)
    custom_pd.save_data(test, meta_data.test_path, print_message = "Test data saved sucessfully...")
    custom_pd.save_data(null_data, meta_data.prediction_path, print_message = "Prediction data saved sucessfully...")
    