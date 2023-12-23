    """This files files the missing values using imputer
    and 
    """
    
# first fill the missing value: imputer
# label encoder
# standard scaler
# save the data to processed folder
# train the model

from sklearn.model_selection import train_test_split
from src.utils import custom_pd, read_config, encoders
from src.logger import logging
import os, sys, argparse, yaml
import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass


config_file = read_config("params.yaml")

@dataclass
class meta_data:
    # data path
    interim_data_path = config_file["data_source"]["interim_data"]
    processed_data_path = config_file["data_source"]["processed_data"]
    target_col = config_file["base"]["target_col"]
    
    test_size = config_file["base"]["test_size"]
    random_state = config_file["base"]["random_state"]
    
    fit_encoder = config_file["encoder_path"]["fit_encoder"]
    label_encoder_path = config_file["encoder_path"]["label_encoder"]
    standard_scaler_path = config_file["encoder_path"]["standard_scaler_encoder"] 
    
    # model_path = config_file["model_path"]["model_path"]
    # model_name = config_file["model_path"]["model_name"]
    # model_version = config_file["model_path"]["model_version"]
    # model_version_note = config_file["model_path"]["model_version_note"]
    # model_version_path = config_file["model_path"]["model_version_path"]
    # model_version_name = config_file["model_path"]["model_version_name"]
    # model_version_note_name = config_file["model_path"]["model_version_note_name"]
    # model_version_pat    



if __name__ == "__main__":
    pass