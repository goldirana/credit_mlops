"""Summary: This script loads the data from the interim folder and process it
and save it in the processed folder

Process: 
    - Converting age from days to years followed by abs
    - Converting days employed from days to years followed by abs
    - Dropping the columns defined in params.yaml
Returns:
    pd.DataFrame: 
        - data with null values : data/interim/null_data.csv
        - data without null values: data/interim/not_null_data.csv
    
"""
import argparse
import os, yaml, joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import read_config
from src.utils import encoders, custom_pd,  logging
from dataclasses import dataclass


config_file = read_config("params.yaml")
@dataclass
class meta_data:
    label_encoder_path = config_file["encoder_path"]["label_encoder"]
    standard_scaler_path = config_file["encoder_path"]["standard_scaler_encoder"] 
    interim_data = config_file["data_source"]["interim_data"]
    drop_col = config_file["drop_columns"]
    fit_encoder = config_file["encoder_path"]["fit_encoder"]
    target_col = config_file["base"]["target_col"]
    
    null_data = config_file["data_source"]["null_data"] 
    not_null_data = config_file["data_source"]["not_null_data"]
    

def seprate_null(data: pd.DataFrame, with_column: str):
    """Function is called to seprate the null values
    """
    try:
        null_data = data[data[with_column].isnull()]
        not_null_data = data[~data[with_column].isnull()]
        logging.info("Null values seprated sucessfully...")
        return null_data, not_null_data
    except Exception as e:
        logging.error(f"Error in seprating null values... [{e}]")
    
if __name__ ==  "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")
    parsed_args = args.parse_args()
    config_file = parsed_args.config

    # read config file 
    config_file = read_config(config_file)
    
    # read the intermediate data
    data = pd.read_csv(meta_data.interim_data, sep = ",", encoding = 'utf-8')
    # ----------------------------------------------------------------
    logging.info("Processing of data started...")
    data.days_birth = round(abs(data.days_birth/365), 2)
    data.days_employed = round(abs(data.days_employed/365), 2)
    # process the data
    data = custom_pd.drop_columns(data, meta_data.drop_col)
   
    null, not_null = seprate_null(data, with_column= meta_data.target_col)
    logging.info("Null values seprated sucessfully...")
    custom_pd.save_data(null, meta_data.null_data, print_message = "Null data saved..." )
    custom_pd.save_data(not_null, meta_data.not_null_data, print_message= "Processed data saved...")
