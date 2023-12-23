# -*- coding: utf-8 -*-

""" Summary: This file is responsible to create intermediate data by joining 
application_record and credit_record
Returns-> pd.DataFrame
Dire-> data/interim/joined_data.csv"""

import os, sys
# os.chdir("/Users/goldyrana/work/ATU/Sem1/predictive/t_project1/credit_card_approval_prediction")
import pandas as pd
import numpy as np
from src.utils import custom_pd, read_config
import argparse, yaml
from src.logger import logging
from typing import TextIO
from dataclasses import dataclass

# function to read config path
config_file = read_config("params.yaml")

@dataclass
class meta_data:
    # data path
    application_record = config_file["raw_data_source"]["application_record"]
    credit_record = config_file["raw_data_source"]["credit_record"]
    interim_data_path = config_file["data_source"]["interim_data"]
    
    # features manipulation
    drop_col = config_file["drop_columns"]
    greater_than = config_file["base"]["second_label_mapper"]["greater_than"]
    less_than = config_file["base"]["second_label_mapper"]["less_than"]
    target_col = config_file["base"]["target_col"]

    # label mapper
    mapper = config_file["base"]["label_mapper"]
    mapper_agg_function = config_file["base"]["mapper_agg_function"]


def vintage_analysis(data):
    """This function create the labels for the target column"""
    
    for key, value in meta_data.mapper.items(): # loop to convert the values in the dictionary to int defined in params.yaml
        meta_data.mapper[key] = int(value)
    
    data = custom_pd.create_label(data, meta_data.target_col, meta_data.mapper)     
    data_with_label = data.groupby('id').agg({meta_data.target_col: meta_data.mapper_agg_function})

    data = data_with_label.reset_index(inplace=False)
    
    def second_label_mapper(x):
        if int(meta_data.less_than) == int(meta_data.greater_than):
            logging.error("Greater than and less than values are same")
        if x <= int(meta_data.less_than):
            return 0 # Good client
        elif x >= int(meta_data.greater_than):
            return 1 # Bad client 
        else:
            logging.info("Error in creating labels for the target column: Vintage analysis")
            
    data_with_label[meta_data.target_col] = data_with_label[meta_data.target_col].apply(lambda x: second_label_mapper(x))
    data_with_label = data_with_label.reset_index(inplace = False)
    return data_with_label


def create_interm_data(config_file_name: str):
    """Function is called to read the data source
    and save it in the data/interim folder after creating labels
    """
    try:
        # pd.read_csv(meta_data.application_record)
        df1 = custom_pd.lower_name(pd.read_csv(meta_data.application_record))
        df2 = custom_pd.lower_name(pd.read_csv(meta_data.credit_record))
        logging.info("Raw data read sucessfully...")
    except Exception as e:
        print(e)
        logging.info(f"Error in reading the data source [{e}]")

    # drop duplicates records
    df1 = custom_pd.drop_duplicates_records(df1)
    df2 = custom_pd.drop_duplicates_records(df2)
    
    # create labels for the target column
    try:
        df2 = vintage_analysis(df2)
        print(df2.columns, "------------------------------")
        logging.info("Vintage analysis done...")
        
    except Exception as e:
        logging.info(f"Error occured in Vintage analysis [{e}]")
        sys.exit()
        
    # join the data
    df = df1.join(df2.set_index("id"), how = 'left', on = 'id', lsuffix="main")
    return df
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")
    parsed_args = args.parse_args()
    config_file_name = parsed_args.config
    config_file = read_config("params.yaml")
    
    # Create staging/intermediate data
    data = create_interm_data(config_file_name)
    try:
        custom_pd.save_data(data, meta_data.interim_data_path)
    except Exception as e:
        logging.info(f"Error in saving intermediate data [{e}]")
        sys.exit()
    

