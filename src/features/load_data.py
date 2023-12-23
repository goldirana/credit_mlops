import os
import yaml
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# function to read config path
def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

