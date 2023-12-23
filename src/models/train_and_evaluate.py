"""Summary: Train the model and save it to models.
"""


from sklearn.metrics import accuracy_score, f1_score,log_loss, precision_score, recall_score, roc_auc_score, balanced_accuracy_score
from src.utils import read_config, read_pkl, write_pkl, import_module_str
from src.utils import custom_pd, encoders
from src.logger import logging
import os, sys, argparse, yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass
import mlflow
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA


config_file = read_config("params.yaml")

@dataclass
class meta_data:
    # data source
    train_data_path = config_file["data_source"]["train_data_path"]
    test_data_path = config_file["data_source"]["test_data_path"]
    
    target_col = config_file["base"]["target_col"]
    random_state = config_file["base"]["random_state"]
    # model attributes
    model_path = config_file["model"]["model_path"]
    model_name = config_file["model"]["model_name"]
    model_module_name = config_file["model"]["model_module_name"]
    model_run_id = config_file["model"]["model_run_id"]
    model_version_note = config_file["model"]["model_version_note"]
    model_note = config_file["model"]["model_note"]
    model_params = config_file["params"]
    
    # modeling data hyperparameters
    pca_ok = config_file["tuning"]["pca_ok"]
    pca_components = config_file["tuning"]["pca_components"]
    smote_ok = config_file["tuning"]["smote_ok"]

    
    # model_version_note = config_file["model_path"]["model_version_note"]
    # model_version_path = config_file["model_path"]["model_version_path"]
    # model_version_name = config_file["model_path"]["model_version_name"]
    # model_version_note_name = config_file["model_path"]["model_version_note_name"]

# to make directory of the model
os.makedirs(meta_data.model_name, exist_ok = True)

def fit_pca(data:pd.DataFrame, n_components: int):
    """Function is called to fit the PCA
    """
    pca = PCA(n_components = n_components)
    pca.fit(data)
    return pca

# Oversampling the dataset
smote = SMOTE(random_state= meta_data.random_state)
# Fit and apply SMOTE only on the training set to avoid data leakage


if __name__ == "__main__":
    mlflow.sklearn.autolog(log_models = True, log_datasets=True,
                           registered_model_name = meta_data.model_name)
    with mlflow.start_run( description = meta_data.model_note):
        
        train = pd.read_csv(meta_data.train_data_path, sep = ",", encoding = 'utf-8')
        y = train.pop(meta_data.target_col) 
        
        test = pd.read_csv(meta_data.test_data_path, sep = ",", encoding = 'utf-8')
        test_y = test.pop(meta_data.target_col)
        
        # get the model name
        model = import_module_str(meta_data.model_module_name)
        
        model_instance = model(**meta_data.model_params)
        
        if meta_data.smote_ok == True:
            train, y = smote.fit_resample(train, y)
            logging.info("Oversampling the dataset and training the model")

        if meta_data.pca_ok == True:
            pca = fit_pca(train, meta_data.pca_components)
            # transform the data
            train = pca.transform(train)
            test = pca.transform(test)
            logging.info("Applying PCA to the dataset")
        
        model_instance.fit(train, y)
        prediction = model_instance.predict(test)
        
        mlflow.log_metric("test_accuracy", accuracy_score(test_y, prediction))
        mlflow.log_metric("test_f1", f1_score(test_y, prediction))
        mlflow.log_metric("test_precision", precision_score(test_y, prediction))
        mlflow.log_metric("balanced_accuracy", balanced_accuracy_score(test_y, prediction))
        mlflow.log_metric("test_roc_auc", roc_auc_score(test_y, prediction))
        mlflow.log_metric("test_recall", recall_score(test_y, prediction))
        mlflow.log_param("test_log_loss", log_loss(test_y, prediction))