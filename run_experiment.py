import sys
import warnings
import numpy as np
import pandas as pd
import experiment_datasets as datax
import data_utils
from time import time
import matplotlib.pyplot as plt
from IPython.display import display 
from sklearn.metrics import fbeta_score
import model_utils as model_utils
from mlflow import mlflow, log_params, log_param

warnings.filterwarnings('ignore')

import json
pd.set_option('display.max_colwidth', -1)

def build_model(config, learner):

    print(f'Building model for {learner.__class__.__name__}')
    seed = int(config['seed'])
    test_size = config['test_size']

    dataset = pd.read_csv(f'datasets/experiments/{config["experiment_name"]}.csv')

    log_param("records", dataset.shape[0])
    log_param("features", dataset.shape[1])
    log_param("learner", learner.__class__.__name__)

    labels = dataset[['RIESGO_VIDA']]
    features = dataset.drop(['RIESGO_VIDA'], axis = 1)

    print("Split dataset for training / test")
    # Import train_test_split
    from sklearn.model_selection import train_test_split

    # Split the 'features' and 'labels' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test_size, random_state = seed, stratify=labels)

    # Show the results of the split
    print("features_final set has {} samples.".format(features.shape[0]))
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))


    # Collect results on the learners
    learner = model_utils.train_predict(
        learner, 2, X_train, y_train, X_test, y_test)


def build_dataset_experiment(dataset_name, dataset):
    datax.experiment[dataset_name](dataset)


def run_experiment(config, learner):
    dataset = None
    print(config)
    log_params(config)

    if(config['experiment_name'] == 'cie10'):
        dataset = data_utils.get_dataset_null_empty()
    else:
        dataset = data_utils.get_dataset()

    log_param('experiment_name', config["experiment_name"])

    build_dataset_experiment(config["experiment_name"], dataset)
    build_model(config, learner)
    print('----------------------------------------------------------------------------------------\n')

if __name__ == '__main__':
    with open(sys.argv[1]) as json_data_file:
        config = json.load(json_data_file)[0]
    
    classifiers = model_utils.init_classifiers(config['seed'])
    for learner in list(classifiers.values()):
        with mlflow.start_run():
            run_experiment(config, learner)
    
