# coding=utf-8

###
#       This script runs the same code used in the notebooks. It's goal
#       is to find the best settings for the pipeline. The parameters's
#       pipeline are in config.json and contains the seeds, split sizes
#       and imput values settings.
#
#       The script will pick the parameters that produces the best model
#       and it will store that parameters in best_config.json.
#
#       The notebooks will use best_config.json replicate the best model
#       gave by this script.
#
#       Commnad to run the script:    
#       python -W ignore pipeline_parameters_search.py
###

import json
import random
import pandas as pd
import numpy as np
import data_utils
import model_utils
import time

from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

from mlxtend.classifier import StackingClassifier

def clean_dataset(path):
    dataset = data_utils.get_dataset()
    dataset = data_utils.clean_afec_dpto(dataset)
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = data_utils.clean_cie_10(dataset)
    dataset = data_utils.remove_features(dataset)
    dataset.to_csv(path, index = False)

def preprocess_dataset(path, encoded_path):
    dataset = pd.read_csv(path)
    labels = dataset[['RIESGO_VIDA']]
    features = dataset.drop(['RIESGO_VIDA'], axis = 1)
    encoded_features = data_utils.encode_features(features, labels)
    encoded_features['RIESGO_VIDA'] = labels
    encoded_features.to_csv(encoded_path, index = False)


def run_pipeline(seed, test_size, imput):
    dataset = pd.read_csv("datasets/encoded_imput_dataset.csv") if imput == 1 else pd.read_csv("datasets/encoded_dataset.csv")
    labels = dataset[['RIESGO_VIDA']]
    features = dataset.drop(['RIESGO_VIDA'], axis = 1)

    rf_classifier = RandomForestClassifier(random_state = seed)
    ada_classifier = AdaBoostClassifier(random_state = seed)
    gauss_classifier = GaussianNB()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test_size, random_state = seed, stratify=labels)

    rf_classifier, default_rf_score, tuned_rf_score, cnf_rf_matrix = model_utils.tune_classifier(rf_classifier, rfParameters, X_train, X_test, y_train, y_test)
    ada_classifier, default_ada_score, tuned_ada_score, cnf_ada_matrix = model_utils.tune_classifier(ada_classifier, adaParameters,  X_train, X_test, y_train, y_test)
    gauss_classifier, default_gauss_score, tuned_gauss_score, cnf_gauss_matrix = model_utils.tune_classifier(gauss_classifier, gaussParameters,  X_train, X_test, y_train, y_test)

    sclf_two, sclf_score = model_utils.get_stack_two(rf_classifier, ada_classifier, X_train, X_test, y_train, y_test, seed)
    sclf_all, sclf_all_score = model_utils.get_stack_all(rf_classifier, ada_classifier, gauss_classifier, X_train, X_test, y_train, y_test, seed)

    return tuned_rf_score, tuned_ada_score, tuned_gauss_score, sclf_score, sclf_all_score


rfParameters = {
  'criterion':['gini', 'entropy'],
  'max_depth':[5, 10, 15, 30],
  'max_features':['auto', 'sqrt', 'log2', None],
  'class_weight': ['balanced', 'balanced_subsample'],
}

adaParameters = {
  'learning_rate':[0.1, 0.5, 1],
  'algorithm' :['SAMME', 'SAMME.R']
}

gaussParameters = {
  'priors':[None],
  'var_smoothing' :[1e-09]
}

with open('config.json') as json_data_file:
    config = json.load(json_data_file)
seeds = config['seeds']
test_sizes = config['test_sizes']
impute = config['impute']

pipeline_results = pd.DataFrame(columns=[
    'seed',
    'test_size',
    'impute',
    'TunedRandomForestClassifier',
    'TunedAdaBoostClassifier',
    'TunedGaussClassifier',
    'StackingTwoBest',
    'StackAll'
    ])

scores = pd.DataFrame(columns=['Model', 'Score'])

print("get_clean_dataset")
clean_dataset("datasets/dataset_clean.csv")

print("preprocess_dataset")
preprocess_dataset("datasets/dataset_clean.csv", "datasets/encoded_dataset.csv")

print("impute_values")
data_utils.impute_values("datasets/dataset_clean.csv", "datasets/dataset_clean_imputed.csv")
print("preprocess_dataset_imputed")
dataset_empt = preprocess_dataset("datasets/dataset_clean_imputed.csv", "datasets/encoded_imput_dataset.csv")


for seed in seeds:
    for test_size in test_sizes:
        for imput in impute:
            print("seed: %f, test_size: %f, imput: %d"%(seed, test_size, imput))
            start_time = time.time()
            tuned_rf_score, tuned_ada_score, tuned_gauss_score, sclf_two, sclf_all = run_pipeline(seed, test_size, imput)

            pipeline_results = pipeline_results.append(
                {
                    'seed': seed,
                    'test_size': test_size,
                    'impute': imput,
                    'TunedRandomForestClassifier': tuned_rf_score,
                    'TunedAdaBoostClassifier': tuned_ada_score,
                    'TunedGaussClassifier':tuned_gauss_score,
                    'StackingTwoBest': sclf_two,
                    'StackAll': sclf_all
                }, ignore_index=True)
            print("--- %s seconds ---" % (time.time() - start_time))
print('pipeline_results')

print(pipeline_results)

scores = scores.append({'Model': 'TunedRandomForestClassifier', 'Score': pipeline_results[['TunedRandomForestClassifier']].max()['TunedRandomForestClassifier']}, ignore_index=True)
scores = scores.append({'Model': 'TunedAdaBoostClassifier', 'Score': pipeline_results[['TunedAdaBoostClassifier']].max()['TunedAdaBoostClassifier']}, ignore_index=True)
scores = scores.append({'Model': 'StackingTwoBest', 'Score': pipeline_results[['StackingTwoBest']].max()['StackingTwoBest']}, ignore_index=True)
scores = scores.append({'Model': 'StackAll', 'Score': pipeline_results[['StackAll']].max()['StackAll']}, ignore_index=True)

scores = scores.sort_values('Score', ascending = False)
print('scores')
print(scores)

max_score = scores.iloc[0]
best_config = pipeline_results.loc[pipeline_results[max_score['Model']] == max_score['Score']]

print('best_config')
print(best_config[['seed', 'test_size', 'impute']])

best_config[['seed', 'test_size', 'impute']].to_json("best_config.json", orient='records')
