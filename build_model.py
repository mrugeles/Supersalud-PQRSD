import sys
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from IPython.display import display 
from sklearn.metrics import fbeta_score
import model_utils as model_utils
import warnings
warnings.filterwarnings('ignore')

import json
pd.set_option('display.max_colwidth', -1)
def build_model(path_dataset, path_results):
    print("Load dataset...")

    with open('best_config.json') as json_data_file:
        config = json.load(json_data_file)[0]
    print(config)
    seed = int(config['seed'])
    test_size = config['test_size']

    dataset = pd.read_csv(path_dataset)

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


    print("Training models...")
    classifiers = model_utils.init_classifiers(seed)

    # Collect results on the learners
    dfResults = pd.DataFrame(columns=['learner', 'train_time', 'pred_time', 'f_test', 'f_train'])

    for clf in list(classifiers.values()):
        clf_name = clf.__class__.__name__ 
        clf, dfResults = model_utils.train_predict(clf, 2, X_train, y_train, X_test, y_test, dfResults)

    dfResults = dfResults.sort_values(by=['f_test'], ascending = False)
    dfResults.to_csv(path_results, index = False)


if __name__ == '__main__':
    build_model(sys.argv[1], sys.argv[2])
    print('----------------------------------------------------------------------------------------\n')

