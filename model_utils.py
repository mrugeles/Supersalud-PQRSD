import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import StackingClassifier

import itertools

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier 
from sklearn.linear_model import RidgeClassifierCV 
from sklearn.linear_model import SGDClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier 

import xgboost as xgb

def init_classifiers(seed):
    return {
        'AdaBoostClassifier': AdaBoostClassifier(random_state = seed),
        'BaggingClassifier': BaggingClassifier(random_state = seed),
        'ExtraTreesClassifier': ExtraTreesClassifier(random_state = seed),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state = seed),
        'RandomForestClassifier': RandomForestClassifier(random_state = seed),
        'XGBClassifier': xgb.XGBClassifier(),
        'LogisticRegression': LogisticRegression(random_state = seed),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(random_state = seed),
        'RidgeClassifier': RidgeClassifier(random_state = seed),
        'RidgeClassifierCV': RidgeClassifierCV(),
        'SGDClassifier': SGDClassifier(random_state = seed),
        #'KNeighborsClassifier': KNeighborsClassifier(),
        #'RadiusNeighborsClassifier': RadiusNeighborsClassifier(),
        'MLPClassifier': MLPClassifier(random_state = seed),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state = seed),
        'ExtraTreeClassifier': ExtraTreeClassifier(random_state = seed)
    }

###
#      This method trains a classifier with the given beta value and splitted data.
#
#      Args:
#       learner (classifier): Classifier to train.
#       beta_value (float): beta value for score.
#       X_train: (dataset): Features dataset to train the model.
#       y_train: (dataset): Targe feature dataset to train the model.
#       X_test: (dataset): Features dataset to test the model.
#       y_test: (dataset): Targe feature dataset to test the model.
#      Returns:
#       learner (classifier): Classifier to trainself.
#       dfResults (dataset): Dataset with information about the trained model.
###

def train_predict(learner, beta_value, X_train, y_train, X_test, y_test, dfResults):
    start = time()
    learner = learner.fit(X_train, y_train)
    end = time()

    train_time = end - start

    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time() # Get end time

    pred_time = end - start

    f_train = fbeta_score(y_train, predictions_train, beta_value)

    f_test =  fbeta_score(y_test, predictions_test, beta_value)

    print("%s trained." % (learner.__class__.__name__))

    dfResults = dfResults.append({'learner': learner.__class__.__name__, 'train_time': train_time, 'pred_time': pred_time, 'f_test': f_test, 'f_train':f_train}, ignore_index=True)
    return learner, dfResults

###
#      This method use grid search to tune a classifier.
#
#      Args:
#       clf (classifier): Classifier to tune.
#       parameters (dict): Clasiffier parameters.
#       X_train: (dataset): Features dataset to train the model.
#       y_train: (dataset): Targe feature dataset to train the model.
#       X_test: (dataset): Features dataset to test the model.
#       y_test: (dataset): Targe feature dataset to test the model.
#      Returns:
#       best_clf (classifier): Classifier with the best score.
#       default_score (float): Classifier score before being tuned.
#       tuned_score (float): Classifier score after being tuned.
#       cnf_matrix (float): Confusion matrix.
###
def tune_classifier(clf, parameters, X_train, X_test, y_train, y_test):

  from sklearn.metrics import make_scorer
  from sklearn.model_selection import GridSearchCV
  from sklearn.ensemble import ExtraTreesClassifier

  c, r = y_train.shape
  labels = y_train.values.reshape(c,)

  scorer = make_scorer(fbeta_score, beta=2)
  grid_obj = GridSearchCV(clf, param_grid=parameters,  n_jobs = 16, scoring=scorer, iid=False)
  grid_fit = grid_obj.fit(X_train, labels)
  best_clf = grid_fit.best_estimator_
  predictions = (clf.fit(X_train, labels)).predict(X_test)
  best_predictions = best_clf.predict(X_test)

  default_score = fbeta_score(y_test, predictions, beta = 2)
  tuned_score = fbeta_score(y_test, best_predictions, beta = 2)

  cnf_matrix = confusion_matrix(y_test, best_predictions)

  return best_clf, default_score, tuned_score, cnf_matrix

def get_stack_two(clf_1, clf_2, X_train, X_test, y_train, y_test, seed):
    sclf = StackingClassifier(classifiers=[clf_1, clf_2],
                          use_features_in_secondary = True,
                          meta_classifier=RandomForestClassifier(random_state = seed))
    sclf = sclf.fit(X_train, y_train)
    y_predictions = sclf.predict(X_test)
    return sclf, fbeta_score(y_test, y_predictions, beta = 2)

def get_stack_all(clf_1, clf_2, clf_3, X_train, X_test, y_train, y_test, seed):
    sclf = StackingClassifier(classifiers=[clf_1, clf_2, clf_3],
                          use_features_in_secondary = True,
                          meta_classifier=RandomForestClassifier(random_state = seed))
    sclf = sclf.fit(X_train, y_train)
    y_predictions = sclf.predict(X_test)
    return sclf, fbeta_score(y_test, y_predictions, beta = 2)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized confusion matrix"
    else:
        title = 'Confusion matrix, without normalization'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def model_validation(model_path, X_validation, y_validation):
    clf = joblib.load(model_path)

    y_predictions = clf.predict(X_validation)

    print("F-score on validation data: {:.4f}".format(fbeta_score(y_validation, y_predictions, beta = 2)))


    cnf_matrix = confusion_matrix(y_validation, y_predictions)
    plot_confusion_matrix(cnf_matrix, classes=['Life not as risk', 'Life at risk'], normalize = True)
    return clf
