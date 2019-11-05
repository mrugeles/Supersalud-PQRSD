from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

import pandas as pd

class Model():

    regressors_parameters = {
        'GradientBoostingRegressor': {
            'loss': ['ls', 'lad', 'huber', 'quantile'],
            'learning_rate':[0.1, 0.001],
            'n_estimators':[100, 150, 200, 250],
            'subsample':[0.5, 1, 1.5, 2],
            'criterion':['friedman_mse', 'mse', 'mae'],
            'min_samples_split':[2, 3, 4],
            'random_state':[100, 200]
        },
        'GaussianProcessRegressor': {
            'normalize_y':[True, False],
            'random_state':[100, 200]
        },
        'ARDRegression': {
            'n_iter':[200, 300, 400],
            'compute_score':[True, False],
            'fit_intercept':[True, False],
            'normalize':[True, False]
        } ,
        'LinearRegression': {
        'fit_intercept':[True, False],
        'normalize':[True, False]
        },
    }

    regressors = None

    def __init__(self):
        self.init_regressors()

    def init_regressors(self):
        self.regressors = {
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'GaussianProcessRegressor': GaussianProcessRegressor(),
            'ARDRegression': ARDRegression(),
            'LinearRegression': LinearRegression(),
        }

    ###
    #      This method trains a model.
    #
    #      Args:
    #       learner (classifier / regressor): Model to train.
    #       scorer (function): score function.
    #       X_train: (dataset): Features dataset to train the model.
    #       y_train: (dataset): Targe feature dataset to train the model.
    #       X_test: (dataset): Features dataset to test the model.
    #       y_test: (dataset): Targe feature dataset to test the model.
    #      Returns:
    #       learner (classifier / regressor): trained model.
    #       dfResults (dataset): Dataset with information about the trained model.
    ###

    def train_predict(self, learner, scorer, X_train, y_train, X_test, y_test):
        start = time()
        learner = learner.fit(X_train, y_train)
        end = time()

        train_time = end - start

        start = time()
        predictions_test = learner.predict(X_test)
        predictions_train = learner.predict(X_train)
        end = time() # Get end time

        pred_time = end - start

        f_train = scorer(y_train, predictions_train)
        f_test =  scorer(y_test, predictions_test)

        return learner

    def train_models(self, learners_dict, scorer, X_train, y_train, X_test, y_test):
        dfResults = pd.DataFrame(columns=['learner', 'train_time', 'pred_time', 'f_test', 'f_train'])
        print(learners_dict)
        for k, learner_name in enumerate(learners_dict):
            learner = learners_dict[learner_name]
            learner = self.train_predict(learner, scorer, X_train, y_train, X_test, y_test, dfResults)
            dfResults = dfResults.append({'learner': learner, 'train_time': train_time, 'pred_time': pred_time, 'f_test': f_test, 'f_train':f_train}, ignore_index=True)

        return dfResults.sort_values(by=['f_test'], ascending = False)

    def get_best_tuned_model(self, learners_dict, learners_parameters, scorer, X_train, X_test, y_train, y_test):
        best_score = None
        best_learner = None
        for k, learner_name in enumerate(learners_dict):
            learner = learners_dict[learner_name]
            parameters = learners_parameters[learner_name]
            learner, default_score, tuned_score = this.tune_learner(learner, parameters, scorer, X_train, X_test, y_train, y_test)
            if(best_score is None or tuned_score > best_score):
                best_score = tuned_score
                best_learner = learner
        return best_learner


    ###
    #      This method use grid search to tune a learner.
    #
    #      Args:
    #       learner (classifier / regressor): learner to tune.
    #       parameters (dict): learner parameters.
    #       X_train: (dataset): Features dataset to train the model.
    #       y_train: (dataset): Targe feature dataset to train the model.
    #       X_test: (dataset): Features dataset to test the model.
    #       y_test: (dataset): Targe feature dataset to test the model.
    #      Returns:
    #       best_learner (classifier / regressor)): Classifier with the best score.
    #       default_score (float): Classifier score before being tuned.
    #       tuned_score (float): Classifier score after being tuned.
    #       cnf_matrix (float): Confusion matrix.
    ###
    def tune_learner(learner, parameters, scorer, X_train, X_test, y_train, y_test):

      c, r = y_train.shape
      labels = y_train.values.reshape(c,)

      grid_obj = GridSearchCV(learner, param_grid=parameters,  scoring=scorer, iid=False)
      grid_fit = grid_obj.fit(X_train, labels)
      best_learner = grid_fit.best_estimator_
      predictions = (learner.fit(X_train, labels)).predict(X_test)
      best_predictions = best_learner.predict(X_test)

      default_score = scorer(y_test, predictions)
      tuned_score = scorer(y_test, best_predictions)

      return best_learner, default_score, tuned_score
