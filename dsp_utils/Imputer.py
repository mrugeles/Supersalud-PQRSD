from enum import Enum
import pandas as pd
#from Model import Model

class ImputerStrategy(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    MODE = 'mode'
    CONSTANT = 'constant'
    REGRESSOR_MODEL = 'regressor_model'
    CLASSIFICATION_MODEL = 'clasification_model'

# Delete features with more of 70% of missing values
# Delete rows with more of 70% of missing values
#Get empty/nulls/zero percentages per feature
    # Impute numeric features
    # Impute categorical features
# Drop duplicated data

class Imputer():

    #model = Model()

    fill_mean = lambda col: col.fillna(col.mean())
    fill_median = lambda col: col.fillna(col.median())
    fill_mode = lambda col: col.fillna(col.mode()[0])

    impute_strategies = {
        ImputerStrategy.MEAN: fill_mean,
        ImputerStrategy.MEDIAN: fill_median,
        ImputerStrategy.MODE: fill_mode

    }


    def impute(self, dataset, impute_strategy):
        if impute_strategy in [ImputerStrategy.MEAN, ImputerStrategy.MEDIAN, ImputerStrategy.MODE]:
            return dataset.apply(self.impute_strategies[impute_strategy], axis=0)
        else:
            return dataset

    def impute_grouped(self, dataset, target_feature, features, impute_strategy):
        dataset[target_feature] = dataset.groupby(features)[target_feature].transform(fill_mean)
        return dataset

    def impute_ordered(self, dataset, feature, grouped_features, impute_strategy):
        return None

    '''
    def regressor_imputer(self, dataset, label):
        #get features with 100% of values
        not_null_features = list(dataset.columns[dataset.isnull().mean()==0])
        if(len(not_null_features)):
            features = not_null_features.copy()
            features.append(label)
            dataset = dataset[features]
            #get rows with not null targets for training
            training_dataset = dataset.loc[dataset[label].isnull() == False]
            #get rows with null values for validation / imputing
            validation_dataset = dataset.loc[dataset[label].isnull()]
            #split training and test data
            X = dataset[not_null_features]
            y = dataset[label]
            #train models
            dfResults = self.model.train_models(self.model.regressors)
            print(dfResults)
            #pick top 3 models
            #tune top 3 models
            #stack 2 best tuned models
            #stack 3 tuned models
            #return best model
        return None
    '''


    def label_encode(self, values):
        """Encode categorical features with LabelEncoder"""
        from sklearn import preprocessing

        enc = preprocessing.LabelEncoder().fit(values)
        encoded_features = enc.transform(values)
        return encoded_features

    def encode_categorical_features(self, df):
        dtypes_df = df.dtypes.to_frame(name = "dtype")
        category_features = list(dtypes_df[dtypes_df['dtype'] == 'object'].index)
        for category in category_features:
            df[category] = self.label_encode(df[category])
    
    def impute_half_missing(self, df, field, impute_field, default_value):
        df[impute_field] = df[field].apply(lambda x: 1 if pd.isnull(x) == True else 0)
        df[field] = df[field].apply(lambda x: default_value if pd.isnull(x) == True else x)