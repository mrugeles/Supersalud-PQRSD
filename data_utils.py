# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from IPython.display import display
from category_encoders import *
import time

def get_pqrd_dataset():
    """Build a pandas dataframe from PQRD files."""
    #
    data_2017 = pd.read_csv("datasets/Base_De_Datos_PQRD_2017.csv", dtype='str')
    data_2017['year'] = 2017
    data_2016 = pd.read_csv("datasets/Base_De_Datos_PQRD_2016.csv", dtype='str')
    data_2016['year'] = 2016
    data_2015 = pd.read_csv("datasets/Base_De_Datos_PQRD_2015.csv", dtype='str')
    data_2015['year'] = 2015


    # Every year some features can be added or removed. So before perform a union we first take all the columns in common.
    colums_2017 = data_2017.columns.values
    colums_2016 = data_2016.columns.values
    colums_2015 = data_2015.columns.values
    ds_columns = reduce(np.intersect1d, (colums_2017, colums_2016, colums_2015))

    #Unifying datasets.
    dataset = data_2017[ds_columns]
    dataset = dataset.append(data_2016[ds_columns])
    dataset = dataset.append(data_2015[ds_columns])

    #Formating the month fields to MM format.
    dataset = dataset.astype(str)
    dataset['MES'] = dataset['MES'].apply(lambda m: '0' + m if int(m) < 10 else m)


    data_columns = dataset.columns.values.tolist()
    for column in data_columns:
        dataset[column] = dataset[column].apply(lambda s: str(s).lower())

    return dataset

# get dataset
def get_dataset():
    """Build a pandas dataframe from PQRD files and remove the year feature that was used for analysis purposes."""
    dataset = get_pqrd_dataset()
    dataset = dataset.drop(['year'], axis = 1)

    return dataset


def get_pqrd_dataset_null_empty():
    """Build a pandas dataframe from PQRD files."""
    #
    data_2017 = pd.read_csv("datasets/Base_De_Datos_PQRD_2017.csv", dtype='str')
    data_2017['year'] = 2017
    data_2016 = pd.read_csv("datasets/Base_De_Datos_PQRD_2016.csv", dtype='str')
    data_2016['year'] = 2016
    data_2015 = pd.read_csv("datasets/Base_De_Datos_PQRD_2015.csv", dtype='str')
    data_2015['year'] = 2015


    # Every year some features can be added or removed. So before perform a union we first take all the columns in common.
    colums_2017 = data_2017.columns.values
    colums_2016 = data_2016.columns.values
    colums_2015 = data_2015.columns.values
    ds_columns = reduce(np.intersect1d, (colums_2017, colums_2016, colums_2015))

    #Unifying datasets.
    dataset = data_2017[ds_columns]
    dataset = dataset.append(data_2016[ds_columns])
    dataset = dataset.append(data_2015[ds_columns])


    #Formating the month fields to MM format.
    dataset['MES'] = dataset['MES'].apply(lambda m: '0' + str(m) if int(m) < 10 else str(m))


    data_columns = dataset.columns.values.tolist()
    for column in data_columns:
        dataset[column] = dataset[column].apply(lambda s: str(s).lower())

    dataset = dataset.astype(str)
    dataset = dataset.replace('0', np.NaN)
    dataset = dataset.replace('nan', np.NaN)
    dataset = dataset.replace('', np.NaN)
    return dataset

# get dataset
def get_dataset_null_empty():
    """Build a pandas dataframe from PQRD files and remove the year feature that was used for analysis purposes."""
    dataset = get_pqrd_dataset_null_empty()
    dataset = dataset.drop(['year'], axis = 1)

    return dataset

# clean AFEC_DPTO
def clean_afec_dpto(dataset):
    """Fix wrong values for some specific cases."""
    dataset['AFEC_DPTO'] = dataset['AFEC_DPTO'].apply(lambda s: 'san andres' if s == 'archipielago de san andres, providencia y santa catalina' or s == 'san andrés' else s)
    dataset['AFEC_DPTO'] = dataset['AFEC_DPTO'].apply(lambda s: 'bogota d.c.' if s == 'bogota d.c' else s)
    return dataset

# clean RIESGO_VIDA
def clean_riesgo_vida(dataset):
    """Maps RIESGO_VIDA for binary classification"""
    dataset = dataset.loc[(dataset['RIESGO_VIDA'] != '0')]
    dataset['RIESGO_VIDA'] = np.where(dataset['RIESGO_VIDA'] == 'si', 1,0)
    return dataset

# clean CIE_10
def clean_cie_10(dataset):
    """Remove records with no values"""
    return dataset.loc[(dataset['CIE_10'] != 0) & (dataset['CIE_10'] != '0')]

# remove features
def remove_features(dataset):
    """Remove features with no statistical or reduntant values"""
    unused_features = ['IDRANGOEDADES', 'ID_MES', 'PQR_GRUPOALERTA', 'PQR_ESTADO', 'ENT_DPTO', 'ENT_MPIO', 'PET_DPTO', 'MACROMOTIVO', 'MOTIVO_GENERAL', 'MOTIVO_ESPECIFICO']
    return dataset.drop(unused_features, axis = 1)

def impute_values(path, imput_path):
    """Impute categorical features with most frequent values"""
    from sklearn.impute import SimpleImputer
    dataset = pd.read_csv(path)

    col_zero_values = set(dataset.columns[dataset.eq('0').mean() > 0])
    imp = SimpleImputer(missing_values = '0', strategy="most_frequent")
    for feature in col_zero_values:
        dataset[feature] = imp.fit_transform(dataset[[feature]])
    dataset.to_csv(imput_path, index = False)

def encode_features(features, labels):
    """Encode categorical features with TargetEncoder"""

    features_columns = features.columns.values.tolist()

    start_time = time.time()
    enc = TargetEncoder(cols=features_columns, return_df = True).fit(features, labels)
    encoded_features = enc.transform(features)
    print("--- %s seconds ---" % (time.time() - start_time))
    return encoded_features
