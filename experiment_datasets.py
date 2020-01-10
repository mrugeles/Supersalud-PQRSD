import data_utils
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def to_year(value_range):
    if(value_range == 'no_cie10'):
        return -1
    
    num = int(value_range[:3])
    unit = value_range[-1]
    
    map_unit = {
        'A': num,
        'M': num / 12,
        'D': num / 365,
        'H': num / 8760
    }
    return int(map_unit[unit])

def get_edad_inf(value):
    value_range = {
        '999': -1,
        'de 0 a 5 años': 0,
        'de 13 a 17 años': 13,
        'de 18 a 24 años': 18,
        'de 25 a 29 años': 25,
        'de 30 a 37 años': 30,
        'de 38 a 49 años': 38,
        'de 50 a 62 años': 50,
        'de 6 a 12 años': 6,
        'mayor de 63 años': 63
            }
    return value_range.get(value, -1)

def get_edad_sup(value):
    value_range = {
        '999': -1,
        'de 0 a 5 años': 5,
        'de 13 a 17 años': 17,
        'de 18 a 24 años': 24,
        'de 25 a 29 años': 29,
        'de 30 a 37 años': 37,
        'de 38 a 49 años': 49,
        'de 50 a 62 años': 62,
        'de 6 a 12 años': 12,
        'mayor de 63 años': 120
            }
    return value_range.get(value, -1)

def in_range(row):
    return row['AFEC_EDADR_INF'] >= row['LIMITE_INFERIOR_EDAD_Y'] and  row['AFEC_EDADR_SUP'] < row['LIMITE_SUPERIOR_EDAD_Y']

def cie10_sexo(value):
    if(value == 1 or value == 2):
        return True
    return False


def label_encode(dataset):
    le = preprocessing.LabelEncoder()
    for column in dataset.columns:
        le.fit(dataset[column])
        dataset[column] = le.transform(dataset[column]) 
    return dataset

def naive(dataset):
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = dataset.sample(frac=0.1, random_state=1)
    dataset = dataset.drop(['PQR_ESTADO'], axis = 1)
    dataset = label_encode(dataset)
    dataset.to_csv("datasets/experiments/naive.csv", index = False)

def basic(dataset):
    dataset = data_utils.clean_afec_dpto(dataset)
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = data_utils.clean_cie_10(dataset)
    dataset = data_utils.remove_features(dataset)
    dataset = dataset.reset_index()
    dataset.drop(['index'], axis = 1)
    dataset = label_encode(dataset)
    dataset.to_csv("datasets/experiments/basic.csv", index = False)

def missing_state(dataset):
    dataset = data_utils.clean_afec_dpto(dataset)
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = data_utils.clean_cie_10(dataset)
    dataset = data_utils.remove_features(dataset)
    dataset = dataset.reset_index()
    dataset = dataset.drop(['index'], axis = 1)

    zero_values = set(dataset.columns[dataset.eq('0').mean() > 0])
    for feature in zero_values:
        dataset[f'{feature}_is_missing'] = dataset[feature].apply(lambda f: 1 if f == '0' else 0)

    features_columns = [column for column in dataset.columns if '_is_missing' not in column]
    
    dataset[features_columns] = label_encode(dataset[features_columns])
    dataset.to_csv("datasets/experiments/missing_state.csv", index = False)

def missing_state_remove_75_percent(dataset):
    dataset = data_utils.clean_afec_dpto(dataset)
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = data_utils.clean_cie_10(dataset)
    dataset = data_utils.remove_features(dataset)
    dataset = dataset.reset_index()
    dataset = dataset.drop(['index'], axis = 1)

    zero_values = set(dataset.columns[dataset.eq('0').mean() > .75])
    dataset = dataset.drop(zero_values, axis = 1)

    zero_values = set(dataset.columns[dataset.eq('0').mean() > 0])
    for feature in zero_values:
        dataset[f'{feature}_is_missing'] = dataset[feature].apply(lambda f: 1 if f == '0' else 0)

    features_columns = [column for column in dataset.columns if '_is_missing' not in column]
    dataset[features_columns] = label_encode(dataset[features_columns])
    dataset.to_csv("datasets/experiments/missing_state_remove_75_percent.csv", index = False)

def missing_state_and_imputing(dataset):
    dataset = data_utils.clean_afec_dpto(dataset)
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = data_utils.clean_cie_10(dataset)
    dataset = data_utils.remove_features(dataset)
    dataset = dataset.reset_index()
    dataset = dataset.drop(['index'], axis = 1)

    imp = SimpleImputer(missing_values = '0', strategy="most_frequent")

    zero_values = list(dataset.columns[dataset.eq('0').mean().mean() > 0])[0]
    print(zero_values)
    for feature in zero_values:
        print(feature)
        dataset[f'{feature}_is_missing'] = dataset[feature].apply(lambda value: 0 if value == '0' else 1)
        
    dataset[zero_values] = imp.fit_transform(dataset[zero_values])

    dtypes_df = dataset.dtypes.to_frame(name = 'dtype')
    dtypes_df = dtypes_df[dtypes_df['dtype'] == 'object']
    non_numeric_features = list(dtypes_df.index)

    dataset[non_numeric_features] = dataset[non_numeric_features].applymap(str)
    dataset[non_numeric_features] = label_encode(dataset[non_numeric_features])
    dataset.to_csv("datasets/experiments/missing_state_and_imputing.csv", index = False)

def imputing(dataset):
    dataset = data_utils.clean_afec_dpto(dataset)
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = data_utils.clean_cie_10(dataset)
    dataset = data_utils.remove_features(dataset)
    dataset = dataset.reset_index()
    dataset = dataset.drop(['index'], axis = 1)

    zero_values = list(dataset.columns[dataset.eq('0').mean() > 0])
    dataset[zero_values] = dataset[zero_values].apply(lambda col: col.fillna(col.mode()[0]), axis=0)
    dataset = dataset.applymap(str)
    dataset = label_encode(dataset)

    dataset.to_csv("datasets/experiments/imputing.csv", index = False)

def normalizing(dataset):
    dataset = data_utils.clean_afec_dpto(dataset)
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = data_utils.clean_cie_10(dataset)
    dataset = data_utils.remove_features(dataset)
    dataset = dataset.reset_index()
    dataset = dataset.drop(['index'], axis = 1)

    dataset = label_encode(dataset)

    scaler = MinMaxScaler() 

    features = dataset.drop(['RIESGO_VIDA'], axis = 1)
    labels = dataset[['RIESGO_VIDA']]

    features[features.columns] = features[features.columns].apply(lambda x: np.log(x + 1))
    features[features.columns] = scaler.fit_transform(features[features.columns])

    dataset = features
    dataset['RIESGO_VIDA'] = labels.values
    dataset.to_csv("datasets/experiments/normalizing.csv", index = False)

def target_encoder(dataset):
    dataset = data_utils.clean_afec_dpto(dataset)
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = data_utils.clean_cie_10(dataset)
    dataset = data_utils.remove_features(dataset)
    dataset = dataset.reset_index()
    dataset = dataset.drop(['index'], axis = 1)

    labels = dataset[['RIESGO_VIDA']]
    features = dataset.drop(['RIESGO_VIDA'], axis = 1)

    encoded_features = data_utils.encode_features(features, labels)

    encoded_features['RIESGO_VIDA'] = labels
    encoded_features.to_csv("datasets/experiments/target_encoder.csv", index = False)

def target_encoder_only_complains(dataset):
    dataset = dataset[
        (dataset['PQR_TIPOPETICION'] != 'peticion de informacion') &
        (dataset['PQR_TIPOPETICION'] != 'consulta y/o solicitud de informacion') 
        ]

    dataset = data_utils.clean_afec_dpto(dataset)
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = data_utils.clean_cie_10(dataset)
    dataset = data_utils.remove_features(dataset)
    dataset = dataset.reset_index()
    dataset = dataset.drop(['index'], axis = 1)

    labels = dataset[['RIESGO_VIDA']]
    features = dataset.drop(['RIESGO_VIDA'], axis = 1)

    encoded_features = data_utils.encode_features(features, labels)

    encoded_features['RIESGO_VIDA'] = labels
    encoded_features.to_csv("datasets/experiments/target_encoder_only_complains.csv", index = False)    

def cie10(dataset):
    cie10_df = pd.read_csv('datasets/CIE10.csv', sep = ';')
    cie10_df['DESCRIPCION_COD_CIE_10_04'] = cie10_df['DESCRIPCION_COD_CIE_10_04'].apply(lambda value: value.lower())
    dataset = pd.merge(left = dataset, right = cie10_df, how = 'left', left_on='CIE_10', right_on='DESCRIPCION_COD_CIE_10_04')

    dataset = dataset.drop(['CIE_10', 'NOMBRE_CAPITULO', 'DESCRIPCION_COD_CIE_10_03', 'DESCRIPCION_COD_CIE_10_04'], axis = 1)

    cie10_columns = [
        'CAPITULO', 
        'COD_CIE_10_03', 
        'COD_CIE_10_04', 
        'SEXO', 
        'LIMITE_INFERIOR_EDAD', 
        'LIMITE_SUPERIOR_EDAD']

    dataset[cie10_columns] = dataset[cie10_columns].replace(np.nan, 'no_cie10', regex=True)
    dataset = dataset[dataset['CAPITULO'] != 'no_cie10']

    dataset['CIE10_SEXO'] = dataset['SEXO'].apply(cie10_sexo)
    dataset['LIMITE_INFERIOR_EDAD_Y'] = dataset['LIMITE_INFERIOR_EDAD'].apply(to_year)
    dataset['LIMITE_SUPERIOR_EDAD_Y'] = dataset['LIMITE_SUPERIOR_EDAD'].apply(to_year)
    dataset['AFEC_EDADR_INF'] = dataset['AFEC_EDADR'].apply(get_edad_inf)
    dataset['AFEC_EDADR_SUP'] = dataset['AFEC_EDADR'].apply(get_edad_sup)
    dataset['CIE10_RANGO_EDAD'] = dataset.apply(in_range, axis=1)

    dataset = dataset.drop(
        [
            'SEXO',
            'LIMITE_INFERIOR_EDAD', 
            'LIMITE_SUPERIOR_EDAD', 
            'LIMITE_INFERIOR_EDAD_Y', 
            'LIMITE_SUPERIOR_EDAD_Y',
            'AFEC_EDADR'

        ], axis = 1
    )


    dataset[
        [
            'AFEC_GENERO',
            'CIE10_SEXO',
            'CIE10_RANGO_EDAD',
            'AFEC_EDADR_INF', 
            'AFEC_EDADR_SUP',

        ]
    ].head()

    dataset = data_utils.clean_afec_dpto(dataset)
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = data_utils.remove_features(dataset)
    dataset = dataset.reset_index()
    dataset = dataset.drop(['index'], axis = 1)

    labels = dataset[['RIESGO_VIDA']]
    features = dataset.drop(['RIESGO_VIDA'], axis = 1)

    encoded_features = data_utils.encode_features(features, labels)

    encoded_features['RIESGO_VIDA'] = labels

    encoded_features.to_csv("datasets/experiments/cie10.csv", index = False)

def cie10_only_complains(dataset):
    dataset = dataset[
        (dataset['PQR_TIPOPETICION'] != 'peticion de informacion') &
        (dataset['PQR_TIPOPETICION'] != 'consulta y/o solicitud de informacion') 
        ]
    cie10_df = pd.read_csv('datasets/CIE10.csv', sep = ';')
    cie10_df['DESCRIPCION_COD_CIE_10_04'] = cie10_df['DESCRIPCION_COD_CIE_10_04'].apply(lambda value: value.lower())
    dataset = pd.merge(left = dataset, right = cie10_df, how = 'left', left_on='CIE_10', right_on='DESCRIPCION_COD_CIE_10_04')

    dataset = dataset.drop(['CIE_10', 'NOMBRE_CAPITULO', 'DESCRIPCION_COD_CIE_10_03', 'DESCRIPCION_COD_CIE_10_04'], axis = 1)

    cie10_columns = [
        'CAPITULO', 
        'COD_CIE_10_03', 
        'COD_CIE_10_04', 
        'SEXO', 
        'LIMITE_INFERIOR_EDAD', 
        'LIMITE_SUPERIOR_EDAD']

    dataset[cie10_columns] = dataset[cie10_columns].replace(np.nan, 'no_cie10', regex=True)
    dataset = dataset[dataset['CAPITULO'] != 'no_cie10']

    dataset['CIE10_SEXO'] = dataset['SEXO'].apply(cie10_sexo)
    dataset['LIMITE_INFERIOR_EDAD_Y'] = dataset['LIMITE_INFERIOR_EDAD'].apply(to_year)
    dataset['LIMITE_SUPERIOR_EDAD_Y'] = dataset['LIMITE_SUPERIOR_EDAD'].apply(to_year)
    dataset['AFEC_EDADR_INF'] = dataset['AFEC_EDADR'].apply(get_edad_inf)
    dataset['AFEC_EDADR_SUP'] = dataset['AFEC_EDADR'].apply(get_edad_sup)
    dataset['CIE10_RANGO_EDAD'] = dataset.apply(in_range, axis=1)

    dataset = dataset.drop(
        [
            'SEXO',
            'LIMITE_INFERIOR_EDAD', 
            'LIMITE_SUPERIOR_EDAD', 
            'LIMITE_INFERIOR_EDAD_Y', 
            'LIMITE_SUPERIOR_EDAD_Y',
            'AFEC_EDADR'

        ], axis = 1
    )


    dataset[
        [
            'AFEC_GENERO',
            'CIE10_SEXO',
            'CIE10_RANGO_EDAD',
            'AFEC_EDADR_INF', 
            'AFEC_EDADR_SUP',

        ]
    ].head()

    dataset = data_utils.clean_afec_dpto(dataset)
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = data_utils.remove_features(dataset)
    dataset = dataset.reset_index()
    dataset = dataset.drop(['index'], axis = 1)

    labels = dataset[['RIESGO_VIDA']]
    features = dataset.drop(['RIESGO_VIDA'], axis = 1)

    encoded_features = data_utils.encode_features(features, labels)

    encoded_features['RIESGO_VIDA'] = labels

    encoded_features.to_csv("datasets/experiments/cie10_only_complains.csv", index = False)

def contains(value, list_values):
    total = [key_word for key_word in list_values if key_word in value] 
    return len(total) > 0 

def risk_cases_encoder(dataset):
    dataset = data_utils.clean_afec_dpto(dataset)
    dataset = data_utils.clean_riesgo_vida(dataset)
    dataset = data_utils.clean_cie_10(dataset)

    mot_esp_cases = ['referencia', 'contra_referencia', 'urgencias', 'entrega de medicamentos', 'citas de consulta medica especializada', 'procedimientos y/o servicios', 'enfermedades raras o hu']
    afec_edad_cases = ['de 6 a 12 años', 'de 0 a 5 años', 'de 13 a 17 años', 'mayor de 63 años']
    cie10_cases = ['vih', 'tumores malignos', 'maternas', 'trasplantados']
    
    dataset['CASO_RIESGO'] = dataset['MOTIVO_ESPECIFICO'].apply(lambda value: contains(value, mot_esp_cases)) 
    dataset['POBESPECIAL'] = dataset['AFEC_POBESPECIAL'].apply(lambda value: False if value == 'no aplica' else True) 
    dataset['EDAD_RIESGO'] = dataset['AFEC_EDADR'].apply(lambda value: contains(value, afec_edad_cases)) 
    dataset['CIE10_RIESGO'] = dataset['CIE_10'].apply(lambda value: contains(value, cie10_cases)) 

    dataset = data_utils.remove_features(dataset)
    dataset = dataset.reset_index()
    dataset = dataset.drop(['index'], axis = 1)

    labels = dataset[['RIESGO_VIDA']]
    features = dataset.drop(['RIESGO_VIDA'], axis = 1)

    encoded_features = data_utils.encode_features(features, labels)

    encoded_features['RIESGO_VIDA'] = labels
    encoded_features.to_csv("datasets/experiments/risk_cases_encoder.csv", index = False)

experiment = {
    'naive': naive,
    'basic': basic,
    'missing_state': missing_state,
    'missing_state_remove_75_percent': missing_state_remove_75_percent,
    'missing_state_and_imputing': missing_state_and_imputing,
    'imputing': imputing,
    'normalizing': normalizing,
    'target_encoder': target_encoder,
    'cie10': cie10,
    'target_encoder_only_complains': target_encoder_only_complains,
    'cie10_only_complains': cie10_only_complains

}
