import data_utils_v2 as data_utils
import pandas as pd



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
    return value_range[value]

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
    return value_range[value]

def in_range(row):
    return row['AFEC_EDADR_INF'] >= row['LIMITE_INFERIOR_EDAD_Y'] and  row['AFEC_EDADR_SUP'] < row['LIMITE_SUPERIOR_EDAD_Y']

def cie10_sexo(value):
    if(value == 1 or value == 2):
        return True
    return False




print(f'Joining pqrd datasets...')
dataset = data_utils.get_dataset()
dataset.to_csv('datasets/dataset.csv', index = False)

print(f'Merging with CIE10 dataset...')
cie10_df = pd.read_csv('datasets/CIE10.csv', sep = ';')
cie10_df['DESCRIPCION_COD_CIE_10_04'] = cie10_df['DESCRIPCION_COD_CIE_10_04'].apply(lambda value: value.lower())
dataset_cie10 = pd.merge(left = dataset, right = cie10_df, how = 'left', left_on='CIE_10', right_on='DESCRIPCION_COD_CIE_10_04')


print(f'dataset_cie10.shape: {dataset_cie10.shape}')

dataset_cie10.to_csv('datasets/dataset_cie10.csv', index = False)
