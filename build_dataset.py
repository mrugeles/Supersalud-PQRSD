import data_utils

dataset = data_utils.get_dataset()
print(f'dataset.shape: {dataset.shape}')

dataset.to_csv('datasets/dataset.csv', index = False)