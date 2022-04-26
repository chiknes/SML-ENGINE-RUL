from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[1]

data = pd.read_csv(PROJECT_ROOT / 'CMAPSSData/train_FD001.txt', sep=' ', header=None)
test_data = pd.read_csv(PROJECT_ROOT / 'CMAPSSData/test_FD001.txt', sep=' ', header=None)
RUL = pd.read_csv(PROJECT_ROOT / 'CMAPSSData/RUL_FD001.txt', sep=' ', header=None)

# NaN values only in these 2 columns
data.drop(columns=[26, 27], inplace=True)
test_data.drop(columns=[26, 27], inplace=True)
# columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3']
#
# for i in range(1, 22):
#     columns.append('sensor_' + str(i))

columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'TRA', 'T2', 'T24', 'T30', 'T50', 'P2', 'P15',
           'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
           'PCNfR_dmd', 'W31', 'W32']
data.columns = columns
test_data.columns = columns


def prepare_training_data(dataset, factor=0):
    df = dataset.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number', 'max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'], inplace=True)
    return df[df['time_in_cycles'] > factor]


training_data = prepare_training_data(data)
print(training_data.head(10))
