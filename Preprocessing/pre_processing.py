from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

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


# sns.heatmap(training_data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
# fig = plt.gcf()
# fig.set_size_inches(25, 25)
# plt.show()
df = training_data[training_data['unit_number'] == 1]
x = df['RUL'].to_numpy()
y = df['Ps30'].to_numpy()
xnew = np.linspace(0, 185, num=41, endpoint=True)

# Define interpolators.
f_linear = interp1d(x, y)
f_cubic = interp1d(x, y, kind='cubic')
plt.plot(x, y, 'o', label='Ps30')
plt.plot(xnew, f_linear(xnew), '-', label='linear')
plt.plot(xnew, f_cubic(xnew), '--', label='cubic')
plt.legend(loc='best')
plt.show()


# plt.figure(figsize = (16, 21))
#
# for i in range(21):
#     temp_data = training_data.iloc[:,i+5]
#     plt.subplot(7,3,i+1)
#     plt.boxplot(temp_data)
#     plt.title("Sensor " + str(i+1) + ", column "+ str(i+6))
# plt.show()

