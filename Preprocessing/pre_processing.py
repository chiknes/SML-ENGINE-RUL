import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
import math
import xgboost
import time
from tqdm import tqdm

data = pd.read_csv('../CMAPSSData/train_FD001.txt', sep=' ', header=None)
test_data = pd.read_csv('../CMAPSSData/test_FD001.txt', sep=' ', header=None)

# NaN values only in these 2 columns
data.drop(columns=[26, 27], inplace=True)
test_data.drop(columns=[26, 27], inplace=True)
columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3']

for i in range(1, 22):
    columns.append('sensor_' + str(i))

data.columns = columns
test_data.columns = columns


def prepare_training_data(data, factor=0):
    df = data.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number', 'max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'], inplace=True)

    return df[df['time_in_cycles'] > factor]


training_data = prepare_training_data(data)
# print(df)
# sns.heatmap(training_data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
# fig = plt.gcf()
# fig.set_size_inches(25, 25)
# plt.show()

X_train = training_data.iloc[:, 0:26]
# print(X_train)
y_train = training_data['RUL']
lm = RandomForestRegressor(n_estimators=70, max_depth=5, n_jobs=-1, random_state=1)
rfe = RFE(lm)  # running RFE
rfe = rfe.fit(X_train, y_train)
print(rfe.support_)  # Printing the boolean results
print(rfe.ranking_)
print(rfe.estimator_)
print(rfe.n_features_)

plt.figure(figsize = (16, 21))

for i in range(21):
    temp_data = training_data.iloc[:,i+5]
    plt.subplot(7,3,i+1)
    plt.boxplot(temp_data)
    plt.title("Sensor " + str(i+1) + ", column "+ str(i+6))
# plt.show()


def train_models(data, model='FOREST'):
    X = data.iloc[:, :14].to_numpy()
    Y = data.iloc[:, 14:].to_numpy()
    Y = np.ravel(Y)
    if model == 'FOREST':
        model = RandomForestRegressor(n_estimators=70, max_features=12, max_depth=5, n_jobs=-1, random_state=1)
        model.fit(X, Y)
        return model
    elif model == 'XGB':
        model = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.018, gamma=0, subsample=0.8,
                                     colsample_bytree=0.5, max_depth=3, silent=True)
        model.fit(X, Y)
        return model
    return


def plot_result(y_true, y_pred):
    rcParams['figure.figsize'] = 12, 10
    plt.plot(y_pred)
    plt.plot(y_true)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel('RUL')
    plt.xlabel('training samples')
    plt.legend(('Predicted', 'True'), loc='upper right')
    plt.show()
    return

# model_1 = train_models(training_data)
