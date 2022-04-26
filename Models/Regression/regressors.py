import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost
from pylab import rcParams
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

from Preprocessing.pre_processing import RUL
from Preprocessing.pre_processing import test_data
from Preprocessing.pre_processing import training_data

from sklearn.feature_selection import SelectKBest, chi2



def train_models(dataset, cols, model_type='FOREST'):
    print("\n", model_type)
    X = dataset.iloc[:, :14].to_numpy()
    Y = dataset['RUL'].to_numpy()
    Y = np.ravel(Y)
    if model_type == 'FOREST':
        model = RandomForestRegressor(n_estimators=70, max_depth=9, random_state=1)
        model.fit(X, Y)
        return model
    elif model_type == 'XGB':
        k = 10
        kf = KFold(n_splits=k, random_state=None)
        model = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.015, subsample=0.6,
                                     colsample_bytree=0.9, max_depth=5, max_leaves=7, max_bin=1023,
                                     booster="gbtree")
        acc_score = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = Y[train_index], Y[test_index]

            model.fit(X_train, y_train)
            pred_values = model.predict(X_test)
            # acc = accuracy_score(pred_values, y_test)
            rmse = round(mean_squared_error(y_test, pred_values), 2) ** 0.5
            acc_score.append(rmse)
        avg_acc_score = sum(acc_score) / k
        print('accuracy of each fold - {}'.format(acc_score))
        print('Avg accuracy : {}'.format(avg_acc_score))

        # model.fit(X, Y)
        # return model
    elif model_type == 'LR':
        model = LinearRegression()
        model.fit(X, Y)
        return model
    return


def plot_result(y_true, y_pred):
    rcParams['figure.figsize'] = 12, 10
    plt.plot(y_pred, color="blue")
    plt.plot(y_true, color="red")
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel('RUL')
    plt.xlabel('training samples')
    plt.legend(('Predicted', 'True'), loc='upper right')
    plt.show()
    return


def score_func(y_true, y_pred):
    print(f' mean absolute error {round(mean_absolute_error(y_true, y_pred), 2)}')
    print(f' root mean squared error {round(mean_squared_error(y_true, y_pred), 2) ** 0.5}')
    print(f' R2 score {round(r2_score(y_true, y_pred), 2)}')
    return


training_data.drop(columns=['Nf_dmd', 'PCNfR_dmd', 'P2', 'T2', 'TRA', 'farB', 'epr'], inplace=True)
train_df = training_data.drop(columns=['unit_number', 'setting_1', 'setting_2', 'P15', 'NRc'])
# print(train_df.shape)
sel_cols = [1, 6, 7, 8, 11, 12, 13, 15, 16, 17, 19, 21, 22, 25]

test_max = test_data.groupby('unit_number')['time_in_cycles'].max().reset_index()
test_max.columns = ['unit_number', 'max']
test_data = test_data.merge(test_max, on=['unit_number'], how='left')
test_data.drop(columns=['Nf_dmd', 'PCNfR_dmd', 'P2', 'T2', 'TRA', 'farB', 'epr'], inplace=True)

test_df = test_data[test_data['time_in_cycles'] == test_data['max']].reset_index()
# test_data = test_data.iloc[:, sel_cols]
test_df.drop(columns=['index', 'max', 'unit_number', 'setting_1', 'setting_2', 'P15', 'NRc'],
             inplace=True)
# print(test_data.shape)
test_df = test_df.to_numpy()

y_true = RUL[0].to_numpy()


bestfeatures = SelectKBest(score_func=chi2, k=10)
X = train_df.iloc[:, :14]
fit = bestfeatures.fit(X.to_numpy(), train_df.iloc[:, 14:].to_numpy())
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))

def test_model(train_data, test_dataset, y_true, cols, model):
    model_1 = train_models(train_data, cols, model)
    y_pred = model_1.predict(test_dataset)
    score_func(y_true, y_pred)
    # plot_result(y_true, y_pred)


# test_model(train_df, test_df, y_true, sel_cols, "FOREST")
# test_model(train_df, test_df, y_true, sel_cols, "XGB")
for algo in ['LR', 'XGB', 'FOREST']:
    test_model(train_df, test_df, y_true, sel_cols, algo)

sys.exit()

columns_to_be_dropped = [0, 1, 2, 3, 4]
train_data = pd.read_csv("../../CMAPSSData/train_FD001.txt", sep="\s+", header=None)
test_data = pd.read_csv("../../CMAPSSData/test_FD001.txt", sep="\s+", header=None)
true_rul = pd.read_csv("../../CMAPSSData/RUL_FD001.txt", sep='\s+', header=None)

window_length = 15
shift = 1
early_rul = 150
processed_train_data = []
processed_train_targets = []

# How many test windows to take for each engine. If set to 1 (this is the default), only last window of test data for
# each engine is taken. If set to a different number, that many windows from last are taken.
# Final output is the average output of all windows.
num_test_windows = 5
processed_test_data = []
num_test_windows_list = []

train_data_first_column = train_data[0]
test_data_first_column = test_data[0]

# Scale data for all engines
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data = scaler.fit_transform(train_data.drop(columns=columns_to_be_dropped))
test_data = scaler.transform(test_data.drop(columns=columns_to_be_dropped))

train_data = pd.DataFrame(data=np.c_[train_data_first_column, train_data])
test_data = pd.DataFrame(data=np.c_[test_data_first_column, test_data])

num_train_machines = len(train_data[0].unique())
num_test_machines = len(test_data[0].unique())


def process_targets(data_length, early_rul=None):
    """
    Takes datalength and earlyrul as input and
    creates target rul.
    """
    if early_rul == None:
        return np.arange(data_length - 1, -1, -1)
    else:
        early_rul_duration = data_length - early_rul
        if early_rul_duration <= 0:
            return np.arange(data_length - 1, -1, -1)
        else:
            return np.append(early_rul * np.ones(shape=(early_rul_duration,)), np.arange(early_rul - 1, -1, -1))


def process_input_data_with_targets(input_data, target_data=None, window_length=1, shift=1):
    """Depending on values of window_length and shift, this function generates batchs of data and targets
    from input_data and target_data.

    Number of batches = np.floor((len(input_data) - window_length)/shift) + 1

    **We don't check input dimensions uisng exception handling. So readers should be careful while using these
    functions. If input data are not of desired dimension, either error occurs or something undesirable is
    produced as output.**

    Arguments:
        input_data: input data to function (Must be 2 dimensional)
        target_data: input rul values (Must be 1D array)s
        window_length: window length of data
        shift: Distance by which the window moves for next batch. This is closely related to overlap
               between data. For example, if window length is 30 and shift is 1, there is an overlap of
               29 data points between two consecutive batches.

    """
    num_batches = int(np.floor((len(input_data) - window_length) / shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(np.nan, repeats=num_batches * window_length * num_features).reshape(num_batches,
                                                                                                window_length,
                                                                                                num_features)
    if target_data is None:
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
        return output_data
    else:
        output_targets = np.repeat(np.nan, repeats=num_batches)
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
            output_targets[batch] = target_data[(shift * batch + (window_length - 1))]
        return output_data, output_targets


def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows=1):
    """ This function takes test data for an engine as first input. The next two inputs
    window_length and shift are same as other functins.

    Finally it takes num_test_windows as the last input. num_test_windows sets how many examplles we
    want from test data (from last). By default it extracts only the last example.

    The function return last examples and number of last examples (a scaler) as output.
    We need the second output later. If we are extracting more than 1 last examples, we have to
    average their prediction results. The second scaler halps us do just that.
    """
    max_num_test_batches = int(np.floor((len(test_data_for_an_engine) - window_length) / shift)) + 1
    if max_num_test_batches < num_test_windows:
        required_len = (max_num_test_batches - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, max_num_test_batches
    else:
        required_len = (num_test_windows - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, num_test_windows


# Process training and test data sepeartely as number of engines in training and test set may be different.
# As we are doing scaling for full dataset, we are not bothered by different number of engines in training and test set.

# Process trianing data
for i in np.arange(1, num_train_machines + 1):
    temp_train_data = train_data[train_data[0] == i].drop(columns=[0]).values

    # Verify if data of given window length can be extracted from training data
    if (len(temp_train_data) < window_length):
        print("Train engine {} doesn't have enough data for window_length of {}".format(i, window_length))
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")

    temp_train_targets = process_targets(data_length=temp_train_data.shape[0], early_rul=early_rul)
    data_for_a_machine, targets_for_a_machine = process_input_data_with_targets(temp_train_data, temp_train_targets,
                                                                                window_length=window_length,
                                                                                shift=shift)

    processed_train_data.append(data_for_a_machine)
    processed_train_targets.append(targets_for_a_machine)

processed_train_data = np.concatenate(processed_train_data)
processed_train_targets = np.concatenate(processed_train_targets)

# Process test data
for i in np.arange(1, num_test_machines + 1):
    temp_test_data = test_data[test_data[0] == i].drop(columns=[0]).values

    # Verify if data of given window length can be extracted from test data
    if (len(temp_test_data) < window_length):
        print("Test engine {} doesn't have enough data for window_length of {}".format(i, window_length))
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")

    # Prepare test data
    test_data_for_an_engine, num_windows = process_test_data(temp_test_data, window_length=window_length, shift=shift,
                                                             num_test_windows=num_test_windows)

    processed_test_data.append(test_data_for_an_engine)
    num_test_windows_list.append(num_windows)

processed_test_data = np.concatenate(processed_test_data)
true_rul = true_rul[0].values

# Shuffle training data
index = np.random.permutation(len(processed_train_targets))
processed_train_data, processed_train_targets = processed_train_data[index], processed_train_targets[index]

print("Processed trianing data shape: ", processed_train_data.shape)
print("Processed training ruls shape: ", processed_train_targets.shape)
print("Processed test data shape: ", processed_test_data.shape)
print("True RUL shape: ", true_rul.shape)

target_scaler = MinMaxScaler(feature_range=(0, 1))
processed_train_targets = target_scaler.fit_transform(processed_train_targets.reshape(-1, 1)).reshape(-1)


def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.0001


window_length = 15


def create_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(512, 3, activation="relu", input_shape=(window_length, processed_train_data.shape[2])),
        tf.keras.layers.Conv1D(96, 5, activation="relu"),
        tf.keras.layers.Conv1D(32, 5, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model


processed_train_data, processed_val_data, processed_train_targets, processed_val_targets = train_test_split(
    processed_train_data,
    processed_train_targets,
    test_size=0.2,
    random_state=83)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

model = create_compiled_model()
history = model.fit(processed_train_data, processed_train_targets, epochs=30,
                    validation_data=(processed_val_data, processed_val_targets),
                    callbacks=callback,
                    batch_size=64, verbose=2)

rul_pred_scaled = model.predict(processed_test_data).reshape(-1)
rul_pred = target_scaler.inverse_transform(rul_pred_scaled.reshape(-1, 1)).reshape(-1)

preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])
mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights=np.repeat(1 / num_windows, num_windows))
                             for ruls_for_each_engine, num_windows in zip(preds_for_each_engine, num_test_windows_list)]
RMSE = np.sqrt(mean_squared_error(true_rul, mean_pred_for_each_engine))
print("RMSE: ", RMSE)
