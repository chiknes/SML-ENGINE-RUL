import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from Preprocessing.pre_processing import test_data
from Preprocessing.pre_processing import training_data

TTF = 10
training_data['label'] = np.where(training_data['RUL'] <= TTF, 1, 0)

X_class = training_data.iloc[:, [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]].to_numpy()
Y_class = training_data.iloc[:, 27:].to_numpy()
Y_class = np.ravel(Y_class)

test_max = test_data.groupby('unit_number')['time_in_cycles'].max().reset_index()
test_max.columns = ['unit_number', 'max']
fd_001_test = test_data.merge(test_max, on=['unit_number'], how='left')
test = fd_001_test[fd_001_test['time_in_cycles'] == fd_001_test['max']].reset_index()
test.drop(columns=['index', 'max', 'unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3', 'sensor_1',
                   'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'], inplace=True)
X_001_test = test.to_numpy()

ros = RandomOverSampler(random_state=0)
ros.fit(X_class, Y_class)
X_resampled, y_resampled = ros._fit_resample(X_class, Y_class)
print('Number of elements before re-sampling:', len(X_class))
print('Number of elements after re-sampling:', len(X_resampled))

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=3)

forest = RandomForestClassifier(n_estimators=70, max_depth=8, random_state=193)
forest.fit(X_train, y_train)

model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)


def classificator_score(y_, y_p):
    print(f' accuracy score {round(accuracy_score(y_, y_p), 2)}')
    print(f' precision score {round(precision_score(y_, y_p), 2)}')
    print(f' recall score {round(recall_score(y_, y_p), 2)}')
    print(f' F1 score {round(f1_score(y_, y_p), 2)}')
    return


classificator_score(y_test, forest.predict(X_test))

y_xgb_pred = model_xgb.predict(X_001_test)
classificator_score(y_test, model_xgb.predict(X_test))
