from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from Preprocessing.pre_processing import training_data

X_train = training_data.iloc[:, 5:26]
y_train = training_data['RUL']
lm = LinearRegression()
rfe = RFE(lm)  # running RFE
rfe = rfe.fit(X_train, y_train)

sel_cols = []
for i in range(0, 21):
    if rfe.ranking_[i] <= 3:
        sel_cols.append(i+1)
print("Most relevant features")
for prefix in sel_cols:
    print("sensor_{}".format(prefix))