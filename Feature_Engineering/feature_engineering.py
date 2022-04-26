"""

Usage:
    ./feature_engineering.py

Authors:
    Shailesh, Rishabh 04-26-22
"""

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from Preprocessing.pre_processing import training_data

X_train = training_data.iloc[:, 5:26]
y_train = training_data['RUL']
lm = LinearRegression()
rfe = RFE(lm)  # running RFE
rfe = rfe.fit(X_train, y_train)

# Choosing sensors having ranking below 4
sel_cols = []
for i in range(0, 21):
    if rfe.ranking_[i] <= 3:
        sel_cols.append(i+1)
print("Most relevant features")

# print all the relevant sensors
for prefix in sel_cols:
    print("sensor_{}".format(prefix))


# bestfeatures = SelectKBest(score_func=chi2, k=10)
# X = train_df.iloc[:, :14]
# fit = bestfeatures.fit(X.to_numpy(), train_df.iloc[:, 14:].to_numpy())
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# # concat two dataframes for better visualization
# featureScores = pd.concat([dfcolumns, dfscores], axis=1)
# featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
# print(featureScores.nlargest(10, 'Score'))