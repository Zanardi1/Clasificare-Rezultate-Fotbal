# Data libraries
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter("ignore")

# We read in the data sets using pandas
teams = pd.read_csv("teams.csv")
results = pd.read_csv("results.csv")
fixtures = pd.read_csv("fixtures.csv")

# It appears that the results data set only provides the number of goals scored by each team in each game.
#
# We create the binary indicator variables:
#
# HomeWin: 1 if the home team wins, 0 otherwise
# Draw: 1 if the teams draw, 0 otherwise
# AwayWin: 1 if the away team wins, 0 otherwise

# Creating binary indicator variables for wins/draws/losses
results['HomeWin'] = np.where(
    results['HomeScore'] > results['AwayScore'], 1, np.where(
        results['HomeScore'] == results['AwayScore'], 0, 0))
results['Draw'] = np.where(
    results['HomeScore'] == results['AwayScore'], 1, np.where(
        results['HomeScore'] > results['AwayScore'], 0, 0))
results['AwayWin'] = np.where(
    results['AwayScore'] > results['HomeScore'], 1, np.where(
        results['AwayScore'] == results['HomeScore'], 0, 0))

# It would be helpful to have the team name in the data frame, as opposed to the team ID. We merge results with teams
# data set to obtain this.
# Merging `results` and `teams` dataframes to obtain team names
# Home team names
results = pd.merge(results, teams, left_on='HomeTeamID', right_on='TeamID', how='left')
results = results.drop('TeamID', 1)
results.rename(columns={'TeamName': 'HomeTeamName'}, inplace=True)
# Away team names
results = pd.merge(results, teams, left_on='AwayTeamID', right_on='TeamID', how='left')
results = results.drop('TeamID', 1)
results.rename(columns={'TeamName': 'AwayTeamName'}, inplace=True)

# Removing unnecessary variables
results = results.drop(['HomeScore', 'AwayScore', 'HomeShots', 'AwayShots'], 1)

# We also split the results table into seasons 1 & 2
# Spliting tables by seasons 1 & 2
season1_results = results[results.SeasonID == 1]
season2_results = results[results.SeasonID == 2]

# Setting the features (X) and the outputs (y). The outcome of the match depends only on the teams that are playing it

X = ['HomeTeamID', 'AwayTeamID']
y = ['HomeWin', 'Draw', 'AwayWin']

# Splitting the season 1 data in training and testing data
X_train, X_test, y_train, y_test = train_test_split(season1_results[X], season1_results[y])

# Using a Random Forest Classifier as a classifier
R = RandomForestClassifier(n_estimators=100)
R.fit(X_train, y_train)
print('Model''s accuracy for the test set: ', accuracy_score(y_test, R.predict(X_test)))

# Display the predicted outcomes and real for the second season
print(R.predict(season2_results[X]))
print(season2_results[y])

# Creating and displaying the confusion matrix
cm = confusion_matrix(season2_results[y].values.argmax(axis=1), R.predict(season2_results[X]).argmax(axis=1))
disp = ConfusionMatrixDisplay(cm, display_labels=['Home Win', 'Draw', 'Away Win'])
disp.plot()
plt.show()
