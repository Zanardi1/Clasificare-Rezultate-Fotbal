# Data libraries
import pandas as pd
import numpy as np

# We read in the data sets using pandas
teams = pd.read_csv("teams.csv")
results = pd.read_csv("results.csv")
fixtures = pd.read_csv("fixtures.csv")
# players = pd.read_csv("players.csv")
# startingXI = pd.read_csv("startingXI.csv")
odds = pd.read_csv("odds.csv")

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

print(results.head())

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

print(results.head())

# We also split the results table into seasons 1 & 2
# Spliting tables by seasons 1 & 2
season1_results = results[results.SeasonID == 1]
season2_results = results[results.SeasonID == 2]

print(season1_results.head())
