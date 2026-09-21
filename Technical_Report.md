# Team Selection using Linear Regression

## Problem to Solve:
Given the two IPL teams that are set to face each other on a given IPL match date, the goal was to predict which 11 players (under some team selection constraints) are expected to perform the best (earn the most fantasy points)?

## Approach:
Our approach was to predict a player's performances against a given set of opponents, on a given date, against a given team, on the basis of how well that player has historically performed under similar circumstances. In order to do this, we analyze all the past data of that player's performance (2022 onwards up till the last match played before the one on the given date) in order to generate specific statistical features. A linear combination of these features then acts as a measure of the number of fantasy points we expect that player to score on that day. The team of the best 11 players that satisfies the constraints of:

1-8 Batsmen,

1-8 Bowlers,

1-8 Wicket-Keepers,

1-8 All-Rounders

and at least one player from each team

is then selected on the basis of these predicted fantasy points.

## Model (Linear Combination of Historical Features):
The model captures the player's historical performance in situations similar to the ones they would face on the match day by calculating the following four features over all delivery data from 2022 onwards:

1. The average of the average points scored by the player at his team's home venue and the average points scored by him at the opposing team's home venue.

This aims to capture how the particular venues he may be playing on on the date of the match tend to influence his performance.

2. A moving average of points scored across historical seasons on match days around the serial number corresponding to the number of matches that have been played by the player's team in the current season.

If, for example, the player is about to play the 10th match of his current season, then the model looks at his performance in previous seasons on the 9th, 10th and 11th matches he played, as well as the 7th to 9th matches played in the ongoing season and calculates a moving average over it.

This feature aims to capture patterns in how the number of matches he has played in a given season, i.e., whether he currently finds himself in the early, mid or late-season, affect his performance.

3. The model considers all historical matchups that the player has had with the players in the opposite team's roster on that day. It then calculates the ratio of all the points he has scored against those players upon those scored by those players against him. It then calculates the logarithm of that ratio.

This aims to capture whether the player has historically done better against that particular lineup of opponents, or whether they have bested him.

4. Similar to feature no. 3, the model considers all the matches that the player's team has played against the opposite team and calculates the logarithm of the ratio of the points scored by the player's team against the opponent team upon those scored by the opponent team against the player's team.

This looks to measure whether the player's team has had any advantage over that particular opponent team and how it would affect the player's individual performance that day.

These features are then linearly added to get the player's predicted fantasy points for that match. The weights for the features were determined using linear regression on the features against the actual fantasy points scored by the players on the corresponding historical match dates.

## Other Approaches Investigated:
We also investigated the approaches of Random Forests with CAT-Boosting, as well as a Hierarchical-Bayes model with Monte-Carlo Sampling for the purposes of selecting the optimal team and assessed their performance against the linear features approach.

Over all the 2026 matches, the Hierarchical-Bayes model averaged 5.94 / 11 Dream Team 11 player, whereas over all 2025 matches, the CAT-Boost model averaged 5.97 / 11 top-11 hits.
