# Soccer Player Ratings Analysis
Keywords: Statistics, STATA

## Summary
This case presents an analysis of the impact of demographics () on soccer performance ratings per role (Forward, Midfielder, Defender, Goalkeeper) in STATA. While previous works ... no significant effects were found of age, height, weight, BMI, or left-footedness on performance ratings. However, significant height differences were found between roles, with Goalkeepers being on average 12.4<ins>+</ins>1.5 cm taller than Midfielders (p>0.001), suggesting that taller players tend towards a Goalkeeper role.
![alt text](https://github.com/JorisRoelofs/Portfolio/blob/main/image.jpg?raw=true)

## Context
Being a trillion dollay industry, professional soccer teams face increasing pressure to optimize player recruitment and development using objective data. Understanding how demographics such as age, height, weight, position, and footedness affect performance ratings can guide scouting decisions and team composition.

## Data Overview
Source: Performance data provided by researcher Chris Snijders (UEFA Euro 2016, Premier League 2017, Bundesliga 2017 and FIFA World Cup 2018) and merged with demographics scraped from www.fifaindex.com (2016-2019).
Key variables:
- 
- 
- 
Data preprocessing:
- Merging: Used Excel macro <name_macros.txt> to overcome character differences in player names, followed by index matching on player name. Of the original 43690 original entries 6305 lacked demographic data, which were left out of the analysis after they were determined to be random (no pattern in country, age, role, etc.).
- 
- 

## Results
while weight differences were marginal (<2.2kg, p>0.1), resulting in a substantial BMI (Body Mass Index) difference of 2.8

![alt text](https://github.com/JorisRoelofs/[reponame]/blob/[branch]/image.jpg?raw=true)

## Appendix
### Appendix A: Explanation of Variables
|     Role                |     Established   predictors                                                                                          |     Variable name   explanation                                                                                                                                                                                                                                                               |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|     Universal           |     ss_goals     team_rating                                                                                          |     goals scored by the player during the match     average rating of the player's teammates during the game                                                                                                                                                                                  |
|     Goalkeeper          |     ss_dangmistakes     ss_goals_ag_otb     ss_goals_ag_itb     ss_saves_itb     team_goals      opp_bestfw_rating    |     dangerous mistakes made by the player     number of goals conceded from outside the box     number of goals conceded from inside the box     number of saves made from inside the box     number of goals scored by the team     rating of the best-rated forward of the opponent team    |
|     Defender            |     ss_assists     ss_chances2score     ss_clearances     team_pos_rating     opp_goals                               |     number of assists made given by the player     chances to score that the player had during the game     number of clearances made by the player     average rating of the teammates of the same position     number of goals scored by the opponent team                                  |
|     Midfielder          |     ss_assists     ss_passes_acc     ss_crosses_acc     team_pos_rating                                               |     number of assists made given by the player     number of passes successfully completed in the game by the player     number of accurate (completed) crosses by the player in the game     average rating of the teammates of the same position                                            |
|     Forward             |     ss_touches                                                                                                        |     number of times the player has touched the ball during the game, for any reason                                                                                                                                                                                                           |