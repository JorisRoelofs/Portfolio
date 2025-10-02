# Soccer Player Ratings Analysis
Keywords: Statistics, STATA, Excel

## Summary
Given the high stakes involved professional soccer teams face increasing pressure to optimize player recruitment and development using objective data. Understanding how demographics affect performance for different roles can guide scouting decisions and team composition. While age, height, and BMI were found to differ substantially between roles (especially Goalkeepers) their effect on performance was marginal. More indicative role-specific performance metrics are discussed for each role.

## Analysis
This case presents an analysis of demographics and performance in different soccer roles (Forward, Midfielder, Defender, Goalkeeper). Two datasets were merged, cleaned, and analyzed using robust clustered linear regression in STATA. Demograpics were found to differ significantly between Goalkeepers and other roles. Compared to Midfielders, Goalkeepers tend to be taller (12.4<ins>+</ins>1.5 cm, p>0.001), older (2.7<ins>+</ins>1.1 years, p>0.001, and leaner (-2.8<ins>+</ins>0.6 BMI, p>0.001).

![alt text](https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/All%20Analyses/Box%20Plot%20-%20Demographics%20per%20Roles.png?raw=true)

Nevertheless, no substantial effects were found of age, height, weight, BMI, or left-footedness on individual performance ratings by experts (Kicker, Bild, Skysport).

![alt text](Demographics roles)

Experts even rated the top 2.5% tallest, oldest, and lightest Goalkeepers marginally worse on performance than their short, young, and heavy counterparts.

![alt text](Extreme cases)

While it is possible that these resulted in performance differences earlier in their career, selecting taller, older, and leaners players for a Goalkeeper role does little to affect performance at a profesional level. More relevant role-specific performance indicators were determined in the following models.

|     Role                |     Performance predictors                                                                                            |     Variable name   explanation                                                                                                                                                                                                                                                               |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|     Universal           |     ss_goals     team_rating                                                                                          |     goals scored by the player during the match     average rating of the player's teammates during the game                                                                                                                                                                                  |
|     Goalkeeper          |     ss_dangmistakes     ss_goals_ag_otb     ss_goals_ag_itb     ss_saves_itb     team_goals      opp_bestfw_rating    |     dangerous mistakes made by the player     number of goals conceded from outside the box     number of goals conceded from inside the box     number of saves made from inside the box     number of goals scored by the team     rating of the best-rated forward of the opponent team    |
|     Defender            |     ss_assists     ss_chances2score     ss_clearances     team_pos_rating     opp_goals                               |     number of assists made given by the player     chances to score that the player had during the game     number of clearances made by the player     average rating of the teammates of the same position     number of goals scored by the opponent team                                  |
|     Midfielder          |     ss_assists     ss_passes_acc     ss_crosses_acc     team_pos_rating                                               |     number of assists made given by the player     number of passes successfully completed in the game by the player     number of accurate (completed) crosses by the player in the game     average rating of the teammates of the same position                                            |
|     Forward             |     ss_touches                                                                                                        |     number of times the player has touched the ball during the game, for any reason                                                                                                                                                                                                           |

## Data Overview
Source:
- Performance dataset: Provided by researcher Chris Snijders from four profesional cups (UEFA Euro 2016, Premier League 2017, Bundesliga 2017 and FIFA World Cup 2018)
- Demographic dataset: Scraped from www.fifaindex.com (2016-2019).

Data preprocessing:
- **Merging**: The Excel macro StripAccent.bas was used to overcome character differences in player names, followed by index matching on player name. 6305 of the 43690 rows lacked corresponding demographic data, which were excluded after they were determined to be random.
- **Removal**: Performance ratings by algorithms were dropped as they might include demographic variables in their calculations, leaving 24229 rows.
- **Standardization**: Performance ratings were standardized (M=0, SD=10) for each different source.
- **Cleaning**: Variables were renamed and reordered for clarity, and duplicate variables were dropped.
- **New Variables**: BMI was calculated based on height and weight, age during each competition was calculated based on birthday, and a left-footedness dummy was created.

Key variables:
|Performance|rating|individual performance ratings by experts from Kicker, Bild, and Skysport|
|Demographics|agecup|age during competition|
||height|single height value found|
||weightcup|weight during competition|
||BMI|BMI during competition|
||leftfoot|0 = right-footed, 1 = left-footed|
|Universal|ss_goals|goals scored by the player during the match|
|performance|team_rating|average rating of the player's teammates|
|indicators|||
|Goalkeeper|ss_dangmistakes|dangerous mistakes made by the player|
|performance|ss_goals_ag_otb|number of goals conceded from outside the box|
|indicators|ss_goals_ag_itb|number of goals conceded from inside the box|
||ss_saves_itb|number of saves made from inside the box|
||team_goals|number of goals scored by the team|
||opp_bestfw_rating|rating of the best-rated forward of the opponent team|
|Defender|ss_assists|number of assists made given by the player|
|performance|ss_chances2score|chances to score that the player had|
|indicators|ss_clearances|number of clearances made by the player|
||team_pos_rating|average rating of the teammates of the same position|
||opp_goals|number of goals scored by the opponent team|
|Midfielder|ss_assists|number of assists made given by the player|
|performance|ss_passes_acc|number of passes successfully completed by the player|
|indicators|ss_crosses_acc|number of accurate (completed) crosses by the player|
||team_pos_rating|average rating of the teammates of the same position|
|Forward|ss_touches|number of times the player has touched the ball|
|performance|||
|indicators|||

## Appendix
Interaction effects between the demographic extremes and universal performance predictors of goals (Figure A.1) and team rating (Figure A.2), but they did little to explain differences in performance ratings (R^2<0.05).

![alt text](Goals interaction)

![alt text](Team rating interaction)