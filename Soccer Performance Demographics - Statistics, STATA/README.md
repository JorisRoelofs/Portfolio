# Soccer Player Ratings Analysis
Skills: Statistics, R, STATA, Excel

## Summary
Given the high stakes involved professional soccer teams face increasing pressure to optimize player recruitment and development using objective data. Understanding how demographics affect performance for different roles can guide scouting decisions and team composition. Two datasets were merged, cleaned, and analyzed using Excel, STATA, and R. While age, height, and BMI were found to differ substantially between roles (especially Goalkeepers) their effect on performance was negligible to non-existent, suggesting that players are partially selected based on appearance. More indicative role-specific performance metrics are discussed for each role to guide scouting decisions and team composition.

## Results
Demograpics were found to differ significantly between Goalkeepers and other roles. Compared to Midfielders, Goalkeepers tend to be taller (12.4<ins>+</ins>1.5 cm, p>0.001), older (2.7<ins>+</ins>1.1 years, p>0.001, and leaner (-2.8<ins>+</ins>0.6 BMI, p>0.001). Nevertheless, even as the shortest role Midfielders are substantially taller than the male population average of ~175cm (p<0.001, d=16.1).

![Height, weight, age, and BMI differences between Goalkeepers, Defenders, Midfielders, and Forwards](https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Boxplot%20Demograpics%20-%20Competition.pdf)
*Figure 1. Box plot showing demographical differences between the four roles.*

Nevertheless, no substantial effects were found of age, height, weight, BMI, or footedness on individual player performance ratings by experts (Kicker, Bild, Skysport).

![Linear regression output for each of the four roles](Demographics roles)
<p float="left">
  <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Demographics%20-%20Goalkeeper.png" width="50%" />
  <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Demographics%20-%20Defender.png" width="50%" /> 
</p>
<p float="left">
  <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Demographics%20-%20Midfielder.png" width="50%" />
  <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Demographics%20-%20Forward.png" width="50%" /> 
</p>
*Figure 2. (Robust clustered) linear regression models using demographic variables to predict expert ratings*

Experts even rated the top 2.5% tallest, oldest, and lightest Goalkeepers marginally worse on performance than their short, young, and heavy counterparts.

![alt text](Extreme cases)
*Figure 3. (Robust clustered) linear regression models using demographic extremes (top/bottom 2.5%) to predict expert ratings*


While it is possible that these resulted in performance differences earlier in their career, selecting taller, older, and leaners players for a Goalkeeper role does little to affect performance at a profesional level. More relevant role-specific performance indicators were determined in the following models.

![alt text](performative models)
*Figure 4. (Robust clustered) linear regression models using result-driven performance indactors as comparison to demographic variables*


|     Role                |     Performance predictors                                                                                            |     Variable name   explanation                                                                                                                                                                                                                                                               |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|     Universal           |     ss_goals     team_rating                                                                                          |     goals scored by the player during the match     average rating of the player's teammates during the game                                                                                                                                                                                  |
|     Goalkeeper          |     ss_dangmistakes     ss_goals_ag_otb     ss_goals_ag_itb     ss_saves_itb     team_goals      opp_bestfw_rating    |     dangerous mistakes made by the player     number of goals conceded from outside the box     number of goals conceded from inside the box     number of saves made from inside the box     number of goals scored by the team     rating of the best-rated forward of the opponent team    |
|     Defender            |     ss_assists     ss_chances2score     ss_clearances     team_pos_rating     opp_goals                               |     number of assists made given by the player     chances to score that the player had during the game     number of clearances made by the player     average rating of the teammates of the same position     number of goals scored by the opponent team                                  |
|     Midfielder          |     ss_assists     ss_passes_acc     ss_crosses_acc     team_pos_rating                                               |     number of assists made given by the player     number of passes successfully completed in the game by the player     number of accurate (completed) crosses by the player in the game     average rating of the teammates of the same position                                            |
|     Forward             |     ss_touches                                                                                                        |     number of times the player has touched the ball during the game, for any reason                                                                                                                                                                                                           |
*Table 1.*

## Method
Sources:
- **Performance dataset**: 105 performance metrics for each player per match from four profesional cups (UEFA Euro 2016, Premier League 2017, Bundesliga 2017 and FIFA World Cup 2018), provided by researcher Chris Snijders.
- **Demographic dataset**: date of birth, height, weight per year, and footedness scraped for each player between 2016 and 2019 from www.fifaindex.com.

Data preprocessing:
- **Merging**: The Excel macro StripAccent.bas was used to overcome character differences in player names, followed by index matching on player name. 6305 of the 43690 rows lacked corresponding demographic data, which were excluded after they were determined to be random.
- **Removal**: Performance ratings by algorithms were dropped as they might include demographic variables in their calculations, leaving 24229 rows for 1442 players.
- **Standardization**: Performance ratings were standardized (M=0, SD=10) for each different source.
- **Cleaning**: Variables were renamed and reordered for clarity, and duplicate variables were dropped.
- **New Variables**: BMI was calculated based on height and weight, age during each competition was calculated based on birthday, and a left-footedness dummy was created.

Key variables:

|Category|Variable|Description|
|-|-|-|
|Performance|rating|Individual performance ratings by experts from Kicker, Bild, and Skysport|
|Demographics|agecup|Age during competition|
||height|Height constant found in dataset|
||weightcup|Weight during competition|
||bmi|BMI during competition|
||leftfoot|0 = right-footed, 1 = left-footed|
|Universal performance indicators|ss_goals|Goals scored by the player during the match|
||team_rating|Average rating of the player's teammates|
||||
|Goalkeeper performance indicators|ss_dangmistakes|Dangerous mistakes made by the player|
||ss_goals_ag_otb|Number of goals conceded from outside the box|
||ss_goals_ag_itb|Number of goals conceded from inside the box|
||ss_saves_itb|Number of saves made from inside the box|
||team_goals|Number of goals scored by the team|
||opp_bestfw_rating|Rating of the best-rated forward of the opponent team|
|Defender performance indicators|ss_assists|Number of assists made given by the player|
||ss_chances2score|Chances to score that the player had|
||ss_clearances|Number of clearances made by the player|
||team_pos_rating|Average rating of the teammates of the same position|
||opp_goals|Number of goals scored by the opponent team|
|Midfielder performance indicators|ss_assists|Number of assists made given by the player|
||ss_passes_acc|Number of passes successfully completed by the player|
||ss_crosses_acc|Number of accurate (completed) crosses by the player|
||team_pos_rating|Average rating of the teammates of the same position|
|Forward performance indicators|ss_touches|Number of times the player has touched the ball|

Assumption checking:
As the same player playing in multiple matches should not be considered entirely different cases, a multilevel regression was initially used to separate players. However, when checking for the model assumptions heteroscedasticity and error independence did not hold for many of these models, so ‘robust’ linear regression was chosen in the end with clustering to separate players. The conclusions did not change with the new models. Outlier counts were within the margins of a normal distribution (<5% for 1.96 STD, <1% for 2.5 STD, <0.1% for 3.29 STD), and residual normality was not relevant for the new robust models.

Final analyses:
Box plots were drawn in R for comparison between roles and cups. Robust clustered linear regression were used in STATA for the predictive models. A robust clustered one-sample t-test was later added using R to compare Midfielder heights to the population average.

## Appendix
Demographic differences were also compared between the competitions to ensure compatibility. As can be seen below no major differences were found, allowing them to be used together in the analysis.
![Box plot showing height, weight, age, and BMI differences between Euro 2016, Bundesliga 2017, Premier League 2017, and World Cup 2018](https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Boxplot%20Demograpics%20-%20Competition.pdf)
*Figure A.1. Box plot showing demographical differences between the four competitions.*


Interaction effects between the demographic extremes and universal performance predictors of goals (Figure A.1) and team rating (Figure A.2), but they did little to explain differences in performance ratings (R^2<0.05).

![alt text](Goals interaction)

![alt text](Team rating interaction)