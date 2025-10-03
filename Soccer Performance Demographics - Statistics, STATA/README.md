# Expert Performance Ratings and Demographics in Soccer
***Skills***_: Statistics, R, STATA, Excel_

## Overview
Given the high stakes involved professional soccer teams face increasing pressure to optimize player recruitment and development using objective data. Understanding how demographics affect performance for different roles can guide scouting decisions and team composition. Two datasets were merged, cleaned, and analyzed using Excel, STATA, and R. While age, height, and BMI were found to differ substantially between roles (especially Goalkeepers) their effect on performance was negligible to non-existent, suggesting that players are partially selected based on appearance. More indicative role-specific performance metrics are discussed for each role to guide scouting decisions and team composition.

## Data Sources
- **Performance dataset**: 134 metrics for each player per match from four profesional cups (UEFA Euro 2016, Premier League 2017, Bundesliga 2017 and FIFA World Cup 2018). Provided by researcher Chris Snijders.
- **Demographic dataset**: Height, weight, date of birth, and footedness for each player between 2016 and 2019. Scraped from www.fifaindex.com.

## Data Processing
- **Merging**: The Excel macro "StripAccent.bas" was used to standardize player names, followed by index matching. Rows without demographic matches were removed (6,305 of 43,690). No patterns were found that differentiated these from the remaining rows.
- **Filtering**: Algorithmic performance ratings (SofaScore, WhoScored, The Guardian) were removed to avoid potential contamination of predictors (13,086 of 37,385). The final dataset includes 24,229 expert ratings for 1,442 players.
- **Standardization**: Performance ratings were standardized to mean 0 and standard deviation 10 for each expert source separately (Kicker, Bild, Skysport).
- **Derived Variables**: BMI was calculated based on height and weight, age during each competition was calculated based on birthday, and a left-footedness dummy was created.
- **Outliers**: Counts of extreme values were determined to be withing acceptable limits (<5% for 1.96 STD, <1% for 2.5 STD, <0.1% for 3.29 STD).

## Statistical Analysis
- **Predictive Modeling**: Robust clustered linear regressions were computed in STATA to account for repeated measures per player, heteroscedasticity, and residual correlations.
- **Hypothesis Testing**: A robust clustered one-sample t-test was computed in R to compare Midfielder heights against the population average.
- **Visualization**: Box plots were generated in R to compare demographics across roles and competitions.

## Results
### Demographic Differences Between Roles
Goalkeepers displayed significant differences compared to other roles. Relative to Midfielders, Goalkeepers were taller (+12.4 ± 1.5 cm, p < 0.001), older (+2.7 ± 1.1 years, p < 0.001), and leaner (−2.8 ± 0.6 BMI, p < 0.001). Nevertheless, even as the shortest role Midfielders (179.7cm) were substantially taller than the male population average of 172.7cm (p < 0.001, Cohen’s d = 23.5).

<p align="center">
  <img src=https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Boxplot%20Demograpics%20-%20Role.png title = "Height, weight, age, and BMI differences between Goalkeepers, Defenders, Midfielders, and Forwards"\>
</p>

<table>
  <tr>
    <td align="center">
      <b>Goalkeeper</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Role%20Differences%20-%20Height.png"/>
    </td>
    <td align="center">
      <b>Defender</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Role%20Differences%20-%20Height.png"/> 
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Midfielder</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Role%20Differences%20-%20Age.png"/>
    </td>
    <td align="center">
      <b>Defender</b><br>
      <img src=https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Role%20Differences%20-%20BMI.png"/> 
      </td>
  </tr>
</table>


### Predictive Value pf Demographics
Despite these differences, no substantial effects were found of age, height, weight, BMI, or footedness on individual player performance ratings by experts (Kicker, Bild, Skysport).

<table>
  <tr>
    <td align="center">
      <b>Goalkeeper</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Demographics%20-%20Goalkeeper.png"/>
    </td>
    <td align="center">
      <b>Defender</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Demographics%20-%20Defender.png"/> 
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Midfielder</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Demographics%20-%20Midfielder.png"/>
    </td>
    <td align="center">
      <b>Defender</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Demographics%20-%20Forward.png"/> 
      </td>
  </tr>
</table>

*Figure 2. (Robust clustered) linear regression models using demographic variables to predict expert ratings*

Experts even rated the top 2.5% tallest, oldest, and lightest Goalkeepers marginally worse on performance than their short, young, and heavy counterparts.

<table>
  <tr>
    <td align="center">
      <b>Goalkeeper</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Demographic%20Extremes%20-%20Goalkeeper.png"/>
    </td>
    <td align="center">
      <b>Defender</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Demographic%20Extremes%20-%20Defender.png"/> 
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Midfielder</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Demographic%20Extremes%20-%20Midfielder.png"/>
    </td>
    <td align="center">
      <b>Defender</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Demographic%20Extremes%20-%20Forward.png"/> 
      </td>
  </tr>
</table>

*Figure 3. (Robust clustered) linear regression models using demographic extremes (top/bottom 2.5%) to predict expert ratings*

### Role-Specific Performance Metrics
While it is possible that these resulted in performance differences earlier in their career, selecting taller, older, and leaners players for a Goalkeeper role does little to affect performance at a profesional level. More relevant role-specific performance indicators were determined in the following models.

<table>
  <tr>
    <td align="center">
      <b>Goalkeeper</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Performance%20Metrics%20-%20Goalkeeper.png" title="ss_dangmistakes: dangerous mistakes by player, ss_goals_ag_otb: goals conceded from outside the box, ss_goals_ag_itb: goals conceded from inside the box, ss_saves_itb: saves made from inside the box, team_goals: goals by team, team_rating: average rating of other team members, opp_gk_rating: rating of opponent goalkeeper, opp_bestfw_rating: highest rating of opponent forwards"/>
    </td>
    <td align="center">
      <b>Defender</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Performance%20Metrics%20-%20Defender.png" title="ss_goals: goals by player, ss_assists: assists by player, ss_chances2score: chances player had to score, ss_clearances: clearances made by player, team_rating: average rating of other team players, team_pos_rating: average rating of other players in this position, opp_goals: goals made by opponents"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Midfielder</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Performance%20Metrics%20-%20Midfielder.png" title="ss_goals: goals by player, ss_assists: assists by player, ss_passes_acc: accurate passes by player, ss_crosses_acc: accurate crosses by player, team_rating: average rating of other team members, team_pos_rating: average rating of other players in this position"/>
    </td>
    <td align="center">
      <b>Defender</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Performance%20Metrics%20-%20Forward.png" title="ss_goals: goals by player, ss_touches: number of times the player touched the ball, team_rating: average rating of other team members"/> 
      </td>
  </tr>
</table>

*Figure 4. (Robust clustered) linear regression models using result-driven performance indactors as comparison to demographic variables*


|Role|Variable|Explanation|
|-|-|-|
|Goalkeeper|ss_dangmistakes<br>ss_goals_ag_otb<br>ss_goals_ag_itb<br>ss_saves_itb<br>team_goals<br>team_rating<br>opp_bestfw_rating|dangerous mistakes made by the player<br>number of goals conceded from outside the box<br>number of goals conceded from inside the box<br>number of saves made from inside the box<br>number of goals scored by the team<br>average rating of the player's teammates<br>rating of the best-rated forward of the opponent team|
|Defender|ss_goals<br>ss_assists<br>ss_chances2score<br>ss_clearances<br>team_rating<br>team_pos_rating<br>opp_goals|goals scored by the player<br>number of assists made given by the player<br>number of chances the player had to score<br>number of clearances made by the player<br>average rating of teammates with the same role<br>average rating of the player's teammates<br>number of goals scored by the opponent team|
|Midfielder|ss_goals<br>ss_assists<br>ss_passes_acc<br>ss_crosses_acc<br>team_rating<br>team_pos_rating|goals scored by the player<br>number of assists made given by the player<br>number of passes successfully completed by the player<br>number of accurate (completed) crosses by the player<br>average rating of teammates with the same role<br>average rating of the player's teammates|
|Forward|ss_goals<br>ss_touches<br>team_rating|goals scored by the player<br>how often the player touched the ball, for any reason<br>average rating of the player's teammates|

*Table 1. Explanation of perfomance variables*

## Appendix
Demographic differences were also compared between the competitions to ensure compatibility. As can be seen below no major differences were found, allowing them to be used together in the analysis.
<p align="center">
  <img src=https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Boxplot%20Demograpics%20-%20Competition.png title = "Height, weight, age, and BMI differences between Euro 2016, Bundesliga 2017, Premier League 2017, and World Cup 2018"\>
</p>

*Figure A.1. Box plot showing demographical differences between the four competitions.*


Interaction effects between the demographic extremes and universal performance predictors of goals (Figure A.2) and team rating (Figure A.3), but they did little to explain differences in performance ratings (R^2<0.05).

<table>
  <tr>
    <td align="center">
      <b>Goalkeeper</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Goal%20Interactions%20-%20Goalkeeper.png"/>
    </td>
    <td align="center">
      <b>Defender</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Goal%20Interactions%20-%20Defender.png"/> 
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Midfielder</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Goal%20Interactions%20-%20Midfielder.png"/>
    </td>
    <td align="center">
      <b>Defender</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Goal%20Interactions%20-%20Forward.png"/> 
      </td>
  </tr>
</table>

*Figure A.2.*

<table>
  <tr>
    <td align="center">
      <b>Goalkeeper</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Team%20Rating%20Interactions%20-%20Defender.png"/>
    </td>
    <td align="center">
      <b>Defender</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Team%20Rating%20Interactions%20-%20Defender.png"/> 
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Midfielder</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Team%20Rating%20Interactions%20-%20Midfielder.png"/>
    </td>
    <td align="center">
      <b>Defender</b><br>
      <img src="https://github.com/JorisRoelofs/Portfolio/blob/main/Soccer%20Performance%20Demographics%20-%20Statistics%2C%20STATA/Visuals/Regression%20Team%20Rating%20Interactions%20-%20Forward.png"/> 
      </td>
  </tr>
</table>

*Figure A.3.*