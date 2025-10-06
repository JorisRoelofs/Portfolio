# Expert Performance Ratings and Demographics in Soccer
***Skills***_: Statistics, R, STATA, Excel_

## Overview
Professional soccer teams face increasing pressure to optimize player recruitment and development using objective data. As such, this project evaluates how player demographics (age, height, weight, BMI, and footedness) relate to expert performance ratings across different roles. Two datasets were merged, cleaned, and analyzed using Excel, STATA, and R. Demographic characteristics were found to differ substantially between roles, yet had minimal predictive value for performance. Role-specific performance indicators provided more actionable insights for scouting, training, and team composition.

## Data Sources
- **Performance dataset**: 134 metrics for each player per match from four professional cups (UEFA Euro 2016, Premier League 2017, Bundesliga 2017 and FIFA World Cup 2018). Provided by researcher Chris Snijders.
- **Demographic dataset**: Height, weight, date of birth, and footedness for each player between 2016 and 2019. Scraped from www.fifaindex.com.

## Data Processing
- **Merging**: The Excel macro "StripAccent.bas" was used to standardize player names, followed by index matching. Rows without demographic matches were removed (6,305 of 43,690). No patterns were found that differentiated these from the remaining rows.
- **Filtering**: Algorithmic performance ratings (SofaScore, WhoScored, The Guardian) were removed to avoid potential data contamination (13,086 of 37,385). The final dataset includes 24,229 rows for 1,442 players.
- **Standardization**: Performance ratings were standardized to mean 0 and standard deviation 10 for each expert source separately (Kicker, Bild, Skysport).
- **Derived Variables**: BMI was calculated from height and weight. Age at each competition was calculated based on birthday. A dummy-variable for left-footedness was created.
- **Outliers**: Counts of extreme values were determined to be withing acceptable limits (<5% for 1.96 STD, <1% for 2.5 STD, <0.1% for 3.29 STD).
- **Compatibility**: The competitions were checked for demographic differences to ensure compatibility. No major differences were observed, confirming that they could be analyzed together.

<p align="center">
  <img src=Visuals/Boxplot%20Demograpics%20-%20Competition.png \>
</p>
*Figure 1. Box plot showing minimal demographical differences between the four competitions.*

## Statistical Analysis
- **Predictive Modeling**: Robust clustered linear regressions were computed in STATA to account for repeated measures per player, heteroscedasticity, and residual correlations. Role-dependent performance indicators were hand-picked using literature reviews and tested across several models to determine which indicators offered substantial predictive power for each role.
- **Hypothesis Testing**: A robust clustered one-sample t-test was computed in R to compare Midfielder heights against the population average.
- **Visualization**: Box plots were generated in R to compare demographics across roles and competitions.

## Results
### Demographic Differences Between Roles
Goalkeepers displayed significant differences compared to other roles. Relative to Midfielders, Goalkeepers were taller (+12.4 ± 1.5 cm, p < 0.001), older (+2.7 ± 1.1 years, p < 0.001), and leaner (−2.8 ± 0.6 BMI, p < 0.001). Nevertheless, even as the shortest role Midfielders (179.7cm) were substantially taller than the male population average of 172.7cm (p < 0.001, Cohen’s d = 23.5).

<p align="center">
  <img src=Visuals/Boxplot%20Demograpics%20-%20Role.png \>
</p>
*Figure 2. Box plot showing the demographical differences between soccer roles.*

<p align="center">
  <img src=Visuals/Regression%20Demographics.png% \>
</p>
*Figure 3. (Robust clustered) linear models showing substantial demographical differences between Goalkeepers and outfield roles.*

### Predictive Value of Demographics
Despite these differences, no substantial effects were found of age, height, weight, BMI, or footedness on individual player performance ratings by experts (Kicker, Bild, Skysport).

<p align="center">
  <img src=Visuals/Regression%20Demographics.png width=100% \>
</p>
*Figure 4. (Robust clustered) linear regression models showing insignificant effects of demographical differences on expert performance ratings.*

Even in extreme demographical subsets (top/bottom 2.5%) effects were mostly insignificant. In contrast to the Goalkeeper-archetype, the top 2.5% tallest and lightest players were rated slightly worse while the top 2.5% youngest and heaviest players were rated slightly better, although explained variance is minor (p < 0.05, R^2 = 0.01).

<p align="center">
  <img src=Visuals/Regression Demograpic Extremes.png \>
  Visuals/Regression Demographic Extremes.png
</p>
*Figure 5. (Robust clustered) linear regression models showing insubstantial differences in expert ratings between demographic extremes (top/bottom 2.5%).*

### Role-Specific Performance Metrics
More relevant role-specific performance indicators were determined in the following models.

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

*Figure 6. (Robust clustered) linear regression models using result-driven performance indactors as comparison to demographic variables.*


|Role|Variable|Explanation|
|-|-|-|
|Goalkeeper|ss_dangmistakes<br>ss_goals_ag_otb<br>ss_goals_ag_itb<br>ss_saves_itb<br>team_goals<br>team_rating<br>opp_bestfw_rating|dangerous mistakes made by the player<br>number of goals conceded from outside the box<br>number of goals conceded from inside the box<br>number of saves made from inside the box<br>number of goals scored by the team<br>average rating of the player's teammates<br>rating of the best-rated forward of the opponent team|
|Defender|ss_goals<br>ss_assists<br>ss_chances2score<br>ss_clearances<br>team_rating<br>team_pos_rating<br>opp_goals|goals scored by the player<br>number of assists made given by the player<br>number of chances the player had to score<br>number of clearances made by the player<br>average rating of teammates with the same role<br>average rating of the player's teammates<br>number of goals scored by the opponent team|
|Midfielder|ss_goals<br>ss_assists<br>ss_passes_acc<br>ss_crosses_acc<br>team_rating<br>team_pos_rating|goals scored by the player<br>number of assists made given by the player<br>number of passes successfully completed by the player<br>number of accurate (completed) crosses by the player<br>average rating of teammates with the same role<br>average rating of the player's teammates|
|Forward|ss_goals<br>ss_touches<br>team_rating|goals scored by the player<br>how often the player touched the ball, for any reason<br>average rating of the player's teammates|

*Table 1. Explanation of the role-specific performance variables.*

### Interaction Effects
To determine whether demographic variables might have an indirect effect, interactions were evaluated between the demographic extremes and the performance predictors dominant across several roles (goals and team ratings). Nevertheless, the interactions explained only a minor part of the variance (R^2 < 0.05).

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

*Figure 7. (Robust clustered) linear regression models testing interaction effects between demographic extremes and goals scored.*

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

*Figure 8. (Robust clustered) linear regression models testing interaction effects between demographic extremes and team ratings.*

## Conclusions
- **Demographic differences exist yet have limited predictive value:** Goalkeepers were consistently taller, older, and leaner than outfield players. These differences reflect known expectations for Goalkeepers, yet they were found to have little effect on expert performance ratings. It is possible that these demographic traits benefit players earlier in their careers (e.g. height advantages for inexperienced goalkeepers), which would explain why the majority of players were substantially taller than the average population. Nevertheless, these effects were not found in the top national and international competitions, showing that other performance indicators should be used at professional levels.
- **Extreme values suggest a recruitment bias:** Comparisons between the top and bottom 2.5% of demographic measures showed marginal and mostly insignificant expert rating differences. In the case of Goalkeepers, players who did not fit the traditional Goalkeeper-archetype even performed slightly better. This suggests that recruitment practices might be biased towards stereotypical profiling, where players that deviate from these profiles must outperform their peers to be selected.
- **Performance indicators provide stronger basis for scouting:** Role-specific performance indicators, such as saves for Goalkeepers, clearances for Defenders, and assists for Midfielders, were substantially more predictive of expert performance ratings than demographic variables. While more research is needed to determine the significance of demographic differences in early career performance, data-driven evaluations of match performance offer more actionable insights for scouting, team formation, and role-specific training at a professional level.