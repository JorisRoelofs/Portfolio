/*SETUP*/
cd "[Enter folder directory]"
use Processed Dataset.dta, clear

/*ROLE DIFFERENCES REGRESSIONS*/
* Appendix B part 4 from the report
{
forvalues i = 0/4 {
if `i' == 0 {
di "/////////////////////////////// AGE PER POSITION ///////////////////////////////"
reg agecup i.pos if dup <= 1, robust cluster(player_id)
//reg agecup i.pos if dup <= 1, robust beta
}
if `i' == 1 {
di "/////////////////////////////// HEIGHT PER POSITION ///////////////////////////////"
reg height i.pos if dup <= 1, robust cluster(player_id)
//reg height i.pos if dup <= 1, robust beta
}
if `i' == 2 {
di "/////////////////////////////// WEIGHT PER POSITION ///////////////////////////////"
reg weightcup i.pos if dup <= 1, robust cluster(player_id)
//reg weightcup i.pos if dup <= 1, robust beta
}
if `i' == 3 {
di "/////////////////////////////// BMI PER POSITION ///////////////////////////////"
reg bmi i.pos if dup <= 1, robust cluster(player_id)
//reg bmi i.pos if dup <= 1, robust beta
}
if `i' == 4 {
di "/////////////////////////////// LEFTFOOT PER POSITION ///////////////////////////////"
reg leftfoot i.pos if dup <= 1, robust cluster(player_id)
//reg leftfoot i.pos if dup <= 1, robust beta
}

//ASSUMPTIONS CHECKING
//Fundamental uncertainty
predict stdp, stdp
replace stdp = 2 * stdp
tab stdp

tab pos

//Outliers = NEGLIGIBLE
predict res, residual
egen stdres = std(res)
di "5% cutoff: " round(_N * 0.05)
sum stdres if abs(stdres) > 1.96
di "1% cutoff: " round(_N * 0.01)
sum stdres if abs(stdres) > 2.5
di "0.1% cutoff: " round(_N * 0.001)
sum stdres if abs(stdres) > 3.29

//Linearity = N.A., x-variables are dummies

//All relevant predictors included = TRUE

//No multi-colinearity = N.A., x-variables are dummies

//Evenly distributed residuals = TRUE, but N.A. with robust
/*imtest, white
hettest
hettest, rhs*/

//Independent errors = LIKELY

//Normally distributed residuals = FALSE (non-normal kurtosis), use robust
/*sktest res
swilk res*/

drop stdp res stdres
}

//BOX PLOTS - POSITION DIFFERENCES
* Figure 1 from the report

graph box agecup, over(pos) title("Age per role")
graph export graph_age.pdf, replace
graph box height, over(pos) title("Height per role")
graph export graph_height.pdf, replace
graph box weightcup, over(pos) title("Weight per role")
graph export graph_weight.pdf, replace
graph box bmi, over(pos) title("BMI per role") //pretty much a less extreme version of the height differences
graph export graph_bmi.pdf, replace

gen h2 = (0.01*height)^2
pwcorr agecup weightcup height h2 bmi //so bmi isn't based solely on height, it even correlates more with weight. Probably because weight has a larger coefficient of variation
drop h2


//BOX PLOTS - COMPETITION DIFFERENCES
* Figure 2 from the report

graph box agecup, over(competition)
graph export graph_age_cup.pdf, replace
graph box height, over(competition)
graph export graph_height_cup.pdf, replace
graph box weightcup, over(competition)
graph export graph_weight_cup.pdf, replace
graph box bmi, over(competition)
graph export graph_bmi_cup.pdf, replace
//more variance in age between cups than expected, but nothing noteworty here
}


/*RATING PREDICTION*/
//ESTABLISHING OTHER MODELS
{
/*
forvalues i = 0/3 {
preserve
if `i' == 0 drop if !goalkeeper
if `i' == 1 drop if !defender
if `i' == 2 drop if !midfielder
if `i' == 3 drop if !forward

keep if dup <= 2 //a bit crude, but close enough
tab dup

//EXTREMES CUTOFF POINTS
sum height
egen sdheight = sd(height)
egen mheight = mean(height)
di mheight + (2 * sdheight)
di mheight - (2 * sdheight)

sum agecup
egen sdage = sd(age)
egen mage = mean(agecup)
di mage + (2 * sdage)
di mage - (2 * sdage)

sum weightcup
egen sdweight = sd(weightcup)
egen mweight = mean(weightcup)
di mweight + (2 * sdweight)
di mweight - (2 * sdweight)

//ESTABLISHED PREDICTORS
if `i' == 0 reg rating height agecup weightcup leftfoot , beta

//Goalkeeper: ss_dangmistakes ss_goals_ag_otb ss_goals_ag_itb ss_saves_itb team_goals team_rating opp_gk_rating opp_bestfw_rating

//Forward: ss_goals ss_touches team_rating

//Midfielder: ss_goals ss_assists ss_passes_acc ss_crosses_acc team_rating team_pos_rating

//Defender: ss_goals ss_assists ss_chances2score ss_clearances team_rating team_pos_rating opp_goals
restore
}
*/
}

//REGRESSIONS
* Appendix B part 1-3 from the report
{
gen tall = .
gen short = .
gen old = .
gen young = .
gen heavy = .
gen light = .
replace tall = 0 if height != .
replace short = 0 if height != .
replace old = 0 if agecup != .
replace young = 0 if agecup != .
replace heavy = 0 if weightcup != .
replace light = 0 if weightcup != .

forvalues i = 0/3 {
preserve

if `i' == 0 { //Goalkeeper
drop if !goalkeeper
replace tall = 1 if height > 199
replace short = 1 if height <= 182
replace old = 1 if agecup > 36
replace young = 1 if agecup <= 22
replace heavy = 1 if weightcup > 90
replace light = 1 if weightcup <= 62
di "/////////////////////////////////////////// GOALKEEPER ///////////////////////////////////////////"
}
if `i' == 1 { //Defender
drop if !defender
replace tall = 1 if height > 196
replace short = 1 if height <= 173
replace old = 1 if agecup > 34
replace young = 1 if agecup <= 19
replace heavy = 1 if weightcup > 91
replace light = 1 if weightcup <= 63
di "/////////////////////////////////////////// DEFENDER ///////////////////////////////////////////"
}
if `i' == 2 { //Midfielder
drop if !midfielder
replace tall = 1 if height > 191
replace short = 1 if height <= 168
replace old = 1 if agecup > 33
replace young = 1 if agecup <= 19
replace heavy = 1 if weightcup > 91
replace light = 1 if weightcup <= 62
di "/////////////////////////////////////////// MIDFIELDER ///////////////////////////////////////////"
}
if `i' == 3 { //Forward
drop if !forward
replace tall = 1 if height > 196
replace short = 1 if height <= 170
replace old = 1 if agecup > 34
replace young = 1 if agecup <= 18
replace heavy = 1 if weightcup > 91
replace light = 1 if weightcup <= 62
di "/////////////////////////////////////////// FORWARD ///////////////////////////////////////////"
}

sum tall short old young heavy light if dup <= 1

//why does sum give different number of observations between these variables whereas tab does not?

gen gtall = tall * team_rating
gen gshort = short * team_rating
gen gold = old * team_rating
gen gyoung = young * team_rating
gen gheavy = heavy * team_rating
gen glight = light * team_rating
gen ttall = tall * team_rating
gen tshort = short * team_rating
gen told = old * team_rating
gen tyoung = young * team_rating
gen theavy = heavy * team_rating
gen tlight = light * team_rating

forvalues j = 0/4 {
if `j' == 0 {
di "//// REGRESSION - DEMOGRAPHICS"
reg rating height agecup weightcup leftfoot, robust cluster(player_id)
}
if `j' == 1 {
di "//// REGRESSION - EXTREME DEMOGRAPHICS"
reg rating tall short old young heavy light leftfoot, robust cluster(player_id)
}
if `j' == 2 {
di "//// REGRESSION - ESTABLISHED PREDICTORS"
//Goalkeeper
if `i' == 0 reg rating height agecup weightcup leftfoot ss_dangmistakes ss_goals_ag_otb ss_goals_ag_itb ss_saves_itb team_goals team_rating opp_gk_rating opp_bestfw_rating, robust cluster(player_id)
//Defender
if `i' == 1 reg rating height agecup weightcup leftfoot ss_goals ss_assists ss_chances2score ss_clearances team_rating team_pos_rating opp_goals, robust cluster(player_id)
//Midfielder
if `i' == 2 reg rating height agecup weightcup leftfoot ss_goals ss_assists ss_passes_acc ss_crosses_acc team_rating team_pos_rating, robust cluster(player_id)
//Forward
if `i' == 3 reg rating height agecup weightcup leftfoot ss_goals ss_touches team_rating, robust cluster(player_id)
}
if `j' == 3 {
di "//// REGRESSION - INTERACTION - TEAM RATING"
reg rating ttall tshort told tyoung theavy tlight leftfoot, robust cluster(player_id)
}
if `j' == 4 {
di "//// REGRESSION - INTERACTION - GOALS"
reg rating gtall gshort gold gyoung gheavy glight leftfoot, robust cluster(player_id)
}

di "ASSUMPTION CHECKING"
di "OUTLIER COUNT"
predict res, residual
egen stdres = std(res)
di "5% cutoff: " round(_N * 0.05)
sum stdres if abs(stdres) > 1.96 // < 5%
di "1% cutoff: " round(_N * 0.01)
sum stdres if abs(stdres) > 2.5 // < 1%
di "0.1% cutoff: " round(_N * 0.001)
sum stdres if abs(stdres) > 3.29 // < 0.1%

di "MULTI-COLINEARITY"
if `i' == 0 pwcorr rating height agecup weightcup leftfoot team_rating ss_goals, obs sig star(5) //besides the role-specific established predictors all other variables are dummies or interactions with dummies
vif

di "VARIABLE INCLUSION & LINEARITY"
if `i' == 0 fracpoly reg rating height agecup weightcup leftfoot, robust cluster(player_id)
ovtest

//Heteroscedasticity does not hold so robust is used
//Independent errors does not hold so clustering is used

/*
di "RESIDUAL NORMALITY" //N.A. because robust
sktest res
swilk res
*/
drop res stdres
}
restore
}

drop tall short old young heavy light
}

