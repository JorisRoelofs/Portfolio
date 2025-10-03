//SETUP
cd "[enter folder name]"
import delimited "FIFA player database + demographics.csv", encoding(UTF-8) numericcols(130 132 133 136 137 140 142 143 146 147 150 152 153 156 157 160 162 163 166 167 170 171 172 173)


//RENAME VARIABLES
//Demographics
rename v1 case_id
rename v130 idfifa
rename v131 nationality
rename v132 height
rename v133 weight
rename v134 foot
rename v135 bday
rename v137 age
rename v138 preferredposition

//Age each year
rename v136 age16
rename v146 age17
drop v156 //identical to age17
rename v166 age18

//Weight each year
rename v170 weight16
rename v171 weight17
rename v172 weight18
rename v173 weight19


//CREATE VARIABLES
//Leftfootedness
gen leftfoot = 1 if foot == "L"
replace leftfoot = 0 if foot == "R"
drop foot

//Year of record
gen year = 17
replace year = 16 if competition == "Euro 2016"
replace year = 18 if competition == "World Cup 2018"

//Age during record
gen agecup = age17
replace agecup = age16 if year == 16
replace agecup = age18 if year == 18

//Weight during record
gen weightcup = weight17
replace weightcup = weight16 if year == 16
replace weightcup = weight18 if year == 18

//BMI
gen bmi = weightcup / ((0.01 * height)^2)

//Player Position
gen pos = .
replace pos = 0 if position == "Goalkeeper"
replace pos = 1 if position == "Defender"
replace pos = 2 if position == "Midfielder"
replace pos = 3 if position == "Forward"
label define poslabel 0 "Goalkeeper" 1 "Defender" 2 "Midfielder" 3 "Forward"
label values pos poslabel
tab position pos


//DROP DUPLICATE VARIABLES
pwcorr weight v143 v153 v163
drop v*
//as all new variables except 'age per year' and 'weight per year' remain constant over the years, all duplicate variables are dropped


//CHANGE VARIABLE ORDER
order age age16 age17 age18 weight16 weight17 weight18 weight19, after(bday)
order leftfoot preferredposition, before(bday)


//CHECK FOR DUPLICATES
sort player_id
quietly by player_id:  gen dup = cond(_N==1,0,_n)
tab dup


//DROP NON-HUMAN RATINGS (e.g. The Guardian's algorithm)
tab rat is_human
drop if !is_human


//VERIFY RATINGS ARE STANDARDIZED
sum rating if kicker
sum rating if bild
sum rating if skysports


//SAVE FILE
//save "FIFA regression.dta", replace