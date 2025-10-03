# Setup
## Clear data
rm(list = ls())

## Add libraries
library(clubSandwich)
library(dplyr)
library(haven)
library(ggplot2)
library(Hmisc)
library(lmtest)
library(tidyr)
library(reshape2)
library(patchwork)

## Open dataset
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
data <- read_dta("Processed Dataset.dta")

## Remove players with unknown positions
data <- data %>% filter(position != "")


# Count unique players
length(unique(data$player_id[!is.na(data$player_id)]))


# Midfielder height vs average population
midfielders <- data[data$position == "Midfielder", ]
midfielders$height_centered <- midfielders$height - 175

## Clustered robust one-sample t-test
model <- lm(height_centered ~ 1, data = midfielders)
coeftest(model, vcov = vcovCR(model, cluster = midfielders$player_id, type = "CR2"))

## Clustered robust Cohen’s d
vc <- sqrt(diag(vcovCR(model, cluster = midfielders$player_id, type = "CR2")))
Cd <- coef(model) / vc
Cd


# Box plots
vars <- c("height", "weightcup", "agecup", "bmi")

## Remove time-dependence from role dataset by computing player averages
player_avg <- data %>%
  group_by(player_id, position) %>%
  summarise(across(all_of(vars), ~mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  mutate(position = factor(position, levels = c("Goalkeeper", "Defender", "Midfielder", "Forward")))

## Remove duplicates from competition dataset
player_cup <- data %>%
  group_by(competition, player_id) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(competition = factor(competition, levels = c("Euro 2016", "Bundesliga 2017-18", "Premier League 2017-18", "World Cup 2018"), labels = c("Euro\n2016", "Bundesliga\n2017-2018", "Premier\nLeague\n2017-2018", "World\nCup\n2018")))

## Create boxplot function
make_boxplot <- function(dataset, var_x, var_y, var_label, unit, major_step = 10, minor_step = 5) {
  ### Set fill colors
  fill_colors <- c(
      "Goalkeeper" = "#00BA38",
      "Defender"   = "#ced526",
      "Midfielder" = "#d29f3c",
      "Forward"    = "#d7201e",
      "Euro\n2016"                 = "#54647e",
      "Bundesliga\n2017-2018"      = "#D3010C",
      "Premier\nLeague\n2017-2018" = "#38003c",
      "World\nCup\n2018"           = "#005391"
  )

  ### Plot boxplot
  ggplot(dataset, aes(x = .data[[var_x]], y = .data[[var_y]], fill = .data[[var_x]])) +
    geom_boxplot(alpha = 1, outlier.color = "black") +
    scale_fill_manual(values = fill_colors) +
    scale_y_continuous(
      breaks = scales::breaks_width(major_step),
      minor_breaks = scales::breaks_width(minor_step),
      labels = function(x) paste0(x, unit)
    ) +
    theme_minimal() +
    labs(x = "", y = var_label, title = var_label) +
    theme(
      legend.position = "none",
      axis.title.y = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0.5)
    )
}

## Demographics boxplots by role
p1 <- make_boxplot(player_avg, "position", "height", "Height", " cm")
p2 <- make_boxplot(player_avg, "position", "weightcup", "Weight", " kg")
p3 <- make_boxplot(player_avg, "position", "agecup", "Age", " yo")
p4 <- make_boxplot(player_avg, "position", "bmi", "BMI", expression(" kg/m^2"))
(p1 | p2 | p3 | p4)

## Demographics boxplots by cup
p1 <- make_boxplot(player_cup, "competition", "height", "Height", " cm")
p2 <- make_boxplot(player_cup, "competition", "weightcup", "Weight", " kg")
p3 <- make_boxplot(player_cup, "competition", "agecup", "Age", " yo")
p4 <- make_boxplot(player_cup, "competition", "bmi", "BMI", expression(" kg/m^2"))
(p1 | p2 | p3 | p4)