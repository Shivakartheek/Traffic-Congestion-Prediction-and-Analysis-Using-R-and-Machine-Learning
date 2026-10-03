# =============================================================================
# R/analysis.R
# Statistical analysis of the traffic data.
#   compute_statistics()    -> mean, median, SD, min, max, correlations
#   interpret_statistics()  -> explains the numbers in simple English
#   numeric_summary()       -> summary table for every numeric column
#   congestion_profile()    -> average conditions for Low / Medium / High
# =============================================================================

library(dplyr)
library(tibble)

# ---- Main statistics required by the project ---------------------------------
compute_statistics <- function(df) {
  vc <- df$Vehicle_Count

  table <- tibble(
    Statistic = c("Mean vehicle count", "Median vehicle count",
                  "Standard deviation of vehicle count",
                  "Minimum vehicle count", "Maximum vehicle count",
                  "Average speed (km/h)"),
    Value = c(mean(vc), median(vc), sd(vc), min(vc), max(vc),
              mean(df$Average_Speed))
  ) %>% mutate(Value = round(Value, 2))

  list(
    table         = table,
    mean_vc       = mean(vc),
    median_vc     = median(vc),
    sd_vc         = sd(vc),
    min_vc        = min(vc),
    max_vc        = max(vc),
    mean_speed    = mean(df$Average_Speed),
    cor_vc_speed  = cor(df$Vehicle_Count, df$Average_Speed),
    cor_rain_vc   = cor(df$Rainfall_mm,  df$Vehicle_Count)
  )
}

# ---- Turn a correlation number into words --------------------------------------
describe_correlation <- function(r) {
  if (is.na(r)) return("could not be calculated")
  a        <- abs(r)
  strength <- dplyr::case_when(a < 0.2 ~ "very weak",
                               a < 0.4 ~ "weak",
                               a < 0.6 ~ "moderate",
                               a < 0.8 ~ "strong",
                               TRUE    ~ "very strong")
  direction <- ifelse(r < 0, "negative", "positive")
  paste(strength, direction)
}

# ---- Simple-language explanation of the results ------------------------------------
# The sentences are built from the real computed values, so they always match
# the data (nothing is typed in by hand).
interpret_statistics <- function(stats) {
  cv <- stats$sd_vc / stats$mean_vc      # relative spread of the vehicle count

  spread_text <- if (cv > 0.5) {
    "The spread is large, so traffic volume changes a lot from one hour to another."
  } else if (cv > 0.25) {
    "The spread is moderate, so traffic volume changes noticeably during the day."
  } else {
    "The spread is small, so traffic volume is fairly stable."
  }

  r1 <- stats$cor_vc_speed
  r2 <- stats$cor_rain_vc

  speed_text <- if (!is.na(r1) && r1 < -0.3) {
    "This means that when more vehicles are on the road, vehicles move more slowly."
  } else if (!is.na(r1) && r1 > 0.3) {
    "This means that speed tends to rise together with vehicle count."
  } else {
    "This means vehicle count alone does not fully explain speed."
  }

  rain_text <- if (!is.na(r2) && abs(r2) < 0.2) {
    "Rainfall has little direct link with how many vehicles are counted; its effect is seen mainly in slower speeds and accidents."
  } else if (!is.na(r2) && r2 < 0) {
    "Vehicle counts tend to be lower on rainy periods."
  } else {
    "Vehicle counts tend to be higher on rainy periods."
  }

  c(
    paste0("Vehicle count: the mean is ", round(stats$mean_vc, 1),
           " and the median is ", round(stats$median_vc, 1),
           " vehicles per hour. The standard deviation is ",
           round(stats$sd_vc, 1), ", with counts ranging from ",
           stats$min_vc, " to ", stats$max_vc, ". ", spread_text),
    paste0("Average speed: traffic moves at ", round(stats$mean_speed, 1),
           " km/h on average."),
    paste0("Vehicle count vs speed: the correlation is ", round(r1, 3),
           " (", describe_correlation(r1), "). ", speed_text),
    paste0("Rainfall vs vehicle count: the correlation is ", round(r2, 3),
           " (", describe_correlation(r2), "). ", rain_text)
  )
}

# ---- Summary of every numeric column ----------------------------------------------------
numeric_summary <- function(df) {
  cols <- c("Hour", "Vehicle_Count", "Average_Speed", "Rainfall_mm",
            "Temperature_C", "Traffic_Signal_Count", "Road_Width")
  dplyr::bind_rows(lapply(cols, function(col) {
    x <- df[[col]]
    tibble(Variable = col,
           Mean   = round(mean(x), 2),
           Median = round(median(x), 2),
           SD     = round(sd(x), 2),
           Min    = round(min(x), 2),
           Max    = round(max(x), 2))
  }))
}

# ---- Average conditions for each congestion level ------------------------------------------
congestion_profile <- function(df) {
  df %>%
    group_by(Congestion_Level) %>%
    summarise(Records        = n(),
              Avg_Vehicles   = round(mean(Vehicle_Count), 1),
              Avg_Speed_kmh  = round(mean(Average_Speed), 1),
              Avg_Rainfall   = round(mean(Rainfall_mm), 2),
              Accident_Rate  = paste0(round(100 * mean(Accident == "Yes"), 1), "%"),
              .groups = "drop")
}
