# =============================================================================
# SINGLE-FILE VERSION of the Traffic Congestion project
# Put this file in its own folder (as app.R) and click 'Run App'.
# It contains ALL code: data generation, preprocessing, statistics, graphs,
# model training and the Shiny dashboard. It creates data/ and models/ folders
# next to itself on the first run.
# =============================================================================

# #############################################################################
# PART: R/data_preprocessing.R
# #############################################################################
# =============================================================================
# R/data_preprocessing.R
# Traffic Congestion Prediction and Analysis Using R and Machine Learning
#
# This file contains everything related to DATA:
#   1. Constants (allowed categories, valid numeric ranges)
#   2. generate_traffic_data()   -> creates the SYNTHETIC dataset
#   3. load_or_create_dataset()  -> loads the CSV (or creates it if missing)
#   4. preprocess_traffic_data() -> cleans the data for machine learning
#   5. validate_inputs() / build_new_observation() -> used by the Shiny app so
#      that new user input gets EXACTLY the same treatment as the training data
# =============================================================================

library(readr)
library(dplyr)
library(tibble)
library(stringr)

# -----------------------------------------------------------------------------
# 1. CONSTANTS
# -----------------------------------------------------------------------------
DAY_LEVELS   <- c("Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday")
ROAD_LEVELS  <- c("Highway", "City Road", "Arterial Road", "Residential Street")
YESNO_LEVELS <- c("No", "Yes")
CONG_LEVELS  <- c("Low", "Medium", "High")

NUMERIC_COLS <- c("Hour", "Vehicle_Count", "Average_Speed", "Rainfall_mm",
                  "Temperature_C", "Traffic_Signal_Count", "Road_Width")

# Physically sensible ranges. Values outside these are treated as data errors.
# The same table is used to validate what the user types in the Shiny app.
INPUT_RULES <- tibble::tribble(
  ~field,                 ~label,                          ~min,  ~max, ~integer,
  "Hour",                 "Hour (0-23)",                      0,    23,     TRUE,
  "Vehicle_Count",        "Vehicle count (per hour)",         0,  5000,     TRUE,
  "Average_Speed",        "Average speed (km/h)",             1,   130,    FALSE,
  "Rainfall_mm",          "Rainfall (mm)",                    0,   300,    FALSE,
  "Temperature_C",        "Temperature (deg C)",            -10,    55,    FALSE,
  "Traffic_Signal_Count", "Traffic signal count",             0,    30,     TRUE,
  "Road_Width",           "Road width (m)",                   2,    40,    FALSE
)

# -----------------------------------------------------------------------------
# 2. SYNTHETIC DATA GENERATOR
# -----------------------------------------------------------------------------
# No real dataset is used. The data is SYNTHETIC: it is created with random
# numbers that follow realistic traffic rules (rush hours, rain slows traffic,
# accidents cause jams, weekends are quieter, wide roads carry more vehicles).
# set.seed() makes the result identical every time the code is run.
#
# add_noise_issues = TRUE deliberately adds a few missing values, invalid
# values and duplicate rows so that the preprocessing step has real work to do.
generate_traffic_data <- function(n = 1500, seed = 123, add_noise_issues = TRUE) {
  set.seed(seed)

  # ---- Date, day of week and holiday -----------------------------------------
  all_days <- seq(as.Date("2026-01-01"), as.Date("2026-06-30"), by = "day")
  dates    <- sample(all_days, n, replace = TRUE)
  # "%u" gives 1 = Monday ... 7 = Sunday (independent of the computer language)
  day_of_week <- DAY_LEVELS[as.integer(format(dates, "%u"))]
  month       <- as.integer(format(dates, "%m"))

  # A fixed list of illustrative holiday dates (not an official calendar)
  holiday_dates <- as.Date(c("2026-01-01", "2026-01-26", "2026-03-04",
                             "2026-03-20", "2026-04-03", "2026-05-01",
                             "2026-05-27", "2026-06-26"))
  holiday <- ifelse(dates %in% holiday_dates, "Yes", "No")
  off_day <- day_of_week %in% c("Saturday", "Sunday") | holiday == "Yes"

  # ---- Hour and time -----------------------------------------------------------
  # More records are sampled in the morning (around 9) and evening (around 18)
  h <- 0:23
  hour_weight <- 0.4 + 2.0 * exp(-((h - 9)^2)  / (2 * 1.5^2)) +
                       2.2 * exp(-((h - 18)^2) / (2 * 1.8^2))
  hour_weight[h <= 5] <- hour_weight[h <= 5] * 0.3
  hour   <- sample(h, n, replace = TRUE, prob = hour_weight)
  minute <- sample(0:59, n, replace = TRUE)
  time   <- sprintf("%02d:%02d", hour, minute)

  # "Peak factor" between 0 and 1: how busy this hour normally is
  peak <- 0.15 + 0.85 * pmax(exp(-((hour - 9)^2)  / (2 * 1.6^2)),
                             exp(-((hour - 18)^2) / (2 * 1.8^2))) +
          0.25 * exp(-((hour - 13)^2) / (2 * 2^2))
  peak <- pmin(1, peak)

  # ---- Road properties -----------------------------------------------------------
  road <- sample(ROAD_LEVELS, n, replace = TRUE, prob = c(0.25, 0.35, 0.25, 0.15))

  sig_min <- c("Highway" = 0,  "City Road" = 4,  "Arterial Road" = 3, "Residential Street" = 0)
  sig_max <- c("Highway" = 2,  "City Road" = 10, "Arterial Road" = 8, "Residential Street" = 3)
  w_min   <- c("Highway" = 14, "City Road" = 8,  "Arterial Road" = 10, "Residential Street" = 5)
  w_max   <- c("Highway" = 20, "City Road" = 14, "Arterial Road" = 16, "Residential Street" = 8)
  free_flow_speed <- c("Highway" = 80, "City Road" = 45, "Arterial Road" = 55, "Residential Street" = 30)
  type_multiplier <- c("Highway" = 1.15, "City Road" = 0.95, "Arterial Road" = 1.0, "Residential Street" = 0.45)

  signals <- floor(runif(n, unname(sig_min[road]), unname(sig_max[road]) + 1))
  width   <- round(runif(n, unname(w_min[road]), unname(w_max[road])), 1)
  ff      <- unname(free_flow_speed[road])

  # ---- Weather ---------------------------------------------------------------------
  rain_prob <- c(0.10, 0.10, 0.15, 0.20, 0.25, 0.40)[month]   # Jan..Jun
  rainy     <- rbinom(n, 1, rain_prob)
  rainfall  <- round(rainy * pmin(rexp(n, rate = 1 / 6) + 0.2, 60), 1)
  temp_mean <- c(18, 21, 26, 31, 34, 32)[month]
  temperature <- round(temp_mean + rnorm(n, 0, 2.5), 1)

  # ---- Accident ----------------------------------------------------------------------
  accident_flag <- rbinom(n, 1, 0.05 + 0.10 * peak)
  accident      <- ifelse(accident_flag == 1, "Yes", "No")

  # ---- Vehicle count (vehicles per hour) -----------------------------------------------
  demand_factor <- ifelse(off_day, 0.70, 1.00)
  vehicles <- (150 + 1100 * peak) * unname(type_multiplier[road]) * demand_factor *
              (1 - 0.12 * (rainfall > 0)) + rnorm(n, 0, 70)
  vehicles <- as.integer(round(pmax(50, vehicles)))

  # ---- Average speed (km/h) ----------------------------------------------------------------
  capacity    <- 90 * width                      # rough road capacity
  utilisation <- vehicles / capacity             # 1.0 means the road is "full"
  speed <- ff * (1 - 0.55 * pmin(utilisation, 1.2)) *
           (1 - 0.25 * accident_flag) *
           (1 - 0.004 * pmin(rainfall, 40)) -
           0.5 * signals + rnorm(n, 0, 2.5)
  speed <- round(pmin(pmax(speed, 5), ff), 1)

  # ---- Congestion level (the target we want to predict) ------------------------------------
  # A congestion "score" is built from the traffic conditions plus some random
  # noise. The score is split by percentiles: lowest 40% = Low,
  # next 35% = Medium, top 25% = High.
  score <- 0.90 * pmin(utilisation, 1.3) +
           0.60 * (1 - speed / ff) +
           0.12 * accident_flag +
           0.004 * pmin(rainfall, 40) +
           0.010 * signals +
           rnorm(n, 0, 0.06)
  cut_points <- quantile(score, c(0.40, 0.75))
  congestion <- as.character(cut(score, breaks = c(-Inf, cut_points, Inf),
                                 labels = CONG_LEVELS))

  df <- tibble(
    Date                 = format(dates, "%Y-%m-%d"),
    Time                 = time,
    Day_of_Week          = day_of_week,
    Hour                 = as.integer(hour),
    Vehicle_Count        = vehicles,
    Average_Speed        = speed,
    Rainfall_mm          = rainfall,
    Temperature_C        = temperature,
    Accident             = accident,
    Road_Type            = road,
    Holiday              = holiday,
    Traffic_Signal_Count = as.integer(signals),
    Road_Width           = width,
    Congestion_Level     = congestion
  )

  # ---- Deliberately "dirty" the data a little (for the preprocessing demo) --------------
  if (add_noise_issues) {
    df$Average_Speed[sample(n, 30)]        <- NA   # missing values
    df$Temperature_C[sample(n, 25)]        <- NA
    df$Rainfall_mm[sample(n, 20)]          <- NA
    df$Road_Width[sample(n, 10)]           <- NA
    df$Vehicle_Count[sample(n, 15)]        <- NA
    df$Average_Speed[sample(n, 3)]         <- c(-5, 250, 400)   # impossible speeds
    df$Vehicle_Count[sample(n, 2)]         <- c(-100L, 9000L)   # impossible counts
    duplicates <- df[sample(n, 15), ]                           # 15 duplicate rows
    df <- bind_rows(df, duplicates)
  }

  df %>% arrange(Date, Time)
}

# -----------------------------------------------------------------------------
# 3. LOAD THE CSV (or create it when it does not exist)
# -----------------------------------------------------------------------------
load_or_create_dataset <- function(paths = c("data/traffic_data.csv", "traffic_data.csv")) {
  # Time is read as text ("08:30") and Date is converted later in preprocessing
  col_spec <- readr::cols(.default = readr::col_guess(),
                          Date = readr::col_character(),
                          Time = readr::col_character())

  for (p in paths) {
    if (file.exists(p)) {
      raw <- readr::read_csv(p, col_types = col_spec, show_col_types = FALSE)
      return(list(data = raw, source = p, generated = FALSE))
    }
  }

  # Dataset missing -> create it so the project still runs
  message("Dataset file not found. Generating the synthetic dataset now ...")
  raw <- generate_traffic_data()
  dir.create("data", showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(raw, "data/traffic_data.csv")
  list(data = raw, source = "data/traffic_data.csv (auto-generated)", generated = TRUE)
}

# -----------------------------------------------------------------------------
# 4. PREPROCESSING
# -----------------------------------------------------------------------------
# Small helper: a table with basic facts about a data frame
dataset_info <- function(df) {
  tibble(
    Metric = c("Rows (records)", "Columns (variables)",
               "Missing values", "Duplicate rows"),
    Value  = c(nrow(df), ncol(df), sum(is.na(df)), sum(duplicated(df)))
  )
}

# Most frequent value of a vector (used to fill missing categories)
most_common <- function(x) {
  x <- x[!is.na(x)]
  names(sort(table(x), decreasing = TRUE))[1]
}

preprocess_traffic_data <- function(raw, verbose = TRUE) {
  say <- function(...) if (verbose) cat(..., "\n")

  # The CSV must contain all required columns
  required <- c("Date", "Time", "Day_of_Week", "Hour", "Vehicle_Count", "Average_Speed",
                "Rainfall_mm", "Temperature_C", "Accident", "Road_Type", "Holiday",
                "Traffic_Signal_Count", "Road_Width", "Congestion_Level")
  missing_cols <- setdiff(required, names(raw))
  if (length(missing_cols) > 0) {
    stop("The dataset is missing these required columns: ", paste(missing_cols, collapse = ", "))
  }

  # ---- BEFORE cleaning ----------------------------------------------------------
  before <- dataset_info(raw)
  say("\n===== DATASET STRUCTURE (BEFORE CLEANING) =====")
  if (verbose) dplyr::glimpse(raw)
  say("\n===== MISSING VALUES PER COLUMN (BEFORE) =====")
  missing_before <- colSums(is.na(raw))
  if (verbose) print(missing_before)

  df <- raw

  # ---- Step 1: tidy the text columns (trim spaces, "yes" -> "Yes") -------------------
  text_cols <- c("Day_of_Week", "Accident", "Road_Type", "Holiday", "Congestion_Level")
  for (col in intersect(text_cols, names(df))) {
    df[[col]] <- stringr::str_to_title(stringr::str_trim(as.character(df[[col]])))
  }

  # ---- Step 2: convert data types ------------------------------------------------------
  df$Date <- as.Date(as.character(df$Date))
  for (col in intersect(NUMERIC_COLS, names(df))) {
    df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
  }
  # If Hour is missing, rebuild it from the Time column ("08:30" -> 8)
  from_time <- suppressWarnings(as.numeric(substr(as.character(df$Time), 1, 2)))
  df$Hour   <- ifelse(is.na(df$Hour), from_time, df$Hour)

  # ---- Step 3: remove duplicate records ---------------------------------------------------
  n_before_dupes <- nrow(df)
  df <- dplyr::distinct(df)
  duplicates_removed <- n_before_dupes - nrow(df)

  # ---- Step 4: detect impossible values (set to NA so they get imputed) ---------------------
  invalid_fixed <- 0
  for (i in seq_len(nrow(INPUT_RULES))) {
    f   <- INPUT_RULES$field[i]
    bad <- !is.na(df[[f]]) & (df[[f]] < INPUT_RULES$min[i] | df[[f]] > INPUT_RULES$max[i])
    invalid_fixed <- invalid_fixed + sum(bad)
    df[[f]][bad]  <- NA
  }

  # Rows without a target label cannot be used for training -> remove
  df <- df[!is.na(df$Congestion_Level) & df$Congestion_Level %in% CONG_LEVELS, ]

  # ---- Step 5: handle missing values ------------------------------------------------------------
  # Numbers  -> replaced by the MEDIAN (robust against outliers)
  # Categories -> replaced by the MOST COMMON value (mode)
  medians <- sapply(df[NUMERIC_COLS], median, na.rm = TRUE)
  modes   <- sapply(df[c("Day_of_Week", "Accident", "Road_Type", "Holiday")], most_common)
  for (col in NUMERIC_COLS) df[[col]][is.na(df[[col]])] <- medians[[col]]
  for (col in names(modes)) df[[col]][is.na(df[[col]])] <- modes[[col]]

  # ---- Step 6: final data types (factors for categories) ---------------------------------------------
  df <- df %>%
    mutate(
      Day_of_Week          = factor(Day_of_Week, levels = DAY_LEVELS),
      Accident             = factor(Accident,    levels = YESNO_LEVELS),
      Road_Type            = factor(Road_Type,   levels = ROAD_LEVELS),
      Holiday              = factor(Holiday,     levels = YESNO_LEVELS),
      Congestion_Level     = factor(Congestion_Level, levels = CONG_LEVELS),
      Hour                 = as.integer(Hour),
      Traffic_Signal_Count = as.integer(Traffic_Signal_Count),
      Vehicle_Count        = as.numeric(Vehicle_Count)
    )

  # Any row that still has a problem (e.g. an unknown category) is dropped
  df <- df[complete.cases(df), ]

  # ---- Step 7: look for remaining statistical outliers (IQR rule) --------------------------------------
  # Extreme-but-possible values (e.g. very heavy rain) are KEPT because they are
  # real traffic situations. We only report them.
  outlier_table <- dplyr::bind_rows(lapply(
    c("Vehicle_Count", "Average_Speed", "Rainfall_mm", "Temperature_C"),
    function(col) {
      q     <- quantile(df[[col]], c(0.25, 0.75))
      iqr   <- q[2] - q[1]
      lower <- q[1] - 1.5 * iqr
      upper <- q[2] + 1.5 * iqr
      tibble(Variable = col, Lower_Limit = round(unname(lower), 2),
             Upper_Limit = round(unname(upper), 2),
             Outliers = sum(df[[col]] < lower | df[[col]] > upper))
    }))

  # ---- AFTER cleaning -------------------------------------------------------------------------------------
  after <- dataset_info(df)
  comparison <- tibble(
    Metric          = before$Metric,
    Before_Cleaning = before$Value,
    After_Cleaning  = after$Value
  )
  comparison <- bind_rows(comparison, tibble(
    Metric = c("Duplicate rows removed", "Invalid values corrected"),
    Before_Cleaning = NA_real_,
    After_Cleaning  = c(duplicates_removed, invalid_fixed)
  ))

  say("\n===== DATASET STRUCTURE (AFTER CLEANING) =====")
  if (verbose) dplyr::glimpse(df)
  say("\n===== BEFORE vs AFTER =====")
  if (verbose) print(as.data.frame(comparison))
  say("\n===== OUTLIER CHECK (IQR RULE) =====")
  if (verbose) print(as.data.frame(outlier_table))

  list(
    data               = df,
    comparison         = comparison,
    outliers           = outlier_table,
    missing_before     = missing_before,
    duplicates_removed = duplicates_removed,
    invalid_fixed      = invalid_fixed,
    medians            = medians,
    modes              = modes,
    encoded            = encode_categoricals(df)
  )
}

# ---- Encoding categorical variables ---------------------------------------------------
# Shows the numeric codes of each category. (The ML algorithms in R convert factors
# to numbers internally - "dummy variables" - so the same idea is applied
# automatically during model training.)
encode_categoricals <- function(df) {
  df %>%
    mutate(
      Day_Code       = as.integer(Day_of_Week),             # Monday = 1 ... Sunday = 7
      Accident_Code  = as.integer(Accident == "Yes"),       # No = 0, Yes = 1
      Holiday_Code   = as.integer(Holiday == "Yes"),        # No = 0, Yes = 1
      RoadType_Code  = as.integer(Road_Type),               # Highway = 1, City Road = 2 ...
      Congestion_Code = as.integer(Congestion_Level)        # Low = 1, Medium = 2, High = 3
    ) %>%
    select(Day_Code, Accident_Code, Holiday_Code, RoadType_Code, Congestion_Code,
           everything())
}

# -----------------------------------------------------------------------------
# 5. INPUT VALIDATION + NEW OBSERVATION (used by the Shiny prediction form)
# -----------------------------------------------------------------------------
# Returns a character vector of friendly error messages (empty vector = all OK)
validate_inputs <- function(vals) {
  errs <- character(0)

  # ---- categorical inputs ----
  cat_rules <- list(
    Day_of_Week = list(label = "Day of week", allowed = DAY_LEVELS),
    Accident    = list(label = "Accident",    allowed = YESNO_LEVELS),
    Road_Type   = list(label = "Road type",   allowed = ROAD_LEVELS),
    Holiday     = list(label = "Holiday",     allowed = YESNO_LEVELS)
  )
  for (f in names(cat_rules)) {
    v     <- vals[[f]]
    label <- cat_rules[[f]]$label
    if (is.null(v) || length(v) != 1 || is.na(v) || !nzchar(as.character(v))) {
      errs <- c(errs, paste0(label, " is empty. Please choose a value."))
    } else if (!(as.character(v) %in% cat_rules[[f]]$allowed)) {
      errs <- c(errs, paste0(label, " '", v, "' is not valid. Allowed values: ",
                             paste(cat_rules[[f]]$allowed, collapse = ", "), "."))
    }
  }

  # ---- numeric inputs ----
  for (i in seq_len(nrow(INPUT_RULES))) {
    f     <- INPUT_RULES$field[i]
    label <- INPUT_RULES$label[i]
    v     <- suppressWarnings(as.numeric(vals[[f]]))
    if (length(v) != 1 || is.na(v)) {
      errs <- c(errs, paste0(label, " is empty or not a number."))
    } else if (!is.finite(v)) {
      errs <- c(errs, paste0(label, " must be a finite number."))
    } else if (v < INPUT_RULES$min[i] || v > INPUT_RULES$max[i]) {
      errs <- c(errs, paste0(label, " must be between ", INPUT_RULES$min[i],
                             " and ", INPUT_RULES$max[i], "."))
    } else if (INPUT_RULES$integer[i] && abs(v - round(v)) > 1e-8) {
      errs <- c(errs, paste0(label, " must be a whole number."))
    }
  }
  errs
}

# Turn validated input values into a one-row data frame that looks EXACTLY like
# the cleaned training data (same column names, types and factor levels).
build_new_observation <- function(vals) {
  data.frame(
    Day_of_Week          = factor(as.character(vals$Day_of_Week), levels = DAY_LEVELS),
    Hour                 = as.integer(round(as.numeric(vals$Hour))),
    Vehicle_Count        = as.numeric(vals$Vehicle_Count),
    Average_Speed        = as.numeric(vals$Average_Speed),
    Rainfall_mm          = as.numeric(vals$Rainfall_mm),
    Temperature_C        = as.numeric(vals$Temperature_C),
    Accident             = factor(as.character(vals$Accident), levels = YESNO_LEVELS),
    Road_Type            = factor(as.character(vals$Road_Type), levels = ROAD_LEVELS),
    Holiday              = factor(as.character(vals$Holiday), levels = YESNO_LEVELS),
    Traffic_Signal_Count = as.integer(round(as.numeric(vals$Traffic_Signal_Count))),
    Road_Width           = as.numeric(vals$Road_Width),
    stringsAsFactors     = FALSE
  )
}


# #############################################################################
# PART: R/analysis.R
# #############################################################################
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


# #############################################################################
# PART: R/visualization.R
# #############################################################################
# =============================================================================
# R/visualization.R
# All graphs of the project (ggplot2). Every function takes a data frame and
# RETURNS a ggplot object, so the same code is used in the console script and
# in the Shiny dashboard.
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)

# Colours used everywhere: green = Low, amber = Medium, red = High
congestion_colors <- c("Low" = "#2e9e5b", "Medium" = "#f0a202", "High" = "#d64545")

# A clean, light theme shared by all plots
theme_traffic <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title       = element_text(face = "bold", colour = "#1f3b57", size = base_size + 3),
      plot.subtitle    = element_text(colour = "#5f6b7a"),
      axis.title       = element_text(colour = "#44546a"),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom",
      plot.background  = element_rect(fill = "white", colour = NA)
    )
}

# ---- 1. Vehicle count by hour -------------------------------------------------
plot_vehicles_by_hour <- function(df) {
  d <- df %>% group_by(Hour) %>%
    summarise(Avg_Vehicles = mean(Vehicle_Count), .groups = "drop")
  ggplot(d, aes(x = Hour, y = Avg_Vehicles)) +
    geom_col(fill = "#3b82c4", width = 0.8, alpha = 0.85) +
    geom_line(colour = "#1f3b57", linewidth = 1) +
    geom_point(colour = "#1f3b57", size = 2) +
    scale_x_continuous(breaks = 0:23) +
    labs(title = "Average Vehicle Count by Hour",
         subtitle = "Morning and evening rush hours are clearly visible",
         x = "Hour of day (0-23)", y = "Average vehicles per hour") +
    theme_traffic()
}

# ---- 2. Traffic by day of week --------------------------------------------------
plot_traffic_by_day <- function(df) {
  d <- df %>% group_by(Day_of_Week) %>%
    summarise(Avg_Vehicles = mean(Vehicle_Count), .groups = "drop")
  ggplot(d, aes(x = Day_of_Week, y = Avg_Vehicles)) +
    geom_col(fill = "#3b82c4", alpha = 0.85) +
    geom_text(aes(label = round(Avg_Vehicles)), vjust = -0.4, colour = "#1f3b57") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(title = "Average Vehicle Count by Day of Week",
         x = "Day of week", y = "Average vehicles per hour") +
    theme_traffic()
}

# ---- 3. Average speed by congestion level -----------------------------------------
plot_speed_by_congestion <- function(df) {
  ggplot(df, aes(x = Congestion_Level, y = Average_Speed, fill = Congestion_Level)) +
    geom_boxplot(alpha = 0.85, outlier.colour = "grey40") +
    scale_fill_manual(values = congestion_colors) +
    labs(title = "Average Speed by Congestion Level",
         subtitle = "Higher congestion means lower speed",
         x = "Congestion level", y = "Average speed (km/h)") +
    theme_traffic() + theme(legend.position = "none")
}

# ---- 4. Vehicle count vs average speed ----------------------------------------------
plot_vehicles_vs_speed <- function(df) {
  ggplot(df, aes(x = Vehicle_Count, y = Average_Speed)) +
    geom_point(aes(colour = Congestion_Level), alpha = 0.55, size = 1.8) +
    geom_smooth(method = "loess", formula = y ~ x, se = FALSE,
                colour = "#1f3b57", linewidth = 1) +
    scale_colour_manual(values = congestion_colors) +
    labs(title = "Vehicle Count vs Average Speed",
         subtitle = "More vehicles usually means slower traffic",
         x = "Vehicle count (per hour)", y = "Average speed (km/h)",
         colour = "Congestion") +
    theme_traffic()
}

# ---- A reusable "share of congestion levels" bar chart ---------------------------------
plot_congestion_share <- function(df, var, title, xlab, subtitle = NULL) {
  ggplot(df, aes(x = .data[[var]], fill = Congestion_Level)) +
    geom_bar(position = "fill", alpha = 0.9) +
    scale_y_continuous(labels = scales::percent) +
    scale_fill_manual(values = congestion_colors) +
    labs(title = title, subtitle = subtitle, x = xlab,
         y = "Share of records", fill = "Congestion") +
    theme_traffic()
}

# ---- 5. Rainfall vs congestion ----------------------------------------------------------
plot_rainfall_vs_congestion <- function(df) {
  d <- df %>%
    mutate(Rain_Category = cut(Rainfall_mm,
                               breaks = c(-Inf, 0, 2.5, 10, Inf),
                               labels = c("No rain", "Light (0-2.5 mm)",
                                          "Moderate (2.5-10 mm)", "Heavy (>10 mm)")))
  plot_congestion_share(d, "Rain_Category",
                        "Rainfall vs Congestion", "Rainfall category",
                        "Share of Low / Medium / High congestion in each rainfall group")
}

# ---- 6. Accident vs congestion ---------------------------------------------------------------
plot_accident_vs_congestion <- function(df) {
  plot_congestion_share(df, "Accident", "Accident vs Congestion",
                        "Accident reported?",
                        "Accidents sharply increase the share of High congestion")
}

# ---- 7. Holiday vs congestion ------------------------------------------------------------------
plot_holiday_vs_congestion <- function(df) {
  plot_congestion_share(df, "Holiday", "Holiday vs Congestion", "Holiday?",
                        "Holidays usually have less traffic")
}

# ---- 8. Congestion distribution ------------------------------------------------------------------
plot_congestion_distribution <- function(df) {
  d <- df %>% count(Congestion_Level) %>% mutate(pct = n / sum(n))
  ggplot(d, aes(x = Congestion_Level, y = n, fill = Congestion_Level)) +
    geom_col(alpha = 0.9) +
    geom_text(aes(label = paste0(n, " (", round(100 * pct, 1), "%)")),
              vjust = -0.4, colour = "#1f3b57") +
    scale_fill_manual(values = congestion_colors) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(title = "Distribution of Congestion Levels",
         x = "Congestion level", y = "Number of records") +
    theme_traffic() + theme(legend.position = "none")
}

# ---- 9. Road type vs congestion -------------------------------------------------------------------
plot_road_type_vs_congestion <- function(df) {
  ggplot(df, aes(x = Road_Type, fill = Congestion_Level)) +
    geom_bar(position = "dodge", alpha = 0.9) +
    scale_fill_manual(values = congestion_colors) +
    labs(title = "Road Type vs Congestion",
         x = "Road type", y = "Number of records", fill = "Congestion") +
    theme_traffic()
}

# ---- 10. Hourly congestion trend ---------------------------------------------------------------------
plot_hourly_congestion_trend <- function(df) {
  d <- df %>%
    count(Hour, Congestion_Level) %>%
    group_by(Hour) %>%
    mutate(Share = n / sum(n)) %>%
    ungroup()
  ggplot(d, aes(x = Hour, y = Share, colour = Congestion_Level)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 1.8) +
    scale_colour_manual(values = congestion_colors) +
    scale_x_continuous(breaks = 0:23) +
    scale_y_continuous(labels = scales::percent) +
    labs(title = "Hourly Congestion Trend",
         subtitle = "Share of Low / Medium / High records in each hour",
         x = "Hour of day (0-23)", y = "Share of records", colour = "Congestion") +
    theme_traffic()
}

# Returns all ten EDA graphs in one named list
all_eda_plots <- function(df) {
  list(
    "01_vehicles_by_hour"        = plot_vehicles_by_hour(df),
    "02_traffic_by_day"          = plot_traffic_by_day(df),
    "03_speed_by_congestion"     = plot_speed_by_congestion(df),
    "04_vehicles_vs_speed"       = plot_vehicles_vs_speed(df),
    "05_rainfall_vs_congestion"  = plot_rainfall_vs_congestion(df),
    "06_accident_vs_congestion"  = plot_accident_vs_congestion(df),
    "07_holiday_vs_congestion"   = plot_holiday_vs_congestion(df),
    "08_congestion_distribution" = plot_congestion_distribution(df),
    "09_road_type_vs_congestion" = plot_road_type_vs_congestion(df),
    "10_hourly_congestion_trend" = plot_hourly_congestion_trend(df)
  )
}

# ---- Model plots ------------------------------------------------------------------------------------------
# Confusion matrix as a coloured heat-map. 'cm' is the object from caret::confusionMatrix()
plot_confusion_matrix <- function(cm, title = "Confusion Matrix") {
  d <- as.data.frame(cm$table)              # columns: Prediction, Reference, Freq
  d$Prediction <- factor(d$Prediction, levels = rev(levels(d$Prediction)))  # Low at top
  ggplot(d, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(colour = "white", linewidth = 1) +
    geom_text(aes(label = Freq, colour = Freq > max(Freq) / 2),
              size = 6, fontface = "bold") +
    scale_fill_gradient(low = "#eaf3fb", high = "#1f6fb2") +
    scale_colour_manual(values = c("TRUE" = "white", "FALSE" = "#1f3b57"), guide = "none") +
    labs(title = title, x = "Actual level", y = "Predicted level") +
    theme_traffic() + theme(legend.position = "none", panel.grid = element_blank())
}

# Bar chart comparing the four metrics of the three models
plot_model_comparison <- function(metrics) {
  d <- metrics %>%
    pivot_longer(cols = c(Accuracy, Precision, Recall, F1_Score),
                 names_to = "Metric", values_to = "Value")
  ggplot(d, aes(x = Model, y = Value, fill = Metric)) +
    geom_col(position = "dodge", alpha = 0.9) +
    geom_text(aes(label = sprintf("%.2f", Value)),
              position = position_dodge(width = 0.9), vjust = -0.4, size = 3) +
    scale_fill_manual(values = c("Accuracy" = "#1f6fb2", "Precision" = "#3fa7d6",
                                 "Recall" = "#59c3a5", "F1_Score" = "#f0a202")) +
    scale_y_continuous(limits = c(0, 1.08), expand = c(0, 0)) +
    labs(title = "Model Comparison on Test Data", x = NULL, y = "Score") +
    theme_traffic()
}

# Random-forest feature importance (which inputs matter most)
plot_feature_importance <- function(importance_df) {
  ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_col(fill = "#1f6fb2", alpha = 0.9) +
    coord_flip() +
    labs(title = "Random Forest - Feature Importance",
         subtitle = "Higher value = the feature helps more in separating Low / Medium / High",
         x = NULL, y = "Mean decrease in Gini") +
    theme_traffic()
}

# Probability bars shown after a prediction in the Shiny app
plot_prediction_probabilities <- function(probs) {
  d <- tibble::tibble(Level = factor(names(probs), levels = CONG_LEVELS),
                      Probability = as.numeric(probs))
  ggplot(d, aes(x = Level, y = Probability, fill = Level)) +
    geom_col(alpha = 0.9, width = 0.6) +
    geom_text(aes(label = paste0(round(100 * Probability, 1), "%")),
              vjust = -0.4, colour = "#1f3b57") +
    scale_fill_manual(values = congestion_colors) +
    scale_y_continuous(limits = c(0, 1.1), labels = scales::percent, expand = c(0, 0)) +
    labs(title = "Model Confidence", x = NULL, y = "Probability") +
    theme_traffic() + theme(legend.position = "none")
}


# #############################################################################
# PART: R/model_training.R
# #############################################################################
# =============================================================================
# R/model_training.R
# Train, evaluate and use the three classification models:
#   1. Logistic Regression (multinomial - because there are 3 classes)
#   2. Decision Tree
#   3. Random Forest
# Everything is stored in ONE "bundle" list that is saved to
# models/trained_model.rds and loaded again by the Shiny app.
# =============================================================================

library(caret)
library(randomForest)
library(rpart)
library(nnet)
library(dplyr)
library(tibble)

# Columns used as model inputs. Date and Time are NOT used: Hour and Day_of_Week
# already carry that information, and a raw date would only let a model memorise.
FEATURE_COLS <- c("Day_of_Week", "Hour", "Vehicle_Count", "Average_Speed",
                  "Rainfall_mm", "Temperature_C", "Accident", "Road_Type",
                  "Holiday", "Traffic_Signal_Count", "Road_Width")

# ---- 1. Train / test split (80% / 20%) ------------------------------------------
split_data <- function(df, seed = 123, p = 0.8) {
  set.seed(seed)                                        # reproducible split
  idx <- as.vector(caret::createDataPartition(df$Congestion_Level, p = p, list = FALSE))
  df  <- as.data.frame(df)                              # plain data frame works best with caret
  list(train = df[idx, ], test = df[-idx, ])            # stratified: same class mix
}

# ---- 2. Train the three models ----------------------------------------------------
train_models <- function(train_df, seed = 123) {
  # method = "none": train once with fixed settings (fast and easy to explain)
  ctrl <- caret::trainControl(method = "none", classProbs = TRUE)
  x <- train_df[, FEATURE_COLS]
  y <- train_df$Congestion_Level

  # Logistic regression needs numbers on a similar scale, so centre + scale them.
  # caret stores this scaling INSIDE the model and re-applies it automatically
  # when predicting -> the same preprocessing is used for new data.
  set.seed(seed)
  logistic <- caret::train(
    Congestion_Level ~ .,
    data       = train_df[, c(FEATURE_COLS, "Congestion_Level")],
    method     = "multinom",
    trControl  = ctrl,
    tuneGrid   = data.frame(decay = 0),
    preProcess = c("center", "scale"),
    trace      = FALSE, maxit = 500, MaxNWts = 5000
  )

  set.seed(seed)
  tree <- caret::train(
    x = x, y = y, method = "rpart",
    trControl = ctrl,
    tuneGrid  = data.frame(cp = 0.01)
  )

  set.seed(seed)
  forest <- caret::train(
    x = x, y = y, method = "rf",
    trControl = ctrl,
    tuneGrid  = data.frame(mtry = 3),
    ntree = 200, importance = TRUE
  )

  list("Logistic Regression" = logistic,
       "Decision Tree"       = tree,
       "Random Forest"       = forest)
}

# ---- 3. Evaluate one model on the TEST data ------------------------------------------
# Precision, recall and F1 are calculated for each class (Low / Medium / High)
# and then AVERAGED (macro average) so that all three classes count equally.
evaluate_model <- function(model, test_df, model_name) {
  pred <- predict(model, newdata = test_df[, FEATURE_COLS])
  pred <- factor(pred, levels = CONG_LEVELS)
  cm   <- caret::confusionMatrix(data = pred, reference = test_df$Congestion_Level)

  by_class <- as.data.frame(cm$byClass)
  by_class[is.na(by_class)] <- 0     # a class that is never predicted gets score 0

  list(
    confusion = cm,
    summary = tibble(
      Model     = model_name,
      Accuracy  = round(unname(cm$overall["Accuracy"]), 4),
      Precision = round(mean(by_class$Precision), 4),
      Recall    = round(mean(by_class$Recall), 4),
      F1_Score  = round(mean(by_class$F1), 4)
    ),
    per_class = tibble(
      Model     = model_name,
      Class     = CONG_LEVELS,
      Precision = round(by_class$Precision, 4),
      Recall    = round(by_class$Recall, 4),
      F1_Score  = round(by_class$F1, 4)
    )
  )
}

# ---- 4. Train + evaluate everything and pack the results -------------------------------
train_all_models <- function(clean_df, seed = 123) {
  parts  <- split_data(clean_df, seed = seed)
  models <- train_models(parts$train, seed = seed)

  results   <- lapply(names(models), function(nm) evaluate_model(models[[nm]], parts$test, nm))
  names(results) <- names(models)

  metrics   <- dplyr::bind_rows(lapply(results, `[[`, "summary"))
  per_class <- dplyr::bind_rows(lapply(results, `[[`, "per_class"))
  confusion <- lapply(results, `[[`, "confusion")

  # Best model = highest F1-score on the test data (accuracy breaks ties)
  best_model <- metrics %>% arrange(desc(F1_Score), desc(Accuracy)) %>% slice(1) %>% pull(Model)

  # Random-forest feature importance (Mean Decrease in Gini)
  imp <- randomForest::importance(models[["Random Forest"]]$finalModel, type = 2)
  importance <- tibble(Feature = rownames(imp), Importance = as.numeric(imp[, 1])) %>%
    arrange(desc(Importance))

  list(
    models     = models,
    metrics    = as.data.frame(metrics),
    per_class  = as.data.frame(per_class),
    confusion  = confusion,
    best_model = best_model,
    importance = importance,
    train_rows = nrow(parts$train),
    test_rows  = nrow(parts$test),
    seed       = seed,
    feature_cols = FEATURE_COLS,
    created_at = format(Sys.time(), "%Y-%m-%d %H:%M")
  )
}

# ---- 5. Save / load the bundle ---------------------------------------------------------------
save_bundle <- function(bundle, path = "models/trained_model.rds") {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(bundle, path)
  invisible(path)
}

is_valid_bundle <- function(b) {
  is.list(b) && all(c("models", "metrics", "confusion", "best_model") %in% names(b)) &&
    all(c("Logistic Regression", "Decision Tree", "Random Forest") %in% names(b$models))
}

# Load the saved model. If the file is missing or damaged, train a new one.
# Returns list(bundle = ..., note = message or NULL)
load_or_train_model <- function(clean_df, path = "models/trained_model.rds") {
  if (file.exists(path)) {
    bundle <- tryCatch(readRDS(path), error = function(e) NULL)
    if (is_valid_bundle(bundle)) return(list(bundle = bundle, note = NULL))
    note <- "The saved model file could not be read, so a new model was trained."
  } else {
    note <- "No saved model was found, so a new model was trained (first run only)."
  }
  bundle <- train_all_models(clean_df)
  try(save_bundle(bundle, path), silent = TRUE)
  list(bundle = bundle, note = note)
}

# ---- 6. Predict for ONE new observation ---------------------------------------------------------
# 'obs' comes from build_new_observation() in data_preprocessing.R, so it has the
# same columns, types and factor levels as the training data.
predict_congestion <- function(bundle, obs, model_name = "Best model (auto)") {
  name  <- if (model_name %in% names(bundle$models)) model_name else bundle$best_model
  model <- bundle$models[[name]]

  label <- as.character(predict(model, newdata = obs[, FEATURE_COLS]))
  probs <- predict(model, newdata = obs[, FEATURE_COLS], type = "prob")
  probs <- unlist(probs[1, CONG_LEVELS])

  list(label = label, probs = probs, model = name)
}

# ---- 7. Plain-English explanation of a prediction --------------------------------------------------
explain_prediction <- function(obs, label, probs, train_df, model_name) {
  msgs <- character(0)
  hr   <- obs$Hour

  # Time of day
  msgs <- c(msgs, if (hr %in% 7:10 || hr %in% 16:20) {
    paste0("Hour ", hr, " is a rush hour, when roads are normally busiest.")
  } else if (hr %in% 0:5) {
    paste0("Hour ", hr, " is late night, when traffic is normally light.")
  } else {
    paste0("Hour ", hr, " is outside the main rush hours.")
  })

  # Vehicle count compared with history
  pct_v <- round(100 * mean(train_df$Vehicle_Count <= obs$Vehicle_Count))
  msgs <- c(msgs, paste0("A count of ", obs$Vehicle_Count, " vehicles is higher than about ",
                         pct_v, "% of the historical records."))

  # Speed compared with history
  pct_s <- round(100 * mean(train_df$Average_Speed >= obs$Average_Speed))
  msgs <- c(msgs, paste0("An average speed of ", obs$Average_Speed,
                         " km/h is slower than about ", pct_s, "% of the historical records."))

  if (obs$Rainfall_mm > 0) {
    msgs <- c(msgs, paste0("Rainfall of ", obs$Rainfall_mm,
                           " mm usually reduces speed and increases congestion."))
  }
  if (obs$Accident == "Yes") {
    msgs <- c(msgs, "A reported accident blocks lanes and usually raises congestion.")
  }
  if (obs$Holiday == "Yes" || obs$Day_of_Week %in% c("Saturday", "Sunday")) {
    msgs <- c(msgs, "Holidays and weekends usually have fewer commuter vehicles.")
  }

  top <- which.max(probs)
  msgs <- c(msgs, paste0("The ", model_name, " model is ",
                         round(100 * probs[[top]], 1), "% confident in '",
                         names(probs)[top], "' congestion, based on patterns learned from ",
                         format(nrow(train_df), big.mark = ","), " historical records."))
  msgs
}


# #############################################################################
# PART: app.R (Shiny dashboard)
# #############################################################################
# =============================================================================
# app.R  -  Traffic Monitor: Congestion Analysis and Prediction Dashboard
# Traffic Congestion Prediction and Analysis Using R and Machine Learning
#
# HOW TO RUN: open this file in RStudio and click the "Run App" button
#             (or run  shiny::runApp()  from the project folder).
# =============================================================================

library(shiny)
library(shinydashboard)
library(DT)
library(plotly)
library(tidyverse)       # dplyr, ggplot2, readr, tibble, stringr, tidyr

# ---- Load the project modules (see the R/ folder) ------------------------------

# =============================================================================
# 1. START-UP: load data + model. Problems are collected as friendly messages
#    instead of crashing the app.
# =============================================================================
init_app <- function() {
  errors <- character(0)
  notes  <- character(0)
  state  <- list(data = NULL, prep = NULL, bundle = NULL, source = NA_character_)

  # -- dataset --
  ds <- tryCatch(load_or_create_dataset(), error = function(e) {
    errors <<- c(errors, paste("The dataset could not be loaded:", conditionMessage(e)))
    NULL
  })

  if (!is.null(ds)) {
    if (isTRUE(ds$generated)) {
      notes <- c(notes, "The dataset file was missing, so a synthetic dataset was generated and saved to data/traffic_data.csv.")
    }

    # -- preprocessing (cleaning) --
    prep <- tryCatch(preprocess_traffic_data(ds$data, verbose = FALSE), error = function(e) {
      errors <<- c(errors, paste("The dataset could not be cleaned:", conditionMessage(e)))
      NULL
    })

    if (!is.null(prep)) {
      state$prep   <- prep
      state$data   <- prep$data
      state$source <- ds$source

      # -- model (load from models/trained_model.rds or train if missing) --
      m <- tryCatch(load_or_train_model(prep$data), error = function(e) {
        errors <<- c(errors, paste("The prediction model could not be loaded or trained:", conditionMessage(e)))
        NULL
      })
      if (!is.null(m)) {
        state$bundle <- m$bundle
        if (!is.null(m$note)) notes <- c(notes, m$note)
      }
    }
  }

  state$errors <- errors
  state$notes  <- notes
  state
}

app_state <- init_app()

# Table that repeats the user's inputs after a prediction
inputs_table <- function(obs) {
  tibble(
    Field = c("Day of week", "Hour", "Vehicle count", "Average speed (km/h)",
              "Rainfall (mm)", "Temperature (deg C)", "Accident", "Road type",
              "Holiday", "Traffic signal count", "Road width (m)"),
    Value = c(as.character(obs$Day_of_Week), as.character(obs$Hour),
              as.character(obs$Vehicle_Count), as.character(obs$Average_Speed),
              as.character(obs$Rainfall_mm), as.character(obs$Temperature_C),
              as.character(obs$Accident), as.character(obs$Road_Type),
              as.character(obs$Holiday), as.character(obs$Traffic_Signal_Count),
              as.character(obs$Road_Width))
  )
}

# =============================================================================
# 2. LIGHT THEME (custom CSS on top of shinydashboard)
# =============================================================================
custom_css <- "
.content-wrapper, .right-side { background-color: #f4f7fb; }
.skin-blue .main-header .logo { background-color: #ffffff; color: #1f3b57; font-weight: 700;
                                border-bottom: 1px solid #e3e9f0; }
.skin-blue .main-header .logo:hover { background-color: #f4f7fb; }
.skin-blue .main-header .navbar { background-color: #ffffff; border-bottom: 1px solid #e3e9f0; }
.skin-blue .main-header .navbar .sidebar-toggle { color: #1f3b57; }
.skin-blue .main-header .navbar .sidebar-toggle:hover { background-color: #eef3f9; }
.skin-blue .main-sidebar { background-color: #ffffff; border-right: 1px solid #e3e9f0; }
.skin-blue .sidebar-menu > li > a { color: #44546a; border-left: 3px solid transparent; }
.skin-blue .sidebar-menu > li:hover > a,
.skin-blue .sidebar-menu > li.active > a { color: #1f6fb2; background: #eaf3fb; border-left-color: #1f6fb2; }
.box { border-top: 3px solid #1f6fb2; border-radius: 6px; box-shadow: 0 1px 4px rgba(31,59,87,0.08); }
.box-header .box-title { color: #1f3b57; font-weight: 600; }
.small-box { border-radius: 6px; }
.page-title { color: #1f3b57; font-weight: 700; margin-top: 0; }
.muted { color: #5f6b7a; }
.result-box { font-size: 24px; font-weight: 700; text-align: center; padding: 18px 10px;
              border-radius: 8px; color: #ffffff; margin-bottom: 8px; }
.result-Low    { background: #2e9e5b; }
.result-Medium { background: #f0a202; }
.result-High   { background: #d64545; }
.model-note { text-align: center; color: #5f6b7a; }
.placeholder-box { padding: 30px 15px; text-align: center; color: #5f6b7a;
                   background: #f4f7fb; border: 1px dashed #b9c6d6; border-radius: 8px; }
.pipeline { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.pipeline .step { background: #eaf3fb; border: 1px solid #b9d4ec; color: #1f3b57;
                  padding: 8px 12px; border-radius: 6px; font-weight: 600; font-size: 13px; }
.pipeline .arrow { color: #1f6fb2; font-weight: 700; }
.btn-primary { background-color: #1f6fb2; border-color: #1a5e98; }
"

# =============================================================================
# 3. USER INTERFACE
# =============================================================================
ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "Traffic Monitor", titleWidth = 250),

  # ---- Sidebar menu ----
  dashboardSidebar(
    width = 250,
    sidebarMenu(
      id = "tabs",
      menuItem("Home",               tabName = "home",     icon = icon("home")),
      menuItem("Dataset",            tabName = "dataset",  icon = icon("database")),
      menuItem("Traffic Analysis",   tabName = "analysis", icon = icon("chart-line")),
      menuItem("Visualization",      tabName = "viz",      icon = icon("chart-area")),
      menuItem("Model Performance",  tabName = "models",   icon = icon("bullseye")),
      menuItem("Traffic Prediction", tabName = "predict",  icon = icon("car")),
      menuItem("About",              tabName = "about",    icon = icon("info-circle"))
    )
  ),

  dashboardBody(
    tags$head(tags$style(HTML(custom_css))),
    uiOutput("startup_messages"),

    tabItems(
      # ------------------------------------------------------------------ HOME
      tabItem(tabName = "home",
        h2("Traffic Congestion Prediction and Analysis Using R and Machine Learning", class = "page-title"),
        fluidRow(
          valueBoxOutput("vb_records", width = 3),
          valueBoxOutput("vb_vars",    width = 3),
          valueBoxOutput("vb_high",    width = 3),
          valueBoxOutput("vb_best",    width = 3)
        ),
        fluidRow(
          box(title = "Problem Statement", width = 6,
              p("Cities experience traffic congestion during peak hours, weekends, bad weather,
                 accidents and other unusual situations. Congestion causes travel delays, wasted
                 fuel and higher pollution."),
              p("This system studies historical traffic data, finds the factors that contribute to
                 congestion and predicts the expected congestion level for new traffic conditions.")),
          box(title = "Project Objective", width = 6,
              tags$ul(
                tags$li("Analyse historical traffic data with R."),
                tags$li("Find what causes congestion (time, weather, accidents, road type)."),
                tags$li("Predict congestion as Low, Medium or High using machine learning."),
                tags$li("Compare Logistic Regression, Decision Tree and Random Forest."),
                tags$li("Provide an easy R Shiny dashboard for prediction.")
              ))
        ),
        fluidRow(
          box(title = "How the System Works", width = 12,
              div(class = "pipeline",
                  div(class = "step", "Traffic Dataset"), div(class = "arrow", "\u2192"),
                  div(class = "step", "Data Preprocessing"), div(class = "arrow", "\u2192"),
                  div(class = "step", "Exploratory Analysis"), div(class = "arrow", "\u2192"),
                  div(class = "step", "Feature Selection"), div(class = "arrow", "\u2192"),
                  div(class = "step", "Train ML Models"), div(class = "arrow", "\u2192"),
                  div(class = "step", "Model Evaluation"), div(class = "arrow", "\u2192"),
                  div(class = "step", "Best Model"), div(class = "arrow", "\u2192"),
                  div(class = "step", "Shiny Dashboard"), div(class = "arrow", "\u2192"),
                  div(class = "step", "Congestion Prediction")),
              br(),
              p(class = "muted",
                "The system cleans the data, learns patterns with three machine-learning models,
                 measures how good each model is on unseen test data, and uses the best-performing
                 model in the Traffic Prediction tab."))
        )
      ),

      # --------------------------------------------------------------- DATASET
      tabItem(tabName = "dataset",
        h2("Dataset", class = "page-title"),
        fluidRow(
          valueBoxOutput("ds_records", width = 4),
          valueBoxOutput("ds_vars",    width = 4),
          valueBoxOutput("ds_missing", width = 4)
        ),
        fluidRow(box(title = "Traffic Dataset (after cleaning)", width = 12, DTOutput("tbl_data"))),
        fluidRow(
          box(title = "Summary Statistics (numeric variables)", width = 7, DTOutput("tbl_summary")),
          box(title = "Preprocessing: Before vs After", width = 5, tableOutput("tbl_compare"))
        ),
        fluidRow(
          box(title = "Outlier Check (IQR rule)", width = 6, tableOutput("tbl_outliers")),
          box(title = "Dataset Structure", width = 6, verbatimTextOutput("txt_str"))
        )
      ),

      # ------------------------------------------------------ TRAFFIC ANALYSIS
      tabItem(tabName = "analysis",
        h2("Congestion Analysis", class = "page-title"),
        fluidRow(
          valueBoxOutput("st_mean",   width = 4),
          valueBoxOutput("st_median", width = 4),
          valueBoxOutput("st_sd",     width = 4)
        ),
        fluidRow(
          valueBoxOutput("st_min",   width = 4),
          valueBoxOutput("st_max",   width = 4),
          valueBoxOutput("st_speed", width = 4)
        ),
        fluidRow(
          box(title = "What do the statistics mean?", width = 7, uiOutput("interp")),
          box(title = "Average Conditions per Congestion Level", width = 5, tableOutput("tbl_profile"))
        ),
        fluidRow(
          box(title = "Vehicle Count by Hour", width = 6, plotOutput("pl_hour", height = 320)),
          box(title = "Hourly Congestion Trend", width = 6, plotOutput("pl_trend", height = 320))
        ),
        fluidRow(
          box(title = "Traffic by Day of Week", width = 6, plotOutput("pl_day", height = 320)),
          box(title = "Speed Analysis", width = 6, plotOutput("pl_speed", height = 320))
        ),
        fluidRow(
          box(title = "Congestion Distribution", width = 6, plotOutput("pl_dist", height = 320)),
          box(title = "Road Type vs Congestion", width = 6, plotOutput("pl_road", height = 320))
        )
      ),

      # --------------------------------------------------------- VISUALIZATION
      tabItem(tabName = "viz",
        h2("Interactive Visualization", class = "page-title"),
        p(class = "muted", "Hover over the graphs for details. Use the toolbar to zoom or download."),
        tabBox(
          width = 12,
          tabPanel("Traffic by Hour",        plotlyOutput("pv_hour",     height = 420)),
          tabPanel("Traffic by Day",         plotlyOutput("pv_day",      height = 420)),
          tabPanel("Vehicles vs Speed",      plotlyOutput("pv_speed",    height = 420)),
          tabPanel("Weather vs Traffic",     plotlyOutput("pv_rain",     height = 420)),
          tabPanel("Accident vs Congestion", plotlyOutput("pv_accident", height = 420)),
          tabPanel("Holiday vs Congestion",  plotlyOutput("pv_holiday",  height = 420))
        )
      ),

      # ----------------------------------------------------- MODEL PERFORMANCE
      tabItem(tabName = "models",
        h2("Model Performance", class = "page-title"),
        p(class = "muted",
          "All numbers below were measured on the 20% test data that the models never saw during training."),
        fluidRow(
          valueBoxOutput("m_best",  width = 3),
          valueBoxOutput("m_acc",   width = 3),
          valueBoxOutput("m_f1",    width = 3),
          valueBoxOutput("m_split", width = 3)
        ),
        fluidRow(
          box(title = "Model Comparison Table", width = 7, DTOutput("tbl_metrics"),
              p(class = "muted", style = "margin-top:8px;",
                "Precision, Recall and F1 are macro-averages over Low, Medium and High.")),
          box(title = "Comparison Chart", width = 5, plotOutput("pl_model_cmp", height = 300))
        ),
        fluidRow(
          box(title = "Confusion Matrix - Logistic Regression", width = 4, plotOutput("cm_lr", height = 290)),
          box(title = "Confusion Matrix - Decision Tree",       width = 4, plotOutput("cm_dt", height = 290)),
          box(title = "Confusion Matrix - Random Forest",       width = 4, plotOutput("cm_rf", height = 290))
        ),
        fluidRow(
          box(title = "Per-Class Metrics", width = 7, DTOutput("tbl_perclass")),
          box(title = "Random Forest Feature Importance", width = 5, plotOutput("pl_imp", height = 300))
        )
      ),

      # -------------------------------------------------------------- PREDICTION
      tabItem(tabName = "predict",
        h2("Traffic Prediction", class = "page-title"),
        fluidRow(
          box(title = "Enter Traffic Conditions", width = 7,
            fluidRow(
              column(6,
                selectInput("day", "Day of week", choices = DAY_LEVELS, selected = "Monday"),
                numericInput("hour", "Hour of day (0-23)", value = 8, min = 0, max = 23, step = 1),
                numericInput("vehicles", "Vehicle count (per hour)", value = 1450, min = 0, max = 5000, step = 10),
                numericInput("speed", "Average speed (km/h)", value = 18, min = 1, max = 130, step = 1),
                numericInput("rain", "Rainfall (mm)", value = 4, min = 0, max = 300, step = 0.5),
                numericInput("temp", "Temperature (deg C)", value = 28, min = -10, max = 55, step = 0.5)
              ),
              column(6,
                selectInput("accident", "Accident reported?", choices = YESNO_LEVELS, selected = "Yes"),
                selectInput("road", "Road type", choices = ROAD_LEVELS, selected = "Highway"),
                selectInput("holiday", "Holiday?", choices = YESNO_LEVELS, selected = "No"),
                numericInput("signals", "Traffic signal count", value = 2, min = 0, max = 30, step = 1),
                numericInput("width", "Road width (m)", value = 16, min = 2, max = 40, step = 0.5),
                selectInput("model_choice", "Model to use",
                            choices = c("Best model (auto)", "Logistic Regression",
                                        "Decision Tree", "Random Forest"),
                            selected = "Best model (auto)")
              )
            ),
            actionButton("predict_btn", "Predict Traffic", icon = icon("car"),
                         class = "btn-primary btn-lg"),
            actionButton("reset_btn", "Reset", icon = icon("undo"), class = "btn-lg")
          ),
          box(title = "Prediction Result", width = 5, uiOutput("prediction_ui"))
        )
      ),

      # ------------------------------------------------------------------- ABOUT
      tabItem(tabName = "about",
        h2("About", class = "page-title"),
        fluidRow(
          box(title = "Project Information", width = 6,
              p(strong("Title: "), "Traffic Congestion Prediction and Analysis Using R and Machine Learning"),
              p(strong("Type: "), "College mini project (classification + dashboard)"),
              p(strong("Goal: "), "Predict whether traffic congestion will be Low, Medium or High.")),
          box(title = "Technologies Used", width = 6,
              tags$ul(
                tags$li("R Programming and RStudio"),
                tags$li("tidyverse (dplyr, readr, tidyr, stringr) for data handling"),
                tags$li("ggplot2 and plotly for graphs"),
                tags$li("caret, nnet, rpart, randomForest for machine learning"),
                tags$li("R Shiny and shinydashboard for the web dashboard")
              ))
        ),
        fluidRow(
          box(title = "Dataset Information", width = 12,
              p("The dataset is ", strong("synthetic"),
                " (computer-generated). It was created in R with set.seed(123) using realistic traffic rules:
                 rush hours in the morning and evening, slower speed with rain and accidents,
                 quieter weekends and holidays, and wider roads carrying more vehicles. It contains about
                 1,500 records. A few missing values, invalid values and duplicates were added on purpose
                 so that the data-cleaning step has real work to do."),
              p("To use real data, replace data/traffic_data.csv with a file that has the same column names,
                delete models/trained_model.rds and restart the app."))
        ),
        fluidRow(
          box(title = "Limitations", width = 6,
              tags$ul(
                tags$li("The data is synthetic, so accuracy values show the method works, not how it would perform on a real city."),
                tags$li("The congestion label is created from a rule plus random noise, so it is easier to learn than real labels."),
                tags$li("Only one simple train/test split is used (no cross-validation)."),
                tags$li("No live traffic feed, maps, events or road-works information.")
              )),
          box(title = "Future Improvements", width = 6,
              tags$ul(
                tags$li("Train on real sensor or GPS data from a city traffic department."),
                tags$li("Add cross-validation and hyper-parameter tuning."),
                tags$li("Add live weather and traffic APIs."),
                tags$li("Show congestion on a map and forecast the next few hours."),
                tags$li("Try other models such as gradient boosting.")
              ))
        )
      )
    )
  )
)

# =============================================================================
# 4. SERVER
# =============================================================================
server <- function(input, output, session) {

  df     <- app_state$data
  prep   <- app_state$prep
  bundle <- app_state$bundle
  stats  <- if (!is.null(df)) compute_statistics(df) else NULL

  # Small helper to build a value box quickly
  vb <- function(value, subtitle, icon_name, color) {
    valueBox(value, subtitle, icon = icon(icon_name), color = color)
  }

  # ---- Start-up messages (problems and notes) ----
  output$startup_messages <- renderUI({
    tagList(
      lapply(app_state$errors, function(msg) {
        div(class = "alert alert-danger", icon("exclamation-triangle"), " ", msg)
      }),
      lapply(app_state$notes, function(msg) {
        div(class = "alert alert-info", icon("info-circle"), " ", msg)
      })
    )
  })

  # ---------------------------------------------------------------- HOME
  output$vb_records <- renderValueBox({
    req(df); vb(format(nrow(df), big.mark = ","), "Records analysed", "database", "blue")
  })
  output$vb_vars <- renderValueBox({
    req(df); vb(ncol(df), "Variables", "list", "light-blue")
  })
  output$vb_high <- renderValueBox({
    req(df)
    vb(paste0(round(100 * mean(df$Congestion_Level == "High"), 1), "%"),
       "Records with High congestion", "exclamation-triangle", "red")
  })
  output$vb_best <- renderValueBox({
    req(bundle); vb(bundle$best_model, "Best model (by F1-score)", "trophy", "green")
  })

  # ------------------------------------------------------------- DATASET
  output$ds_records <- renderValueBox({
    req(df); vb(format(nrow(df), big.mark = ","), "Number of records", "table", "blue")
  })
  output$ds_vars <- renderValueBox({
    req(df); vb(ncol(df), "Number of variables", "list", "light-blue")
  })
  output$ds_missing <- renderValueBox({
    req(prep); vb(sum(prep$missing_before), "Missing values found and fixed", "eraser", "yellow")
  })

  output$tbl_data <- DT::renderDT({
    req(df)
    d <- df %>% mutate(Date = as.character(Date))
    DT::datatable(d, rownames = FALSE, filter = "top",
                  options = list(pageLength = 10, scrollX = TRUE))
  })
  output$tbl_summary <- DT::renderDT({
    req(df)
    DT::datatable(numeric_summary(df), rownames = FALSE,
                  options = list(dom = "t", pageLength = 10, scrollX = TRUE))
  })
  output$tbl_compare <- renderTable({ req(prep); as.data.frame(prep$comparison) }, na = "-")
  output$tbl_outliers <- renderTable({ req(prep); as.data.frame(prep$outliers) })
  output$txt_str <- renderText({
    req(df); paste(capture.output(str(df)), collapse = "\n")
  })

  # ----------------------------------------------------- TRAFFIC ANALYSIS
  output$st_mean   <- renderValueBox({ req(stats); vb(round(stats$mean_vc, 1), "Mean vehicle count", "car", "blue") })
  output$st_median <- renderValueBox({ req(stats); vb(round(stats$median_vc, 1), "Median vehicle count", "car", "light-blue") })
  output$st_sd     <- renderValueBox({ req(stats); vb(round(stats$sd_vc, 1), "Standard deviation", "chart-bar", "purple") })
  output$st_min    <- renderValueBox({ req(stats); vb(stats$min_vc, "Minimum vehicle count", "arrow-down", "green") })
  output$st_max    <- renderValueBox({ req(stats); vb(stats$max_vc, "Maximum vehicle count", "arrow-up", "red") })
  output$st_speed  <- renderValueBox({ req(stats); vb(paste(round(stats$mean_speed, 1), "km/h"), "Average speed", "tachometer-alt", "yellow") })

  output$interp <- renderUI({
    req(stats)
    tags$ul(lapply(interpret_statistics(stats), tags$li))
  })
  output$tbl_profile <- renderTable({ req(df); as.data.frame(congestion_profile(df)) })

  output$pl_hour  <- renderPlot({ req(df); plot_vehicles_by_hour(df) })
  output$pl_trend <- renderPlot({ req(df); plot_hourly_congestion_trend(df) })
  output$pl_day   <- renderPlot({ req(df); plot_traffic_by_day(df) })
  output$pl_speed <- renderPlot({ req(df); plot_speed_by_congestion(df) })
  output$pl_dist  <- renderPlot({ req(df); plot_congestion_distribution(df) })
  output$pl_road  <- renderPlot({ req(df); plot_road_type_vs_congestion(df) })

  # ------------------------------------------------------ VISUALIZATION
  output$pv_hour     <- renderPlotly({ req(df); ggplotly(plot_vehicles_by_hour(df)) })
  output$pv_day      <- renderPlotly({ req(df); ggplotly(plot_traffic_by_day(df)) })
  output$pv_speed    <- renderPlotly({ req(df); ggplotly(plot_vehicles_vs_speed(df)) })
  output$pv_rain     <- renderPlotly({ req(df); ggplotly(plot_rainfall_vs_congestion(df)) })
  output$pv_accident <- renderPlotly({ req(df); ggplotly(plot_accident_vs_congestion(df)) })
  output$pv_holiday  <- renderPlotly({ req(df); ggplotly(plot_holiday_vs_congestion(df)) })

  # ------------------------------------------------- MODEL PERFORMANCE
  best_row <- reactive({
    req(bundle)
    bundle$metrics[bundle$metrics$Model == bundle$best_model, ]
  })
  output$m_best  <- renderValueBox({ req(bundle); vb(bundle$best_model, "Best model", "trophy", "green") })
  output$m_acc   <- renderValueBox({ vb(sprintf("%.2f%%", 100 * best_row()$Accuracy), "Accuracy of best model", "bullseye", "blue") })
  output$m_f1    <- renderValueBox({ vb(sprintf("%.4f", best_row()$F1_Score), "F1-score of best model", "balance-scale", "light-blue") })
  output$m_split <- renderValueBox({
    req(bundle)
    vb(paste(bundle$train_rows, "/", bundle$test_rows), "Train rows / Test rows (80:20)", "random", "purple")
  })

  output$tbl_metrics <- DT::renderDT({
    req(bundle)
    m <- bundle$metrics
    names(m) <- c("Model", "Accuracy", "Precision", "Recall", "F1 Score")
    DT::datatable(m, rownames = FALSE, options = list(dom = "t", ordering = FALSE)) %>%
      DT::formatRound(2:5, 4) %>%
      DT::formatStyle("Model", target = "row",
                      backgroundColor = DT::styleEqual(bundle$best_model, "#e6f4ea"))
  })
  output$pl_model_cmp <- renderPlot({ req(bundle); plot_model_comparison(bundle$metrics) })
  output$cm_lr <- renderPlot({ req(bundle); plot_confusion_matrix(bundle$confusion[["Logistic Regression"]], "Logistic Regression") })
  output$cm_dt <- renderPlot({ req(bundle); plot_confusion_matrix(bundle$confusion[["Decision Tree"]], "Decision Tree") })
  output$cm_rf <- renderPlot({ req(bundle); plot_confusion_matrix(bundle$confusion[["Random Forest"]], "Random Forest") })
  output$tbl_perclass <- DT::renderDT({
    req(bundle)
    m <- bundle$per_class
    names(m) <- c("Model", "Class", "Precision", "Recall", "F1 Score")
    DT::datatable(m, rownames = FALSE, options = list(dom = "t", pageLength = 9)) %>%
      DT::formatRound(3:5, 4)
  })
  output$pl_imp <- renderPlot({ req(bundle); plot_feature_importance(head(bundle$importance, 10)) })

  # ------------------------------------------------------- PREDICTION
  pred_result <- reactiveVal(NULL)

  observeEvent(input$predict_btn, {
    res <- tryCatch({
      if (is.null(bundle)) {
        stop("The prediction model is not available. Please read the messages at the top of the page.")
      }
      if (is.null(df)) {
        stop("The dataset is not available, so predictions cannot be explained.")
      }

      # 1) collect what the user typed
      vals <- list(
        Day_of_Week = input$day, Hour = input$hour, Vehicle_Count = input$vehicles,
        Average_Speed = input$speed, Rainfall_mm = input$rain, Temperature_C = input$temp,
        Accident = input$accident, Road_Type = input$road, Holiday = input$holiday,
        Traffic_Signal_Count = input$signals, Road_Width = input$width
      )

      # 2) check for empty / invalid / out-of-range values
      errs <- validate_inputs(vals)

      if (length(errs) > 0) {
        list(ok = FALSE, errors = errs)
      } else {
        # 3) same data types and factor levels as the training data
        obs <- build_new_observation(vals)
        out <- predict_congestion(bundle, obs, input$model_choice)
        list(ok = TRUE, obs = obs, label = out$label, probs = out$probs, model = out$model,
             explanation = explain_prediction(obs, out$label, out$probs, df, out$model))
      }
    }, error = function(e) {
      list(ok = FALSE, errors = paste("Prediction could not be completed:", conditionMessage(e)))
    })

    pred_result(res)
    if (!isTRUE(res$ok)) {
      showNotification("Please check the input values.", type = "error")
    }
  })

  observeEvent(input$reset_btn, {
    updateSelectInput(session, "day", selected = "Monday")
    updateNumericInput(session, "hour", value = 8)
    updateNumericInput(session, "vehicles", value = 1450)
    updateNumericInput(session, "speed", value = 18)
    updateNumericInput(session, "rain", value = 4)
    updateNumericInput(session, "temp", value = 28)
    updateSelectInput(session, "accident", selected = "Yes")
    updateSelectInput(session, "road", selected = "Highway")
    updateSelectInput(session, "holiday", selected = "No")
    updateNumericInput(session, "signals", value = 2)
    updateNumericInput(session, "width", value = 16)
    updateSelectInput(session, "model_choice", selected = "Best model (auto)")
    pred_result(NULL)
  })

  output$prediction_ui <- renderUI({
    res <- pred_result()
    if (is.null(res)) {
      div(class = "placeholder-box", icon("info-circle"),
          " Enter the traffic conditions and click 'Predict Traffic'.")
    } else if (!isTRUE(res$ok)) {
      div(class = "alert alert-danger",
          strong("Please correct the following:"),
          tags$ul(lapply(res$errors, tags$li)))
    } else {
      tagList(
        div(class = paste0("result-box result-", res$label),
            paste0("Predicted Congestion Level: ", toupper(res$label))),
        p(class = "model-note", paste("Model used:", res$model)),
        plotOutput("pl_pred_probs", height = 220),
        h4("Input information"),
        tableOutput("tbl_pred_inputs"),
        h4("Why this prediction?"),
        tags$ul(lapply(res$explanation, tags$li))
      )
    }
  })

  output$pl_pred_probs <- renderPlot({
    res <- pred_result()
    req(res, isTRUE(res$ok))
    plot_prediction_probabilities(res$probs)
  })
  output$tbl_pred_inputs <- renderTable({
    res <- pred_result()
    req(res, isTRUE(res$ok))
    as.data.frame(inputs_table(res$obs))
  })
}

shinyApp(ui, server)
