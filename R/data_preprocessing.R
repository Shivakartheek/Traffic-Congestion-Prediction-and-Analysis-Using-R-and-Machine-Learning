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
