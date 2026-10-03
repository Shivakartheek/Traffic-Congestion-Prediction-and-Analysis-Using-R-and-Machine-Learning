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
