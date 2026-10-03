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
