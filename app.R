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
source("R/data_preprocessing.R")
source("R/analysis.R")
source("R/visualization.R")
source("R/model_training.R")

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
