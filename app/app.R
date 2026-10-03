# ==============================================================================
# TITLE: Traffic Congestion Prediction and Analysis
# DESCRIPTION: Complete single-file project generating data, training models, 
# and launching an R Shiny Dashboard.
# ==============================================================================

# 1. LOAD LIBRARIES
library(shiny)
library(shinydashboard)
library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(nnet)
library(e1071)

# ==============================================================================
# 2. DATA GENERATION & PREPROCESSING
# ==============================================================================
set.seed(123) # Fixed seed for reproducibility

generate_data <- function(n = 1200) {
  # Generate realistic synthetic data
  Hour <- sample(0:23, n, replace = TRUE)
  Day_of_Week <- sample(c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"), n, replace = TRUE)
  Road_Type <- sample(c("Highway", "City Road", "Suburban"), n, replace = TRUE)
  Accident <- sample(c("Yes", "No"), n, replace = TRUE, prob = c(0.1, 0.9))
  Holiday <- sample(c("Yes", "No"), n, replace = TRUE, prob = c(0.05, 0.95))
  Rainfall_mm <- round(runif(n, 0, 50), 1)
  Temperature_C <- round(runif(n, 10, 40), 1)
  Traffic_Signal_Count <- sample(1:10, n, replace = TRUE)
  Road_Width <- sample(c(10, 15, 20, 30), n, replace = TRUE)
  
  # Logic for Vehicle Count
  Vehicle_Count <- ifelse(Hour %in% c(8,9,17,18,19), 
                          round(runif(n, 500, 1500)), # Peak hours
                          round(runif(n, 50, 600)))   # Off-peak hours
  
  # Logic for Speed
  Average_Speed <- 80 - (Vehicle_Count * 0.04) - (Rainfall_mm * 0.2)
  Average_Speed <- ifelse(Accident == "Yes", Average_Speed - 20, Average_Speed)
  Average_Speed <- round(ifelse(Average_Speed < 5, 5, Average_Speed), 1)
  
  # Logic for Congestion Level (Target Variable)
  Congestion_Score <- (Vehicle_Count * 0.5) - (Average_Speed * 2) + 
    ifelse(Accident == "Yes", 200, 0) + (Rainfall_mm * 2)
  
  Congestion_Level <- case_when(
    Congestion_Score > 400 ~ "High",
    Congestion_Score > 150 ~ "Medium",
    TRUE ~ "Low"
  )
  
  data <- data.frame(
    Date = as.Date("2026-01-01") + sample(1:365, n, replace = TRUE),
    Time = paste0(sprintf("%02d", Hour), ":00"),
    Day_of_Week, Hour, Vehicle_Count, Average_Speed, Rainfall_mm, 
    Temperature_C, Accident, Road_Type, Holiday, Traffic_Signal_Count, 
    Road_Width, Congestion_Level
  )
  
  # Basic Preprocessing: Convert characters to factors
  data <- data %>% mutate_if(is.character, as.factor)
  return(data)
}

# Generate and save dataset
traffic_df <- generate_data()
write.csv(traffic_df, "traffic_data.csv", row.names = FALSE)

# ==============================================================================
# 3. MACHINE LEARNING MODEL TRAINING
# ==============================================================================
# Train-Test Split (80:20)
set.seed(42)
trainIndex <- createDataPartition(traffic_df$Congestion_Level, p = .8, list = FALSE)
train_data <- traffic_df[trainIndex, ]
test_data  <- traffic_df[-trainIndex, ]

# Features to use
features <- c("Hour", "Vehicle_Count", "Average_Speed", "Rainfall_mm", 
              "Temperature_C", "Accident", "Road_Type", "Holiday")

# Train Models
# 1. Logistic Regression (Multinomial)
log_model <- multinom(Congestion_Level ~ ., data = train_data[, c(features, "Congestion_Level")], trace = FALSE)
# 2. Decision Tree
dt_model <- rpart(Congestion_Level ~ ., data = train_data[, c(features, "Congestion_Level")], method = "class")
# 3. Random Forest
rf_model <- randomForest(Congestion_Level ~ ., data = train_data[, c(features, "Congestion_Level")])

# Helper function to get metrics
get_metrics <- function(model, type="other") {
  if(type == "dt"){
    preds <- predict(model, test_data, type = "class")
  } else {
    preds <- predict(model, test_data)
  }
  cm <- confusionMatrix(as.factor(preds), as.factor(test_data$Congestion_Level))
  
  # Calculate Macro Averages
  accuracy <- round(cm$overall['Accuracy'], 4)
  precision <- round(mean(cm$byClass[,'Pos Pred Value'], na.rm=T), 4)
  recall <- round(mean(cm$byClass[,'Sensitivity'], na.rm=T), 4)
  f1 <- round(mean(cm$byClass[,'F1'], na.rm=T), 4)
  
  return(c(Accuracy=accuracy, Precision=precision, Recall=recall, F1_Score=f1))
}

# Calculate actual metrics
log_metrics <- get_metrics(log_model)
dt_metrics <- get_metrics(dt_model, "dt")
rf_metrics <- get_metrics(rf_model)

metrics_table <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(log_metrics[1], dt_metrics[1], rf_metrics[1]),
  Precision = c(log_metrics[2], dt_metrics[2], rf_metrics[2]),
  Recall = c(log_metrics[3], dt_metrics[3], rf_metrics[3]),
  F1_Score = c(log_metrics[4], dt_metrics[4], rf_metrics[4])
)

# ==============================================================================
# 4. R SHINY DASHBOARD UI
# ==============================================================================
ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "Traffic Monitor Pro"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Home", tabName = "home", icon = icon("home")),
      menuItem("Dataset", tabName = "dataset", icon = icon("table")),
      menuItem("Traffic Analysis", tabName = "analysis", icon = icon("chart-bar")),
      menuItem("Visualization", tabName = "viz", icon = icon("chart-line")),
      menuItem("Model Performance", tabName = "models", icon = icon("cogs")),
      menuItem("Traffic Prediction", tabName = "predict", icon = icon("car")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    )
  ),
  
  dashboardBody(
    tabItems(
      # Tab 1: Home
      tabItem(tabName = "home",
              h2("Traffic Congestion Prediction and Analysis"),
              h4("A B.Tech Mini Project Using R and Machine Learning"),
              hr(),
              box(width = 12, title = "Problem Statement", status = "primary",
                  p("Urban areas face severe traffic congestion during peak hours, bad weather, or accidents. This causes travel delays, fuel waste, and pollution.")),
              box(width = 12, title = "Project Objective", status = "success",
                  p("To analyze historical traffic patterns and predict the likelihood of Low, Medium, or High congestion using Machine Learning algorithms (Random Forest, Decision Tree, Logistic Regression)."))
      ),
      
      # Tab 2: Dataset
      tabItem(tabName = "dataset",
              h2("Dataset View (Synthetic Real-World Data)"),
              fluidRow(
                valueBox(nrow(traffic_df), "Total Records", icon = icon("list"), color = "blue"),
                valueBox(ncol(traffic_df), "Variables", icon = icon("columns"), color = "purple")
              ),
              box(width = 12, dataTableOutput("data_table"))
      ),
      
      # Tab 3: Analysis
      tabItem(tabName = "analysis",
              h2("Statistical Analysis"),
              fluidRow(
                box(width = 6, title = "Summary Statistics", status = "info", verbatimTextOutput("summary_stats")),
                box(width = 6, title = "Key Metrics", status = "warning",
                    p(strong("Mean Vehicle Count: "), round(mean(traffic_df$Vehicle_Count), 2)),
                    p(strong("Median Vehicle Count: "), median(traffic_df$Vehicle_Count)),
                    p(strong("Average Speed (km/h): "), round(mean(traffic_df$Average_Speed), 2)),
                    p(strong("Correlation (Vehicles vs Speed): "), round(cor(traffic_df$Vehicle_Count, traffic_df$Average_Speed), 2))
                )
              )
      ),
      
      # Tab 4: Visualization
      tabItem(tabName = "viz",
              h2("Exploratory Data Analysis (EDA)"),
              fluidRow(
                box(width = 6, plotOutput("plot_hourly")),
                box(width = 6, plotOutput("plot_speed_veh"))
              ),
              fluidRow(
                box(width = 6, plotOutput("plot_weather")),
                box(width = 6, plotOutput("plot_congestion_dist"))
              )
      ),
      
      # Tab 5: Model Performance
      tabItem(tabName = "models",
              h2("Machine Learning Evaluation"),
              box(width = 12, title = "Model Comparison Table", status = "primary",
                  tableOutput("metrics_table")),
              box(width = 12, title = "Why Random Forest?", status = "success",
                  p("Random Forest generally performs best because it creates an ensemble of decision trees, reducing overfitting and handling the non-linear relationships between variables like hour, speed, and accidents efficiently."))
      ),
      
      # Tab 6: Prediction
      tabItem(tabName = "predict",
              h2("Live Traffic Congestion Predictor"),
              fluidRow(
                box(width = 4, title = "Input Conditions", status = "warning",
                    numericInput("p_hour", "Hour of Day (0-23):", value = 8, min=0, max=23),
                    numericInput("p_veh", "Vehicle Count:", value = 1200, min=0),
                    numericInput("p_speed", "Average Speed (km/h):", value = 20, min=0),
                    numericInput("p_rain", "Rainfall (mm):", value = 10, min=0),
                    numericInput("p_temp", "Temperature (C):", value = 25),
                    selectInput("p_acc", "Accident:", choices = c("No", "Yes")),
                    selectInput("p_road", "Road Type:", choices = levels(traffic_df$Road_Type)),
                    selectInput("p_hol", "Holiday:", choices = c("No", "Yes")),
                    actionButton("btn_predict", "Predict Traffic", class = "btn-primary")
                ),
                box(width = 8, title = "Prediction Result", status = "success",
                    h3(textOutput("pred_result"), style="color:blue; font-weight:bold;"),
                    hr(),
                    h4("Explanation:"),
                    textOutput("pred_explain")
                )
              )
      ),
      
      # Tab 7: About
      tabItem(tabName = "about",
              h2("About Project"),
              box(width=12,
                  h4("Technologies Used:"),
                  tags$ul(
                    tags$li("R Programming & RStudio"),
                    tags$li("R Shiny (Dashboard UI)"),
                    tags$li("ggplot2 (Visualization)"),
                    tags$li("RandomForest, rpart, nnet (Machine Learning)")
                  ),
                  h4("Limitations:"),
                  p("This system uses synthetic data. Real-world implementation requires API integration with live traffic sensors (e.g., Google Maps API).")
              )
      )
    )
  )
)

# ==============================================================================
# 5. R SHINY SERVER LOGIC
# ==============================================================================
server <- function(input, output) {
  
  # Dataset table
  output$data_table <- renderDataTable({ traffic_df }, options = list(pageLength = 10))
  
  # Summary Stats
  output$summary_stats <- renderPrint({ summary(traffic_df[, c("Vehicle_Count", "Average_Speed", "Rainfall_mm")]) })
  
  # Plots
  output$plot_hourly <- renderPlot({
    ggplot(traffic_df, aes(x = factor(Hour), y = Vehicle_Count)) +
      geom_boxplot(fill = "skyblue") + theme_minimal() +
      labs(title = "Vehicle Count by Hour", x = "Hour of Day", y = "Vehicles")
  })
  
  output$plot_speed_veh <- renderPlot({
    ggplot(traffic_df, aes(x = Vehicle_Count, y = Average_Speed, color = Congestion_Level)) +
      geom_point(alpha=0.6) + theme_minimal() +
      labs(title = "Speed vs Vehicle Count", x = "Vehicle Count", y = "Speed")
  })
  
  output$plot_weather <- renderPlot({
    ggplot(traffic_df, aes(x = Congestion_Level, y = Rainfall_mm, fill = Congestion_Level)) +
      geom_boxplot() + theme_minimal() +
      labs(title = "Rainfall Impact on Congestion")
  })
  
  output$plot_congestion_dist <- renderPlot({
    ggplot(traffic_df, aes(x = Congestion_Level, fill = Congestion_Level)) +
      geom_bar() + theme_minimal() +
      labs(title = "Distribution of Congestion Levels")
  })
  
  # Metrics Table
  output$metrics_table <- renderTable({ metrics_table })
  
  # Prediction Logic
  observeEvent(input$btn_predict, {
    tryCatch({
      
      # FIXED CODE: Ensure categorical inputs match the exact factor levels of the training dataset
      new_data <- data.frame(
        Hour = as.numeric(input$p_hour),
        Vehicle_Count = as.numeric(input$p_veh),
        Average_Speed = as.numeric(input$p_speed),
        Rainfall_mm = as.numeric(input$p_rain),
        Temperature_C = as.numeric(input$p_temp),
        Accident = factor(input$p_acc, levels = levels(traffic_df$Accident)),
        Road_Type = factor(input$p_road, levels = levels(traffic_df$Road_Type)),
        Holiday = factor(input$p_hol, levels = levels(traffic_df$Holiday))
      )
      
      # Predict using Random Forest (Best Model)
      prediction <- predict(rf_model, new_data)
      
      output$pred_result <- renderText({
        paste("Predicted Congestion Level:", toupper(as.character(prediction)))
      })
      
      output$pred_explain <- renderText({
        reason <- ""
        if(input$p_acc == "Yes") reason <- "An accident is severely restricting traffic flow. "
        if(input$p_rain > 15) reason <- paste(reason, "Heavy rainfall is causing slow speeds. ")
        if(input$p_veh > 800) reason <- paste(reason, "High volume of vehicles on the road. ")
        
        if(reason == "") reason <- "Normal traffic conditions are expected based on current volume and speed."
        
        paste("Based on the input parameters:", reason)
      })
      
    }, error = function(e) {
      output$pred_result <- renderText({paste("Error in Prediction:", e$message)})
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)