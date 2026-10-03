# Code Explanation (Beginner Friendly)

For every major part: **what it does**, **why it is needed**, **expected output**, **important R functions**.

---

## 1. `R/data_preprocessing.R`

### 1.1 Constants (`DAY_LEVELS`, `ROAD_LEVELS`, `INPUT_RULES` ...)
- **What:** Lists of allowed categories and the valid range of each number.
- **Why:** Training, cleaning and the Shiny form must all use the same rules.
- **Output:** None (they are stored for later use).
- **Functions:** `c()`, `tibble::tribble()`.

### 1.2 `generate_traffic_data()`
- **What:** Creates the synthetic dataset: random dates, hours (more at rush hours), road types, weather, accidents, vehicle count, speed and congestion level. Finally adds a few missing values, impossible values and duplicate rows.
- **Why:** No real dataset is used; the generator makes the project self-contained and reproducible.
- **Output:** A table of about 1,500 rows and 14 columns.
- **Functions:** `set.seed()` (same random numbers every time), `sample()` (random choice), `runif()`, `rnorm()`, `rexp()`, `rbinom()` (random numbers), `pmin()/pmax()` (limit values), `quantile()` and `cut()` (turn the score into Low/Medium/High), `bind_rows()`, `arrange()`.

### 1.3 `load_or_create_dataset()`
- **What:** Reads `data/traffic_data.csv`; if the file does not exist, generates and saves it.
- **Why:** Handles the "missing dataset" error without crashing.
- **Output:** The raw table plus information about where it came from.
- **Functions:** `file.exists()`, `readr::read_csv()`, `write_csv()`, `dir.create()`.

### 1.4 `preprocess_traffic_data()`
- **What:** Cleans the data step by step: tidy text, convert types, remove duplicates, set impossible values to missing, fill missing values (median/mode), convert categories to factors, report outliers with the IQR rule, and prepare numeric encodings.
- **Why:** Machine-learning models need clean, consistent data.
- **Output:** A list containing the clean data, a before/after table, an outlier table and the encoded table. With `verbose = TRUE` it prints dataset structure before and after.
- **Functions:** `glimpse()`, `colSums(is.na())`, `str_trim()`, `str_to_title()`, `as.Date()`, `as.numeric()`, `distinct()`, `median()`, `factor()`, `complete.cases()`, `quantile()`.

### 1.5 `encode_categoricals()`
- **What:** Adds numeric codes (Accident: No=0/Yes=1, Day: Monday=1 ... etc.).
- **Why:** Shows how categories become numbers. (The models do the same internally using dummy variables.)
- **Functions:** `as.integer()` on factors.

### 1.6 `validate_inputs()` and `build_new_observation()`
- **What:** Check what the user typed (empty, not a number, outside range, wrong category) and create a one-row table that looks exactly like the training data.
- **Why:** Gives friendly errors and guarantees the model receives correctly prepared data.
- **Output:** A vector of error messages (empty if everything is fine) / a one-row data frame.
- **Functions:** `is.na()`, `%in%`, `paste0()`, `factor(levels = ...)`.

---

## 2. `R/analysis.R`

### 2.1 `compute_statistics()`
- **What:** Mean, median, standard deviation, minimum, maximum of vehicle count; average speed; correlation of vehicle count with speed and rainfall with vehicle count.
- **Why:** Summarises the data in numbers.
- **Output:** A list with a table and individual values.
- **Functions:** `mean()`, `median()`, `sd()`, `min()`, `max()`, `cor()`.

### 2.2 `interpret_statistics()` and `describe_correlation()`
- **What:** Turn the numbers into sentences ("strong negative correlation ... vehicles move more slowly").
- **Why:** The project requires a simple-language explanation; building the text from real values keeps it correct.
- **Functions:** `paste0()`, `case_when()`, `ifelse()`.

### 2.3 `numeric_summary()` and `congestion_profile()`
- **What:** A summary table for all numeric columns, and average conditions for Low/Medium/High.
- **Functions:** `lapply()`, `bind_rows()`, `group_by()`, `summarise()`.

---

## 3. `R/visualization.R`

### 3.1 Ten EDA graphs (`plot_vehicles_by_hour()` ... `plot_hourly_congestion_trend()`)
- **What:** Each function takes the clean data and returns a ggplot graph.
- **Why:** Graphs show patterns faster than tables. The same functions are reused by the script and the dashboard.
- **Output:** A `ggplot` object (shows in the Plots pane or in the dashboard).
- **Functions:** `ggplot()`, `aes()`, `geom_col()`, `geom_boxplot()`, `geom_point()`, `geom_smooth()`, `geom_bar(position = "fill")`, `geom_line()`, `scale_fill_manual()`, `labs()`, `theme()`.

### 3.2 `theme_traffic()` and `congestion_colors`
- **What:** One consistent light theme and colours (green/amber/red).
- **Why:** Professional, easy-to-read look.

### 3.3 `plot_confusion_matrix()`, `plot_model_comparison()`, `plot_feature_importance()`, `plot_prediction_probabilities()`
- **What:** Model-related graphs: heat-map of the confusion matrix, bar chart of metrics, importance of each feature, and probability bars after a prediction.
- **Functions:** `geom_tile()`, `geom_text()`, `pivot_longer()`, `coord_flip()`, `reorder()`.

---

## 4. `R/model_training.R`

### 4.1 `split_data()`
- **What:** Splits the data 80% train / 20% test, keeping the class proportions the same.
- **Why:** The test part checks the model on data it has never seen.
- **Output:** A list with `train` and `test`.
- **Functions:** `set.seed()`, `caret::createDataPartition()`.

### 4.2 `train_models()`
- **What:** Trains Logistic Regression (`multinom`), Decision Tree (`rpart`) and Random Forest (`rf`) through one interface, `caret::train()`.
- **Why:** Three different algorithms can be compared fairly. Scaling for logistic regression is stored in the model and applied automatically to new data.
- **Output:** A list of three trained models.
- **Functions:** `trainControl()`, `train()`, `preProcess = c("center","scale")`.

### 4.3 `evaluate_model()`
- **What:** Predicts on the test data and computes accuracy and macro-averaged precision, recall and F1, plus the confusion matrix.
- **Why:** Gives measured performance - nothing is invented.
- **Output:** A list with the confusion matrix and metric tables.
- **Functions:** `predict()`, `confusionMatrix()`, `mean()`.

### 4.4 `train_all_models()`
- **What:** Runs split + train + evaluate, picks the best model (highest F1, accuracy breaks ties), extracts Random Forest feature importance and packs everything in one list ("bundle").
- **Output:** The bundle, saved to `models/trained_model.rds`.
- **Functions:** `bind_rows()`, `arrange()`, `randomForest::importance()`.

### 4.5 `save_bundle()`, `load_or_train_model()`
- **What:** Save/load the bundle with `saveRDS()` / `readRDS()`. If the file is missing or damaged a new model is trained.
- **Why:** The app starts fast and never crashes because of a model-file problem.

### 4.6 `predict_congestion()` and `explain_prediction()`
- **What:** Predict the class and probabilities for one row, then write simple reasons (rush hour, vehicle count compared with history, slow speed, rain, accident, holiday).
- **Output:** e.g. label "High" with probabilities for Low/Medium/High and a list of sentences.
- **Functions:** `predict(type = "prob")`, `which.max()`, `mean()`.

---

## 5. `app.R` (Shiny)

### 5.1 `init_app()`
- **What:** At start-up loads the data, cleans it and loads/trains the model. Problems are collected as messages.
- **Why:** The app shows a clear message instead of crashing.
- **Functions:** `tryCatch()`.

### 5.2 UI (`dashboardPage()`)
- **What:** Header, sidebar menu with seven items and seven tabs, value boxes, tables, plots, prediction form. A small CSS block creates the light theme.
- **Functions:** `dashboardHeader()`, `dashboardSidebar()`, `sidebarMenu()`, `tabItems()`, `box()`, `valueBoxOutput()`, `plotOutput()`, `plotlyOutput()`, `DTOutput()`, `selectInput()`, `numericInput()`, `actionButton()`.

### 5.3 Server (`server()`)
- **What:** Produces every output: `renderValueBox()`, `renderPlot()`, `renderPlotly(ggplotly(...))`, `renderDT()`, `renderTable()`.
- **Prediction:** When **Predict Traffic** is clicked, `observeEvent()` collects the inputs, runs `validate_inputs()`, builds the one-row data frame, calls `predict_congestion()` and stores the result in a `reactiveVal()`. `renderUI()` then shows either the error list or **Predicted Congestion Level: HIGH / MEDIUM / LOW**, probability bars, the input summary and the explanation.
- **Expected output:** A coloured result box (green/amber/red).

---

## 6. `run_pipeline.R`
- **What:** Runs the whole project in the console and saves graphs, the model, `outputs/model_comparison.csv` and `outputs/results_summary.md`.
- **Why:** Gives the real numbers for the report and proves the project works without the dashboard.
- **Output:** Printed tables in the Console and files in `outputs/`.
