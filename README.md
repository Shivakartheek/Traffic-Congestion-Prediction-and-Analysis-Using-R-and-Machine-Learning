# Traffic Congestion Prediction and Analysis Using R and Machine Learning

A college mini project that analyses traffic data, predicts congestion as **Low / Medium / High**
with three machine-learning models, and serves everything in an **R Shiny** dashboard
("Traffic Monitor").

> **Dataset note:** the data is **synthetic** (generated in R with `set.seed(123)`).
> No real traffic dataset is used. Model scores therefore show that the method works,
> not how it would perform on a real city.

## Folder structure

```text
traffic-congestion-project/
├── app.R                      <- Shiny dashboard (click "Run App")
├── run_pipeline.R             <- runs the whole project in the console, saves real results
├── traffic_data.csv           <- dataset (copy; created on first run)
├── requirements.txt           <- package list
├── README.md
├── data/traffic_data.csv      <- dataset used by the app (created on first run)
├── models/trained_model.rds   <- trained models (created on first run)
├── R/
│   ├── data_preprocessing.R   <- dataset generator, cleaning, input validation
│   ├── analysis.R             <- statistics and their explanation
│   ├── visualization.R        <- all ggplot2 graphs
│   └── model_training.R       <- 3 models, evaluation, prediction
├── outputs/                   <- graphs and results_summary.md (after run_pipeline.R)
├── documentation/
│   ├── project_report.md      <- full college report (27 sections)
│   ├── viva_questions.md      <- 30 viva questions with answers
│   └── code_explanation.md    <- beginner-friendly code explanation
└── single_file_version/app.R  <- everything in ONE file (see below)
```

## Quick start (RStudio)

1. Install R and RStudio.
2. Install packages (RStudio Console):
   ```r
   install.packages(c("shiny", "shinydashboard", "DT", "plotly", "tidyverse",
                      "dplyr", "ggplot2", "readr", "tidyr", "stringr", "tibble",
                      "caret", "randomForest", "rpart", "nnet", "e1071", "scales"))
   ```
3. Open `app.R` in RStudio (**File > Open File**, or open the project folder first:
   **File > Open Project/Folder** so the working directory is correct).
4. Click **Run App**.
   - On the first run the app creates `data/traffic_data.csv` and trains the models
     (about 10-40 seconds). Later runs start quickly.
5. Open the **Traffic Prediction** tab, press **Predict Traffic**.

## Get the real result numbers for your report

```r
source("run_pipeline.R")
```

This prints before/after data information, statistics, model metrics and confusion matrices,
saves all graphs to `outputs/plots/`, and writes **`outputs/results_summary.md`** containing
the real measured values. Copy them into Sections 20-21 of the report.

## Single-file version (easiest for beginners)

Open `single_file_version/app.R` and click **Run App**. It contains all the code
in one file and creates its own `data/` and `models/` folders next to it.

## Using your own dataset

Replace `data/traffic_data.csv` with a file that has the same 14 columns
(Date, Time, Day_of_Week, Hour, Vehicle_Count, Average_Speed, Rainfall_mm, Temperature_C,
Accident, Road_Type, Holiday, Traffic_Signal_Count, Road_Width, Congestion_Level),
delete `models/trained_model.rds`, and run the app again so the models are retrained.

## Troubleshooting

| Problem | Fix |
|---|---|
| `there is no package called ...` | Run the `install.packages()` command again |
| `cannot open file 'R/...'` | The working directory is wrong. Use **Session > Set Working Directory > To Source File Location** |
| Changed the dataset but results look old | Delete `models/trained_model.rds` and restart |
| Red message at top of the app | Read it - it explains which file or step failed |
