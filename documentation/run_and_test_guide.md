# Complete Step-by-Step Guide: Run and Test the Project in RStudio

## Part A - Installation

### Step 1: Install R
1. Go to https://cran.r-project.org
2. Choose your system (Windows / macOS / Linux) and download R 4.1 or newer.
3. Run the installer with default options.
   (Windows users: if packages fail to compile later, also install **Rtools** from the same site.)

### Step 2: Install RStudio
1. Go to https://posit.co/download/rstudio-desktop/
2. Download **RStudio Desktop** (free) and install it.
3. Start RStudio. The **Console** is the bottom-left window.

### Step 3: Install the required packages
Paste this into the Console and press Enter (needs internet, takes a few minutes):

```r
install.packages(c("shiny", "shinydashboard", "DT", "plotly", "tidyverse",
                   "dplyr", "ggplot2", "readr", "tidyr", "stringr", "tibble",
                   "caret", "randomForest", "rpart", "nnet", "e1071", "scales"))
```

If R asks "Do you want to install from sources?", type `n`. If it asks for a CRAN mirror, choose any nearby one.

### Step 4: Place the project files
1. Copy the whole folder `traffic-congestion-project` to an easy location, e.g. `C:/Projects/` or your Documents folder.
2. The dataset file goes to `data/traffic_data.csv` (a copy at `traffic_data.csv` in the main folder is also fine).
   **You do not have to create it yourself** - if it is missing, the project generates it automatically (synthetic data, `set.seed(123)`).
3. In RStudio: **File > Open Project/Folder...** (or **Session > Set Working Directory > Choose Directory**) and select `traffic-congestion-project`.
   Check in the Console: `getwd()` must end with `traffic-congestion-project`.

## Part B - Run

### Step 5 (recommended first): Run the console pipeline and get real results
```r
source("run_pipeline.R")
```
Expected:
- Printed dataset structure **before and after** cleaning, missing-value counts and an outlier table.
- Statistics and correlations with explanations.
- A model comparison table (Accuracy, Precision, Recall, F1) and three confusion matrices.
- Files created: `data/traffic_data.csv`, `traffic_data.csv`, `models/trained_model.rds`, `outputs/plots/*.png`, `outputs/model_comparison.csv`, `outputs/results_summary.md`.

Open `outputs/results_summary.md` and copy the tables into Sections 20-21 of `documentation/project_report.md`.

### Step 6: Open `app.R`
**File > Open File... > app.R**

### Step 7: Run the application
Click the **Run App** button at the top-right of the editor (or type `shiny::runApp()` in the Console).
The dashboard opens in an RStudio window; click **Open in Browser** for a bigger view.
On a first run without a saved model the app trains the models (roughly 10-40 seconds) before the window appears.

*Single-file alternative:* open `single_file_version/app.R` and click **Run App**.

## Part C - Test

### Step 8: Test the tabs
| Tab | What to check |
|---|---|
| Home | Four cards (records, variables, % High congestion, best model) and the pipeline steps |
| Dataset | Table is searchable; before/after table; outlier table; structure text |
| Traffic Analysis | Six statistic cards, explanation sentences, six graphs |
| Visualization | Six interactive graphs (hover to see values) |
| Model Performance | Comparison table with best model highlighted, chart, 3 confusion matrices, per-class table, feature importance |
| Traffic Prediction | See Step 9 |
| About | Text sections |

### Step 9: Test the prediction
1. Open **Traffic Prediction**. The form is pre-filled with a rush-hour example (Monday, hour 8, 1450 vehicles, 18 km/h, 4 mm rain, accident = Yes, Highway, holiday = No, 2 signals, width 16 m).
2. Click **Predict Traffic**.
3. You should see a coloured box *Predicted Congestion Level: ...*, probability bars, the input table and explanation sentences.
   The label is produced by the trained model - it is not typed in the code.
4. Try a calm situation: hour 2, 120 vehicles, 70 km/h, no rain, no accident. Compare the result.
5. Switch **Model to use** between the three models and compare the results.

### Step 10: Test the error handling
| Test | Expected |
|---|---|
| Clear the *Vehicle count* box, click Predict | Red message: value is empty or not a number |
| Hour = 30 | Red message: must be between 0 and 23 |
| Average speed = -5 | Red message: must be between 1 and 130 |
| Signal count = 2.5 | Red message: must be a whole number |
| Temporarily rename `data/traffic_data.csv`, restart app | Blue note: dataset was missing and was regenerated |
| Delete `models/trained_model.rds`, restart app | Blue note: model was missing and was trained again |
| Replace `models/trained_model.rds` with any text file, restart | Blue note: the file could not be read, a new model was trained |

### Step 11: Reset / retrain
- **Reset** button restores the default inputs.
- To use a new dataset: replace `data/traffic_data.csv`, delete `models/trained_model.rds`, rerun the app.

## Part D - Common problems
| Message | Solution |
|---|---|
| `there is no package called 'xyz'` | `install.packages("xyz")` |
| `cannot open file 'R/data_preprocessing.R'` | Working directory is wrong - set it to the project folder |
| App window blank for a long time | First run is training models; wait and watch the Console |
| Red message at top of the app | It names the failing step (dataset, cleaning or model) |
