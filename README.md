# PredictGrad

**Overview**

PredictGrad is a machine learning project designed to forecast the academic performance of students in core Semester 3 subjects within Computer Engineering (CE) and related programs at a local engineering college. By leveraging a dataset encompassing 905 students, the project aims to identify performance patterns and facilitate data-driven academic support initiatives.

**Dataset:** `student_performance_dataset.csv` (905 rows, 56 columns)
**Output:** `student_performance_with_percentiles.csv` (includes derived features)
**Branches Covered:** AIDS, AIML, CE, CEA, CS&IT, CSD, CSE, CST, IT, RAI

**Problem Definition**

The primary objective of this project is to predict student performance in the core Semester 3 subjects – Math-3, Digital Electronics (DE), Full Stack Development (FSD), and Python. This predictive capability, based on data from prior semesters, enables the early detection of students who may face academic challenges.

**Data Collection**

* **Source:** Digital academic records obtained directly from a local engineering college, spanning multiple departments (e.g., CE, IT, AIML).
* **Correctness:** High accuracy, as the data originates from official and verified records.
* **Process:** Records from various departments were consolidated into a unified dataset, ensuring consistency in the format of marks and attendance data.
* **Initial Size:** Over 905 students (before data cleaning).
* **Columns:** Included Student ID, Branch, Division (Div-1/2/3), Gender, Religion, theory and practical marks, attendance records, and mentor assignments.

**Data Preprocessing**

The preprocessing phase transformed the raw data into a format suitable for machine learning analysis through rigorous cleaning and insightful feature engineering. This process resulted in a refined dataset of 905 students.

**Data Cleaning**

* **Dropout Removal:** Students who had withdrawn from the college were excluded, leaving a dataset of 905 currently enrolled students.
* **Identifier Anonymization:** College IDs and student names were removed to protect privacy and replaced with unique, anonymized Student IDs.
* **Inferred Features:** Gender and Religion were inferred from student names using GPT-3.0, with an acknowledgment of potential inaccuracies inherent in this inference method.

**Feature Engineering**

* **Department Extraction:** The department for each student in Semesters 1/2 (A, B, D) and Semester 3 (A, B, C, D) was derived from the first character of their Division (e.g., A1 was transformed to A).
* **Semester Core Theory Totals:**
    * **Semester 1:** The sum of marks from Math-1 Theory, Physics Theory, Java-1 Theory, and Software Engineering Theory was calculated, creating a new column: `Sem1_Core_Theory_Total`.
    * **Semester 2:** The sum of marks from Math-2 Theory, Data Structures using Java Theory, DBMS Theory, Fundamental of Electronics and Electrical Theory, and Java-2 Theory was calculated, creating a new column: `Sem2_Core_Theory_Total`.
    * **Semester 3:** The sum of marks from Math-3 Theory, DE Theory, FSD Theory, and Python Theory was calculated, creating a new column: `Sem3_Core_Theory_Total`.
* **Percentile Calculation:** Percentile ranks (ranging from 0 to 100) were computed for the newly created semester total columns:
    * `Sem1_Core_Theory_Total` $\rightarrow$ `Sem1_Percentile`
    * `Sem2_Core_Theory_Total` $\rightarrow$ `Sem2_Percentile`
    * `Sem3_Core_Theory_Total` $\rightarrow$ `Sem3_Percentile`
* **Output:** All original columns, along with the newly engineered features, were saved to the file `student_performance_with_percentiles.csv`.
* **Script:** The Python script used for these calculations is `calculate_semester_totals_percentiles.py`.

# PredictGrad

## Overview

### Feature Selection for PredictGrad

This section outlines the features selected for modeling the `Sem3_Risk_Flag` (1 if Semester 3 percentile drops by 10+ from Semester 2, else 0) in the PredictGrad project. The dataset (`student_performance_with_features.csv`) was filtered to include 35 predictive features plus the output label, saved to `student_performance_model_features.csv`.

- **Dataset**: `student_performance_with_features.csv` (905 rows, 62 columns)  
- **Output**: `student_performance_model_features.csv` (36 columns: 35 features + `Sem3_Risk_Flag`)  
- **Script**: `select_model_features.py`

---

## Selected Features

The following 35 features were chosen for their relevance to predicting Semester 3 academic risk:

### Demographics
- `Gender`
- `Religion`

### Academic Grouping
- `Branch`
- `Roll-1`

### Semester 1/2 Marks

**Theory**:
- `Math-1 Theory`
- `Physics Theory`
- `Java-1 Theory`
- `Software Engineering Theory`
- `Math-2 Theory`
- `Data Structures using Java Theory`
- `DBMS Theory`
- `Fundamental of Electronics and Electrical Theory`
- `Java-2 Theory`

**Practical**:
- `Physics Practical`
- `Java-1 Practical`
- `Software Engineering Practical`
- `Data Structures using Java Practical`
- `DBMS Practical`
- `Fundamental of Electronics and Electrical Practical`
- `Java-2 Practical`

**Non-Core**:
- `Environmental Science Theory`
- `IOT Workshop Practical`
- `Computer Workshop Practical`

### Semester 1/2 Attendance
- `Math-1 Attendance`
- `Physics Attendance`
- `Java-1 Attendance`
- `Software Engineering Attendance`
- `Environmental Science Attendance`
- `IOT Workshop Attendance`
- `Math-2 Attendance`
- `Data Structures using Java Attendance`
- `DBMS Attendance`
- `Fundamental of Electronics and Electrical Attendance`
- `Java-2 Attendance`

### Engineered Features
- `Sem1_Core_Theory_Total`
- `Sem2_Core_Theory_Total`
- `Sem1_Percentile`
- `Sem2_Percentile`
- `Sem2_Sem1_Percentile_Diff`
- `Sem1_Core_Attendance_Avg`
- `Sem2_Core_Attendance_Avg`

---

## Feature Selection Table

| **Feature Category**     | **Features**                                         | **Reason for Inclusion**                                             | **Reason for Exclusion**                                          |
|--------------------------|------------------------------------------------------|----------------------------------------------------------------------|-------------------------------------------------------------------|
| **Demographics**         | Gender, Religion                                     | May reveal hidden patterns (e.g., socioeconomic or cultural effects) | -                                                                 |
| **Academic Grouping**    | Branch, Roll-1                                       | Branch captures program-specific trends; Roll-1 proxies school merit | Div-1/2/3, Roll-2/3 (redundant)                                  |
| **Semester 1/2 Marks**   | Theory, Practical, Non-Core (listed above)           | Direct indicators of academic performance and skills                 | Semester 3 marks, Sem3 total/percentile (to avoid leakage)       |
| **Attendance**           | All core and non-core attendance (listed above)      | Correlates with engagement and performance                           | -                                                                 |
| **Engineered Features**  | Totals, percentiles, diffs, attendance averages      | Summarize trends; diff captures decline                              | Sem3_Sem2_Percentile_Diff (leakage risk)                         |
| **Others**               | -                                                    | -                                                                    | Student ID, Mentor-1/2/3 (non-predictive)                        |

---

## Problems and Recommendations

- **High Feature Count**:  
  - *Issue*: 35 features may lead to overfitting (905 rows)  
  - *Recommendation*: Perform feature selection (e.g., correlation analysis, Random Forest feature importance)

- **GPT-3.0 Noise**:  
  - *Issue*: Gender and Religion may have inaccuracies  
  - *Recommendation*: Validate during EDA; consider dropping `Religion` if noisy

- **Non-Core Subjects**:  
  - *Issue*: May have lower predictive power  
  - *Recommendation*: Test importance in modeling

- **Additional Feature Ideas**:  
  - `Branch_Roll1_Rank`: Categorize `Roll-1` within each `Branch`  
  - `Sem2_Low_Attendance_Flag`: 1 if `Sem2_Core_Attendance_Avg` < 70%, else 0

- **General Note**:  
  - Feature set is logical, but pruning is advised for simplicity

---

## Usage

- **Modeling**: Use `student_performance_model_features.csv` to train a model (e.g., Random Forest) to predict `Sem3_Risk_Flag`
- **EDA**: Analyze feature correlations (e.g., attendance vs. risk) to refine feature selection

# Feature Selection for PredictGrad

## Overview

This section outlines the **final features** selected for modeling `Sem3_Risk_Flag` (1 if Semester 3 percentile drops by 10+ from Semester 2, else 0) in the **PredictGrad** project. The selected 12 features are optimized for tree-based models (e.g., Random Forest) to reduce multicollinearity and reflect attendance policy constraints (≥75% = bonus, <75% = penalty — not reflected in raw marks).

- **Dataset**: `student_performance_with_features.csv` (905 rows, 62 columns)  
- **Output**: `student_performance_model_features_v2.csv` (13 columns: 12 features + `Sem3_Risk_Flag`)  
- **Script**: `select_final_features_v5.py`

---

## Selected Features

### Demographics
- `Gender`
- `Religion`

### Academic Grouping
- `Branch`
- `Roll-1`

### Semester 1/2 Marks
- `Sem1_Core_Theory_Total`
- `Sem2_Core_Theory_Total`

### Semester 1/2 Attendance
- `Sem1_Core_Attendance_Avg`
- `Sem2_Core_Attendance_Avg`
- `Sem1_Attendance_Threshold` (1 if average ≥75%)
- `Sem2_Attendance_Threshold` (1 if average ≥75%)

### Engineered Features
- `Sem2_Percentile`
- `Sem2_Sem1_Percentile_Diff`

---

## Feature Selection Table

| **Feature Category**     | **Features**                                          | **Reason for Inclusion**                                                        | **Reason for Exclusion**                                                                 |
|--------------------------|-------------------------------------------------------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Demographics**         | `Gender`, `Religion`                                  | May reveal hidden patterns (e.g., socioeconomic/cultural effects)               | –                                                                                        |
| **Academic Grouping**    | `Branch`, `Roll-1`                                     | Captures academic trends and merit via school placement                         | `Div-1/2/3`, `Roll-2/3` — redundant with `Branch`, `Roll-1`                              |
| **Marks (Sem 1/2)**      | `Sem1/2_Core_Theory_Total`                             | Represents total core performance; complements percentile                       | Individual marks and non-core (e.g., `Math-1 Theory`, `Env Science Theory`)              |
| **Attendance (Sem 1/2)** | `Sem1/2_Core_Attendance_Avg`, `Sem1/2_Attendance_Threshold` | Combines detailed and policy-driven attendance signals                          | Individual subject attendance (e.g., `Math-1 Attendance`) due to redundancy              |
| **Engineered**           | `Sem2_Percentile`, `Sem2_Sem1_Percentile_Diff`        | Aligns with percentile-based target and performance trend                       | `Sem1_Percentile` (redundant); `Sem3_Sem2_Percentile_Diff` (leakage risk)               |
| **Others**               | –                                                     | –                                                                                | `Student ID`, `Mentor-1/2/3`, Semester 3 data (to avoid label leakage)                   |

---

## Correlation and Policy Notes

- **High Correlation**:  
  - Between `Sem1/2_Core_Theory_Total` and `Sem2_Percentile` — retained for modeling, watch for redundancy.
  - Between attendance averages and threshold flags — included for both granular and policy insights.

- **Attendance Policy**:  
  - ≥75% = bonus marks  
  - <75% = penalties  
  - **Note**: Raw marks are used (no bonus/penalty applied), but `Sem1/2_Attendance_Threshold` flags incorporate policy logic.

- **Standardization**:  
  - Not required for tree-based models (e.g., Random Forest).  
  - Raw attendance averages retained.

- **Model Recommendation**:  
  - **Random Forest** (robust to mixed data types, small sample size, and allows feature importance).

- **Feature Strategy**:  
  - Start with 12 selected features to **avoid overfitting**.  
  - Add subject-specific or niche features (e.g., `Java-1 Theory`) only if necessary.

---

## Recommendations

- Validate `Religion` field in EDA — drop if noisy.
- Run correlation analysis to confirm no high redundancy remains.
- After model training, inspect **feature importances**:
  - Drop low-impact features (e.g., `Sem1_Attendance_Threshold`) if needed.

---

## Usage

- **Modeling**: Use `student_performance_model_features_v2.csv` to train a **Random Forest** model for predicting `Sem3_Risk_Flag`.
- **EDA**: Explore distributions, correlations, and policy thresholds before training.

#  Preprocessing for PredictGrad

## Overview

This section outlines the features and preprocessing steps used to model **`Sem3_Risk_Flag`** (1 if Semester 3 percentile drops by ≥10 from Semester 2, else 0) in the PredictGrad project.

- **Dataset**: `student_performance_with_features.csv` (905 rows, 62 columns)  
- **Feature Selection Output**: `student_performance_model_features_v2.csv` (13 columns: 12 features + target)  
- **Preprocessing Output**: `student_performance_preprocessed.csv` (24 columns after one-hot encoding)  
- **Scripts**: 
  - `create_model_features_v2.py` (feature selection)  
  - `preprocess_features_v3.py` (preprocessing)

---

## Selected Features

### Demographics
- `Gender` → Encoded as: `Gender_M`
- `Religion` → Encoded as: `Religion_Hindu`, `Religion_Jain`, `Religion_Muslim`, `Religion_Sikh`

### Academic Grouping
- `Branch` → Encoded as: `Branch_AIML`, `Branch_CE`, `Branch_CEA`, `Branch_CS&IT`, `Branch_CSD`, `Branch_CSE`, `Branch_CST`, `Branch_IT`, `Branch_RAI`
- `Roll-1`

### Semester 1/2 Marks
- `Sem1_Core_Theory_Total`
- `Sem2_Core_Theory_Total`

### Semester 1/2 Attendance
- `Sem1_Core_Attendance_Avg`
- `Sem2_Core_Attendance_Avg`
- `Sem1_Attendance_Threshold`
- `Sem2_Attendance_Threshold`

### Engineered Features
- `Sem2_Percentile`
- `Sem2_Sem1_Percentile_Diff`

---

## Preprocessing Steps

### Categorical Encoding
- One-hot encode: `Gender`, `Religion`, `Branch`  
  → Use `drop_first=True` to prevent multicollinearity.

### Missing Values
- None found.  
- Imputation logic included:
  - Median for numerical
  - Mode for categorical (as fallback)

### Data Validation
- `Religion` distribution: Hindu dominant (829), minorities sparse (e.g., Sikh: 1)
  - Keep for now, monitor importance.
- `Sem3_Risk_Flag` distribution: 80.22% (class 0), 19.78% (class 1)
  - Use `class_weight='balanced'` in Random Forest

### Data Types
| Feature Type | Examples | Data Type |
|--------------|----------|-----------|
| Integer      | `Roll-1`, `Theory Totals` | `int32` |
| Float        | `Attendance_Avg`, `Percentile Diff` | `float32` |
| Binary       | One-hot features, thresholds, target | `int8` |



### Attendance Policy
- Use `Attendance_Avg` for engagement
- Use `Attendance_Threshold` to reflect ≥75% policy rule

---

## Feature Selection Table

| Feature Category      | Features                                     | Reason for Inclusion                                              | Reason for Exclusion                                        |
|-----------------------|----------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------|
| **Demographics**      | `Gender`, `Religion`                         | May reveal patterns (e.g., socioeconomic effects)                 | —                                                           |
| **Academic Grouping** | `Branch`, `Roll-1`                           | Branch captures trends; Roll-1 proxies merit                      | Div-1/2/3, Roll-2/3 (redundant)                             |
| **Sem 1/2 Marks**     | `Sem1/2_Core_Theory_Total`                   | Captures absolute academic performance                            | Individual/non-core marks (due to correlation)              |
| **Sem 1/2 Attendance**| `Core_Attendance_Avg`, `Attendance_Threshold`| Granular + policy-based representation                            | Individual attendance columns (highly correlated)           |
| **Engineered**        | `Sem2_Percentile`, `Sem2_Sem1_Percentile_Diff`| Aligns with target logic                                          | Sem1_Percentile (redundant), Sem3_Sem2_Diff (leakage risk) |
| **Others**            | —                                            | —                                                                 | ID, Mentor fields, Sem 3 marks (non-predictive or leakage)  |

---

## Correlation & Policy Notes

### 📌 High Correlations:
- `Sem1/2_Core_Theory_Total` ↔ `Sem2_Percentile`: **Kept**, monitor feature importance
- Attendance averages ↔ Threshold flags: **Complementary**, both retained

### Attendance Policy:
- `≥75%` earns **bonus marks**  
- `<75%` triggers **penalty**  
- Data uses **raw attendance**, so `Attendance_Threshold` is created manually.

---

## Model Notes & Strategy

- **Model**: Random Forest (good for:
  - Mixed data types
  - Small dataset (905 rows)
  - Feature interpretability)

- **Feature Strategy**: 
  - Start with 12 carefully selected features
  - Add niche/high-impact ones if model underperforms

---

## Recommendations

- **Religion**: Monitor feature importance; drop if noisy
- **Class Imbalance**: Use `class_weight='balanced'`
- **Post-modeling Pruning**:
  - Drop low-importance features (e.g., `Sem1_Attendance_Threshold` if needed)

---

## Usage

- **For Modeling**: Use `student_performance_preprocessed.csv` with Random Forest or other classifiers
- **For EDA**: Explore correlations, feature-target relationships, and cross-tabs

Here's your content converted into clean, well-structured **Markdown** for documentation or a project README:

---

# Exploratory Data Analysis (EDA) for PredictGrad

## Overview

This document details the exploratory data analysis (EDA) conducted for the **PredictGrad** project, which aims to predict the `Sem3_Risk_Flag`:

* `1` if a student's Semester 3 percentile drops by 10+ compared to Semester 2.
* `0` otherwise.

EDA was performed in two versions to refine features and validate their predictive power before modeling with Random Forest.

* **Dataset**:

  * `student_performance_preprocessed.csv` (v1)
  * `student_performance_preprocessed_v2.csv` (v2, 23 columns after dropping 1 feature)
* **Scripts**:

  * `eda_analysis.py` (v1, `artifact_id: 116bb1fa-4794-4784-81e8-77368335c9a8`)
  * `eda_analysis_v2.py` (v2, `artifact_id: 030ebc15-25bb-495b-9e13-f8952fdbbe59`)
* **Outputs**:

  * Plots saved to `eda_plots/` (v1) and `eda_plots_v2/` (v2)

---

## EDA Process

The analysis focused on **numerical features** to explore:

* **Correlation Matrix**: Identify multicollinearity.
* **Histograms**: Visualize feature distributions by class (`Sem3_Risk_Flag`).
* **Box Plots**: Compare medians and spreads by class.

---

## Version 1: `eda_analysis.py`

* **Dataset**: `student_performance_preprocessed.csv` (24 columns)
* **Numerical Features Analyzed**:
  `Roll-1`, `Sem1/2_Core_Theory_Total`, `Sem1/2_Core_Attendance_Avg`, `Sem2_Percentile`, `Sem2_Sem1_Percentile_Diff`

### Findings

#### Correlation Matrix

* **High Correlation** (0.99):
  `Sem2_Core_Theory_Total` ↔ `Sem2_Percentile` — severe multicollinearity
* **Moderate Correlation**:

  * `Sem1/2_Core_Attendance_Avg` ↔ `Sem2_Percentile` (\~0.26–0.31)
  * `Sem1_Core_Theory_Total` ↔ `Sem2_Percentile` (0.87)
* **Negative Correlation**:

  * `Roll-1` ↔ `Sem2_Percentile` (-0.32) — higher roll numbers link to lower performance

#### Histograms

* `Sem1/2_Core_Attendance_Avg`:

  * Both classes peak at 90–100%
  * Longer tail <75% for `Sem3_Risk_Flag = 1`
  * **Weak class separation**

* `Sem2_Percentile`:

  * `Sem3_Risk_Flag = 0`: Bimodal (\~20, \~80)
  * `Sem3_Risk_Flag = 1`: Centered at 40–60
  * **Strong separation**

#### Box Plots

* `Sem1/2_Core_Attendance_Avg`:

  * Medians overlap (90%), IQRs: 85–95%
  * More outliers <75% for at-risk students
  * **Weak predictor**

* `Sem2_Percentile`:

  * `Flag = 0`: Median \~65, IQR 40–80
  * `Flag = 1`: Median \~50, IQR 30–65
  * **Strong class separation**

---

## Version 2: `eda_analysis_v2.py`

### Changes

* **Dropped Feature**: `Sem2_Core_Theory_Total` due to 0.99 correlation with `Sem2_Percentile`
* **Dataset Updated**: `student_performance_preprocessed_v2.csv` (23 columns)
* **Numerical Features Analyzed**: 6 (excluding the dropped one)

### Findings

#### Correlation Matrix

* Multicollinearity reduced by dropping `Sem2_Core_Theory_Total`
* Other correlations remained stable:

  * `Sem1/2_Core_Attendance_Avg` ↔ `Sem2_Percentile` (\~0.26–0.31)
  * `Sem1_Core_Theory_Total` ↔ `Sem2_Percentile` (0.87)

#### Histograms & Box Plots

* Distributions unchanged since the dropped feature was not plotted
* **Key Insight**:

  * `Sem2_Percentile` remains a **strong predictor**
  * `Sem1/2_Core_Attendance_Avg` remains **weak**

---

## Parameter Changes

### Feature Reduction

* **Dropped**: `Sem2_Core_Theory_Total`

  * Columns reduced from 24 → 23
  * Features used by model: 12 → 11

### Updated Modeling Features (v2)

* `Gender`, `Religion`, `Branch`, `Roll-1`
* `Sem1_Core_Theory_Total`, `Sem1/2_Core_Attendance_Avg`
* `Sem1/2_Attendance_Threshold`, `Sem2_Percentile`
* `Sem2_Sem1_Percentile_Diff`

---

## Monitor Features

* **`Sem1/2_Core_Attendance_Avg`**:

  * Weak predictor; if feature importance is low, may drop and rely on `Sem1/2_Attendance_Threshold`

* **`Religion`**:

  * Sparse categories (e.g., Christian = 3, Sikh = 1)
  * Monitor for noise; may be excluded if unhelpful

---

## Class Imbalance Strategy

* Class distribution:

  * `Sem3_Risk_Flag = 0`: 80.22%
  * `Sem3_Risk_Flag = 1`: 19.78%
* **Approach**:

  * Use `class_weight='balanced'` in Random Forest

---

## How EDA Changed Our Approach

### Feature Reduction

* Dropped `Sem2_Core_Theory_Total` to:

  * Simplify the model
  * Reduce multicollinearity
  * Retain `Sem2_Percentile` as a cleaner, directly relevant predictor

### Feature Prioritization

* **Top Predictors**:

  * `Sem2_Percentile`
  * `Sem2_Sem1_Percentile_Diff`

* **Under Review**:

  * `Sem1/2_Core_Attendance_Avg` — weak separation
  * May rely instead on policy-aligned threshold (`Sem1/2_Attendance_Threshold`)

### Pitch Insight

* `Sem2_Percentile` clearly shows at-risk students clustering at 40–60 range

  * **Actionable insight**: Students <50th percentile may need intervention

### Model Strategy

* **Random Forest** chosen:

  * Handles mixed types (categorical + numerical)
  * Supports feature importance ranking
* **Metric Priority**:

---

Data Splitting for PredictGrad
Overview
This document outlines the data splitting process for the PredictGrad project to predict Sem3_Risk_Flag. A hybrid approach is used: 20% test set and 5-fold cross-validation (Train and CV) on the remaining 80%. This ensures robust training, hyperparameter tuning, and unbiased final evaluation for the college pitch.

Dataset: student_performance_preprocessed_v2.csv (905 rows, 23 columns: 11 features + Sem3_Risk_Flag).
Script: split_data.py
Outputs:
student_performance_fold_data.csv (724 rows, for 5-fold Train and CV).
student_performance_test.csv (181 rows, for final testing).



Splitting Process

Hybrid Approach:
Test Set: 20% (181 rows) reserved for final evaluation, mimicking real-world unseen data.
Train and CV Set: 80% (724 rows) for training and tuning via 5-fold CV.


Stratification:
Splits are stratified by Sem3_Risk_Flag to maintain class balance (80.22% Sem3_Risk_Flag = 0, 19.78% Sem3_Risk_Flag = 1).


Breakdown:
Test Set: 181 rows (~145/Sem3_Risk_Flag = 0, ~36/Sem3_Risk_Flag = 1).
Train and CV Set: 724 rows (~580/Sem3_Risk_Flag = 0, ~144/Sem3_Risk_Flag = 1).
5-Fold Train and CV: 579 train, ~145 validation per fold (116/29 per fold).




Random Seed: Set to 42 for reproducibility.

Class Distribution

Original: 80.22% (726) Sem3_Risk_Flag = 0, 19.78% (179) Sem3_Risk_Flag = 1.
Train and CV Set: 80.25% (580) Sem3_Risk_Flag = 0, 19.75% (144) Sem3_Risk_Flag = 1.
Test Set: 80.11% (145) Sem3_Risk_Flag = 0, 19.89% (36) Sem3_Risk_Flag = 1.

Feature Update and Data Splitting for PredictGrad (v3)
Overview
This document outlines the feature update and data splitting process for the PredictGrad project to predict Sem3_Risk_Flag. We updated the feature set by dropping low-impact features, preprocessed the data, and split it into train + CV and test sets, aligning with previous steps but reflecting the new feature set.

Input Dataset: student_performance_model_features_v2.csv (905 rows, 13 columns: 12 features + Sem3_Risk_Flag).
Script: update_features_and_split.py (artifact_id: e5a2d8f3-2c9e-4a1b-9f8d-6c7b4e1a0f2d).
Outputs:
Updated feature-selected dataset: student_performance_model_features_v3.csv (905 rows, 9 columns: 8 features + Sem3_Risk_Flag).
Preprocessed dataset: student_performance_preprocessed_v3.csv (905 rows, 18 columns after encoding).
Train + CV set: student_performance_fold_data_v3.csv (724 rows, 80%).
Test set: student_performance_test_v3.csv (181 rows, 20%).



Feature Updates

Dropped Features:
Religion: Low importance (e.g., Religion_Sikh: 0.0005) and sparse categories (e.g., Sikh: 1 student).
Gender: Low importance (Gender_M: 0.0138) and weak predictive power per EDA.
Sem1/2_Core_Attendance_Avg: Moderate importance (0.152, 0.099) but weak predictive power (overlapping distributions in EDA); rely on Sem1/2_Attendance_Threshold.


New Feature Set: Reduced from 12 features to 8 features (before encoding):
Remaining features: Roll-1, Sem1_Core_Theory_Total, Sem1/2_Attendance_Threshold, Sem2_Percentile, Sem2_Sem1_Percentile_Diff, Branch.



Preprocessing

One-Hot Encoding: Encoded Branch (e.g., Branch_AIML, Branch_CE, etc.), dropping the first category (Branch_AI), resulting in 9 Branch_* columns.
Data Types:
int32: Roll-1, Sem1_Core_Theory_Total.
float32: Sem2_Percentile, Sem2_Sem1_Percentile_Diff.
int8: Sem1/2_Attendance_Threshold, Sem3_Risk_Flag, Branch_*.



Splitting Process

Hybrid Approach:
Train + CV set: 80% (724 rows) for 5-fold CV.
Test set: 20% (181 rows) for final evaluation.


Stratification: Stratified by Sem3_Risk_Flag to maintain class balance (80.22% Sem3_Risk_Flag = 0, 19.78% Sem3_Risk_Flag = 1).
Random Seed: Set to 42 for reproducibility.

Feature Update and Model Retraining for PredictGrad (v3)
Overview
This document outlines the model retraining process for the PredictGrad project to predict Sem3_Risk_Flag using the updated dataset (v3) after dropping low-impact features. We applied SMOTE, custom class weights, and expanded hyperparameter tuning to improve performance on the minority class (Sem3_Risk_Flag = 1).

Dataset:
Train + CV set: student_performance_fold_data_v3.csv (724 rows, 18 columns).
Test set: student_performance_test_v3.csv (181 rows, 18 columns).


Script: update_and_retrain.py (artifact_id: 246cd228-ad71-4d56-9793-aebd784a4d5c).
Outputs:
Updated model: model_outputs_v2/random_forest_model_v2.pkl.
Updated feature importance: model_outputs_v2/feature_importance_v2.csv.



Retraining Process

Class Imbalance Adjustments:
SMOTE: Oversampled the minority class (Sem3_Risk_Flag = 1) in training folds to balance the class distribution (original: 80.25% Sem3_Risk_Flag = 0, 19.75% Sem3_Risk_Flag = 1).
Custom Class Weights: Set class_weight={0: 1, 1: 6} (1:6 ratio) to penalize false negatives, prioritizing recall for Sem3_Risk_Flag = 1.


Hyperparameter Tuning:
Expanded grid: n_estimators: [100, 200, 300], max_depth: [10, 20, None], min_samples_split: [2, 5], min_samples_leaf: [1, 2].
Scoring: F1-score (macro) to balance performance across classes.


Cross-Validation: 5-fold CV using StratifiedKFold (~579 train, ~145 validation per fold).






**Contact**

For any inquiries or further information regarding this project, please feel free to connect with me on LinkedIn: [https://www.linkedin.com/in/shail-k-patel/](https://www.linkedin.com/in/shail-k-patel/)