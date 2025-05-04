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

**Contact**

For any inquiries or further information regarding this project, please feel free to connect with me on LinkedIn: [https://www.linkedin.com/in/shail-k-patel/](https://www.linkedin.com/in/shail-k-patel/)