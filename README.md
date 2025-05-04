# Data Preprocessing Documentation

## Overview
This document outlines the preprocessing steps applied to the `student_performance_dataset.csv` (905 students, CE and CE-adjacent programs) to support predictive modeling of Semester 3 academic performance. The process involves feature engineering to calculate core theory mark totals and percentiles for Semesters 1, 2, and 3, enhancing the dataset for machine learning tasks.

- **Dataset**: `student_performance_dataset.csv`  
- **Output**: `student_performance_with_percentiles.csv`  
- **New Columns**:  
  - `Sem1_Core_Theory_Total`  
  - `Sem2_Core_Theory_Total`  
  - `Sem3_Core_Theory_Total`  
  - `Sem1_Percentile`  
  - `Sem2_Percentile`  
  - `Sem3_Percentile`  

---

## Preprocessing Steps

### 1. Load Dataset
- Read `student_performance_dataset.csv` (905 rows, 56 columns).

### 2. Calculate Semester 1 Core Theory Total
- **Columns**:  
  - `Math-1 Theory`  
  - `Physics Theory`  
  - `Java-1 Theory`  
  - `Software Engineering Theory`  
- **New Column**: `Sem1_Core_Theory_Total` (sum of the above)

### 3. Calculate Semester 2 Core Theory Total
- **Columns**:  
  - `Math-2 Theory`  
  - `Data Structures using Java Theory`  
  - `DBMS Theory`  
  - `Fundamental of Electronics and Electrical Theory`  
  - `Java-2 Theory`  
- **New Column**: `Sem2_Core_Theory_Total`

### 4. Calculate Semester 3 Core Theory Total
- **Columns**:  
  - `Math-3 Theory`  
  - `DE Theory`  
  - `FSD Theory`  
  - `Python Theory`  
- **New Column**: `Sem3_Core_Theory_Total`

### 5. Calculate Percentiles
- Computed percentile ranks (0–100) for:
  - `Sem1_Core_Theory_Total` → `Sem1_Percentile`  
  - `Sem2_Core_Theory_Total` → `Sem2_Percentile`  
  - `Sem3_Core_Theory_Total` → `Sem3_Percentile`  

### 6. Save Output
- Saved all original columns plus new columns to `student_performance_with_percentiles.csv`.

---

## Usage

**Purpose**: Enables percentile-based analysis (e.g., predict Semester 3 percentile drop) for identifying at-risk students.
