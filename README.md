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

**Contact**

For any inquiries or further information regarding this project, please feel free to connect with me on LinkedIn: [https://www.linkedin.com/in/shail-k-patel/](https://www.linkedin.com/in/shail-k-patel/)