# Student-Dropout-Prediction- NTAMBARA RUKAKA STEVEN
This notebook provides a comprehensive exploratory data analysis of the Student Dropout dataset, exploring factors that contribute to student dropout rates. Dataset Characteristics:      10,000 student records     19 features (12 numeric, 7 categorical)     Binary target: Dropout (0=No, 1=Yes)     23.54% dropout rate

# Student Dropout Prediction — Big Data Analytics Project

> A Big Data Analytics coursework project applying data preprocessing, exploratory data analysis (EDA), and machine learning to predict student dropout from institutional records.

---

## 1. Project Overview

Student attrition is costly for institutions and students alike. This project analyzes a **10,000-record student dataset** (`student_dropout_dataset_v3.csv`) to:

1. Clean and preprocess raw institutional data (missing values, outliers, feature engineering).
2. Explore relationships between demographic, behavioural, and academic factors and dropout.
3. Train and compare multiple machine learning classifiers to predict dropout risk.
4. Identify the strongest predictors of dropout using feature-importance analysis.
5. Produce a full scientific manuscript reporting methodology, results, and evidence-based recommendations.

This repository replaces the previous, unrelated "GovCA Image Validation System" content that was mistakenly present in this location — that project is unrelated to this coursework submission and has been removed.

---

## 2. Dataset

**File:** `student_dropout_dataset_v3.csv`
**Records:** 10,000 students · **Attributes:** 19

| Category | Fields |
|---|---|
| Demographic | Age, Gender, Family_Income, Parental_Education |
| Behavioural | Internet_Access, Study_Hours_per_Day, Attendance_Rate, Assignment_Delay_Days, Travel_Time_Minutes, Part_Time_Job, Stress_Index |
| Academic | GPA, Semester_GPA, CGPA, Semester, Department |
| Target | Dropout (1 = dropped out, 0 = retained) |

Overall dropout rate in the raw data: **23.5%** (2,354 of 10,000 students).

---

## 3. Methodology / Tools Used

| Stage | Tool / Library |
|---|---|
| Data wrangling & cleaning | Python, pandas, numpy |
| Missing-value imputation | Median (numeric), Mode (categorical) |
| Outlier treatment | IQR capping (winsorisation) |
| Feature engineering | GPA_Trend, Engagement_Score |
| Visualization | matplotlib, seaborn |
| Machine learning | scikit-learn — Logistic Regression, Decision Tree, Random Forest, SVM (RBF) |
| Evaluation | Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix |
| Manuscript generation | Python + docx (Word) |

> **Scalability note:** this pipeline runs on a single machine at the current 10K-row scale using pandas/scikit-learn. For institution-wide, multi-year data (millions of rows), the same logic is designed to port to **Apache Spark** (PySpark + MLlib) with data staged in **HDFS**, real-time student-event ingestion via **Kafka**, and risk scores served from **SQL** or **MongoDB** — as discussed in Section 3.5 of the research paper.

---

## 4. Repository Contents

```
├── README.md                              # This file
├── student_dropout_dataset_v3.csv         # Raw dataset (input)
├── cleaned_dataset.csv                    # Cleaned/engineered dataset (output)
├── results_report.json                    # Machine-readable summary of all findings
├── figures/                                # All 12 generated charts (PNG)
│   ├── fig1_class_distribution.png
│   ├── fig2_dropout_by_department.png
│   ├── fig3_dropout_by_semester.png
│   ├── fig4_cgpa_distribution.png
│   ├── fig5_attendance_boxplot.png
│   ├── fig6_correlation_heatmap.png
│   ├── fig7_stress_violin.png
│   ├── fig8_job_scholarship.png
│   ├── fig9_model_comparison.png
│   ├── fig10_roc_curves.png
│   ├── fig11_confusion_matrix.png
│   └── fig12_feature_importance.png
├── pipeline.py                            # Full preprocessing + EDA + modeling script
└── Student_Dropout_Research_Paper.docx    # Full scientific manuscript (final deliverable)
```

---

## 5. How to Reproduce

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python pipeline.py
```

This regenerates `cleaned_dataset.csv`, all figures in `figures/`, and `results_report.json`.

---

## 6. Key Findings (Summary)

- **Academic performance (GPA, Semester GPA, CGPA)** is the strongest predictor of dropout — students who drop out show markedly lower cumulative GPA.
- **Stress Index** is the second-strongest correlate of dropout, ahead of any demographic factor — student well-being should be monitored alongside grades.
- **Attendance rate and engagement** are meaningful secondary predictors and are observable well before a formal withdrawal.
- **Demographic factors (Gender, Internet Access, Parental Education, Family Income)** show only weak association with dropout in this dataset.
- **Logistic Regression** achieved the best F1-score / recall balance; **Random Forest** achieved the highest accuracy and precision — a two-tier screening approach (broad screen + precision re-ranking) is recommended for institutional deployment.

Full details, tables, and figures are in `Student_Dropout_Research_Paper.docx`.

---

## 7. Authors / Course Context

Prepared as a Big Data Analytics coursework submission demonstrating data preprocessing, analytical methods, visualization, interpretation of findings, and evidence-based conclusions, per assignment requirements.
