# Student-Dropout-Prediction-University-of-Kigali
# AUTHOR — NTAMBARA RUKAKA STEVEN
# SUPERVISOR — DR MUSABE JEAN BOSCO

This notebook provides a comprehensive exploratory data analysis of the Student Dropout dataset, exploring factors that contribute to student dropout rates. Dataset Characteristics:      10,000 student records     19 features (12 numeric, 7 categorical)     Binary target: Dropout (0=No, 1=Yes)     23.54% dropout rate

The dataset contains student data from University of Kigali Main Campus (and comparable institutional records) used to predict `Dropout` status, addressing the complex reasons behind student attrition. It includes 10,000 samples, focusing on scenarios that help predict student dropout risk. Attributes include age, gender, family income, academic performance, attendance, stress index, and behavioral factors.

<p align="center">
  <img src="https://uok.ac.rw/wp-content/uploads/2024/05/NEW-UoK-Logo-e1717660118908.png" alt="University of Kigali Logo" width="220"/>
</p>

<h1 align="center">Student Dropout Prediction — University of Kigali</h1>
<p align="center"><b>Big Data Analytics for Early Dropout Risk Detection & Intervention</b></p>

<p align="center">
  <a href="https://learn.uok.ac.rw/my/"><img src="https://img.shields.io/badge/LMS-learn.uok.ac.rw-2E5395?style=flat-square" /></a>
  <a href="https://uok.ac.rw"><img src="https://img.shields.io/badge/University-uok.ac.rw-004225?style=flat-square" /></a>
  <img src="https://img.shields.io/badge/Analytics-Python%20%2B%20scikit--learn-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/status-active-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" />
</p>

---

## 📌 Overview

**University of Kigali (UoK)** — like most higher-education institutions — loses a meaningful share of enrolled students to dropout every academic year. Today, that loss is only visible *after the fact*: a student stops appearing on attendance sheets, stops submitting assignments, and is eventually withdrawn — with no unified, real-time view of *who* is at risk and *why*, until it is too late to intervene.

This project delivers:

1. A **structured Big Data analytics pipeline** — cleaning, imputation, outlier treatment, and feature engineering — applied to a 10,000-record student dataset (`student_dropout_dataset_v3.csv`).
2. **Exploratory data analysis (EDA)** uncovering the demographic, academic, and behavioral patterns behind dropout at UoK-scale institutions.
3. A **prediction layer** (Logistic Regression, Decision Tree, Random Forest, SVM) that flags students at risk of dropout early enough for academic/student-affairs staff to intervene — before withdrawal is finalised.
4. A full **scientific research manuscript** reporting methodology, results, and evidence-based recommendations for an institutional early-warning system.

---

## 🎯 Problem / Causes of Dropout

Student dropout is rarely caused by one factor — it is the cumulative result of academic, financial, behavioral, and psychological pressures. Based on this dataset and the wider literature, the leading causes examined are:

- **Declining academic performance** — falling GPA, Semester GPA, and CGPA relative to a student's own historical trend.
- **Elevated stress levels** — the strongest non-academic correlate of dropout in this dataset (Stress Index).
- **Poor attendance and low engagement** — irregular class attendance and low weekly study hours.
- **Financial pressure** — low family income, and the need to balance a part-time job alongside studies.
- **Logistical burden** — long commute/travel time to campus.
- **Delayed academic conduct** — late assignment submissions as an early behavioral warning sign.
- **Lack of institutional support signals** — absence of scholarship support for financially vulnerable students.

Institutionally, the core problem is that:
- Academic records, attendance systems, and financial-aid data live in **separate systems** with no unified, cross-cutting view of at-risk students.
- Advisors have **no early-warning signal** for which students are drifting toward dropout, or why (grades? stress? finances? logistics?).
- Interventions are applied **reactively and uniformly**, rather than being targeted at the students who need them most.

## ✅ Solution

A data pipeline and analytics layer that:

1. Cleans and standardises institutional student records (demographic, academic, behavioral) into a single analysis-ready dataset.
2. Engineers derived indicators — `GPA_Trend` (short-term academic trajectory) and `Engagement_Score` (composite attendance + study-hours signal) — that sharpen prediction beyond raw fields.
3. Trains and compares multiple machine-learning classifiers to score each student's probability of dropout.
4. Surfaces the dominant, evidence-ranked risk factors (via feature importance) so advisors know *what* to act on, not just *who* to contact.
5. Documents the full pipeline, findings, and recommendations in a reproducible research manuscript suitable for institutional adoption.

---

## 🔗 University of Kigali Digital Properties

Relevant UoK digital properties this analytics work is designed to integrate with, for future live data collection (source: [uok.ac.rw](https://uok.ac.rw)):

| Property | URL | Role in this project |
|---|---|---|
| Main website | https://uok.ac.rw | Institutional context, public-facing reporting |
| Admissions | https://admissions.uok.ac.rw | Enrolment-funnel context for incoming-student risk baselining |
| Online Application | https://apply.uok.ac.rw | Application-stage demographic/financial-aid data source |
| "UoK Online Learning Portal" | https://learn.uok.ac.rw/my/ | Source of attendance, submission, and engagement behavioural data |
| Student Portal (MIS) | https://mis.uok.ac.rw | Student records, GPA/CGPA, registration & academic-year status |
| Student Portal (MyCampus) | https://mycampus.uok.ac.rw | Student self-service; potential channel for risk-flag notifications |
| E-Learning Portal (alias) | https://elearning.uok.ac.rw | Secondary LMS access point |
| Digital / E-Library | http://ebooks.uok.ac.rw | Supplementary engagement signal (resource access) |
| Institutional Repository | https://repository.uok.ac.rw | Research & academic content engagement |
| Academic Calendar | https://uok.ac.rw/academic-calendar/ | Context for term/semester start dates, aligning risk scoring to term timing |

**Social channels** (for advisor-outreach/notification integrations):
[Facebook](https://www.facebook.com/universityofkigaliUOK) · [X (Twitter)](https://x.com/UnivOfKigali) · [YouTube](https://www.youtube.com/@universityofkigali-uok7543) · [LinkedIn](https://www.linkedin.com/school/university-of-kigali/) · [Instagram](https://www.instagram.com/univ_of_kigali/)

> ⚠️ URLs above were verified against the live UoK website at the time of writing and may change; always confirm against https://uok.ac.rw before wiring up live data feeds.

---

## 🏗️ Data Source Architecture

```
Student Records (MIS)          LMS / Moodle Logs             Financial-Aid / Admissions
   (GPA, CGPA, Semester,          (Attendance, login             (Family_Income,
    Department, Age,               frequency, assignment          Scholarship status)
    Gender, Parental_Ed)           delay days)
        │                              │                                │
        └──────────────┬───────────────┴────────────────┬───────────────┘
                        ▼                                ▼
              Unified Student Analytics Dataset (student_dropout_dataset_v3.csv)
                        │
                        ▼
              ┌─────────────────────────┐
              │  Preprocessing Pipeline  │
              │  • Missing-value imputation
              │  • IQR outlier capping
              │  • Feature engineering
              │    (GPA_Trend, Engagement_Score)
              └───────────┬─────────────┘
                          ▼
              ┌─────────────────────────┐
              │  Exploratory Analysis    │
              │  (12 charts: distribution,
              │   correlation, by-department,
              │   by-semester, stress, etc.)
              └───────────┬─────────────┘
                          ▼
              ┌─────────────────────────┐
              │  Machine Learning Models │
              │  Logistic Regression ·   │
              │  Decision Tree ·          │
              │  Random Forest · SVM      │
              └───────────┬─────────────┘
                          ▼
              ┌─────────────────────────┐
              │  Dropout Risk Score /    │
              │  Advisor Early-Warning   │
              │  Dashboard (future work) │
              └─────────────────────────┘
```

At UoK scale (10K records), this pipeline runs on a single machine with pandas/scikit-learn. For institution-wide, multi-year, multi-campus data (Kacyiru, Remera, Musanze), the identical logic can be re-implemented on **Apache Spark** (PySpark + MLlib) with data staged in **HDFS**, near-real-time student events streamed via **Apache Kafka**, and risk scores served through **SQL** data marts or **MongoDB** collections to an advising dashboard.

---

## ⚙️ Setup and Configuration

### 1. Clone & set up the environment

```bash
git clone https://github.com/your-org/Student-Dropout-Prediction-UOK.git
cd Student-Dropout-Prediction-UOK
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables (`.env`)

```env
DATASET_PATH=data/student_dropout_dataset_v3.csv
OUTPUT_DIR=outputs/
MIS_API_BASE_URL=https://mis.uok.ac.rw
MIS_API_TOKEN=your_mis_api_token
LMS_BASE_URL=https://learn.uok.ac.rw
LMS_API_TOKEN=your_moodle_webservice_token
RANDOM_SEED=42
```

### 3. Run the analytics pipeline

```bash
python pipeline.py
```

This performs data cleaning → imputation → outlier capping → feature engineering → EDA chart generation → model training/evaluation, and writes all outputs to `outputs/` (cleaned dataset, `figures/`, `results_report.json`).

### 4. Generate the research manuscript

```bash
node build_paper.js
```

Produces `Student_Dropout_Research_Paper.docx` from the latest `results_report.json` and `figures/`.

---

## 🗂️ Dataset — `student_dropout_dataset_v3.csv`

**Records:** 10,000 students · **Attributes:** 19 · **Overall dropout rate:** 23.5%

| Column | Description |
|---|---|
| `Student_ID` | Unique student identifier |
| `Age`, `Gender`, `Parental_Education` | Demographics |
| `Family_Income` | Annual household income |
| `Internet_Access` | Home internet access (Yes/No) |
| `Study_Hours_per_Day` | Average daily self-study hours |
| `Attendance_Rate` | Percentage class attendance |
| `Assignment_Delay_Days` | Average late-submission days |
| `Travel_Time_Minutes` | Commute time to campus |
| `Part_Time_Job` | Holds part-time employment (Yes/No) |
| `Scholarship` | Receives scholarship (Yes/No) |
| `Stress_Index` | Self-reported stress score |
| `GPA`, `Semester_GPA`, `CGPA` | Academic performance measures |
| `Semester` | Academic year (Year 1–4) |
| `Department` | Academic department |
| **`Dropout`** | **Target: 1 = dropped out, 0 = retained** |

Engineered features added during preprocessing:

| Feature | Description |
|---|---|
| `GPA_Trend` | Semester_GPA − CGPA — short-term academic trajectory |
| `Engagement_Score` | Weighted composite of normalised attendance rate and study hours |

> See `outputs/cleaned_dataset.csv` for the cleaned, imputed, outlier-capped, feature-engineered dataset used for modeling, and `outputs/results_report.json` for the full machine-readable results summary.

---

## 🖼️ Screenshots & Visual References

<p align="center">
  <img src="https://uok.ac.rw/wp-content/uploads/2026/05/Web-banner-1-01-scaled-1.jpg" alt="University of Kigali campus banner" width="600"/>
</p>
<p align="center"><i>University of Kigali — Kacyiru (main), Remera, and Musanze campuses</i></p>

**Analytics outputs generated by this project** (in `outputs/figures/`):

| Figure | Description |
|---|---|
| `fig1_class_distribution.png` | Class distribution of the Dropout target |
| `fig2_dropout_by_department.png` | Dropout rate by academic department |
| `fig3_dropout_by_semester.png` | Dropout rate by academic year |
| `fig4_cgpa_distribution.png` | CGPA distribution: retained vs. dropped out |
| `fig5_attendance_boxplot.png` | Attendance rate vs. dropout status |
| `fig6_correlation_heatmap.png` | Correlation matrix of numeric features |
| `fig7_stress_violin.png` | Stress Index distribution by dropout status |
| `fig8_job_scholarship.png` | Dropout rate by part-time job & scholarship status |
| `fig9_model_comparison.png` | Accuracy / precision / recall / F1 / ROC-AUC across all 4 models |
| `fig10_roc_curves.png` | ROC curves for all classifiers |
| `fig11_confusion_matrix.png` | Confusion matrix — Random Forest |
| `fig12_feature_importance.png` | Random Forest feature-importance ranking |

---

## 🏗️ Project Structure

```
Student-Dropout-Prediction-UOK/
├── data/
│   └── student_dropout_dataset_v3.csv     # Raw dataset (input)
├── outputs/
│   ├── cleaned_dataset.csv                # Cleaned/engineered dataset (output)
│   ├── results_report.json                # Machine-readable summary of all findings
│   ├── figures/                           # 12 generated EDA + model charts (PNG)
│   └── Student_Dropout_Research_Paper.docx # Full scientific manuscript (final deliverable)
├── app/
│   ├── analytics/
│   │   ├── mis_client.py                  # MIS (student records) API client (future work)
│   │   └── moodle_client.py               # Moodle Web Services API client (future work)
│   ├── models/
│   │   └── dropout_predictor.py           # Dropout classification model wrapper
│   └── routes/
│       └── dashboard.py                   # Staff-facing risk dashboard (future work)
├── pipeline.py                            # Full preprocessing + EDA + modeling script
├── build_paper.js                         # Research-manuscript generation script (docx)
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

---

## ▶️ How to Run

```bash
git clone https://github.com/your-org/Student-Dropout-Prediction-UOK.git
cd Student-Dropout-Prediction-UOK
cp .env.example .env                 # fill in MIS/LMS credentials if using live data
pip install -r requirements.txt
python pipeline.py                   # cleans data, runs EDA, trains models, writes outputs/
node build_paper.js                  # builds the full research-paper docx
```

| Output | Location |
|---|---|
| Cleaned dataset | `outputs/cleaned_dataset.csv` |
| Charts | `outputs/figures/*.png` |
| Results summary (JSON) | `outputs/results_report.json` |
| Research manuscript | `outputs/Student_Dropout_Research_Paper.docx` |
| Live LMS (reference) | https://learn.uok.ac.rw/my/ |
| University site (reference) | https://uok.ac.rw |

```bash
pytest tests/ -v --cov=app
```

---

## 🧰 Environment & Tool Versions

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| pandas | 3.0+ | Data wrangling & cleaning |
| numpy | 2.4+ | Numeric computation |
| scikit-learn | 1.8+ | Machine learning (Logistic Regression, Decision Tree, Random Forest, SVM) |
| matplotlib | 3.10+ | Visualization |
| seaborn | 0.13+ | Statistical visualization |
| Node.js + `docx` (npm) | 20+ | Research-manuscript (.docx) generation |
| Moodle Web Services API | REST/JSON | LMS behavioural data source (`learn.uok.ac.rw`), future live integration |
| PostgreSQL | 16 | Analytics warehouse (future live deployment) |
| Apache Spark / Hadoop / Kafka | — | Scale-up path for institution-wide, multi-year data (see Data Source Architecture) |
| Docker | 25+ | Containerisation (future live deployment) |

---

## 📈 Key Findings Snapshot

- Overall dropout rate: **23.5%** (2,354 of 10,000 students).
- Strongest predictors: **CGPA, Semester GPA, GPA** (all negatively correlated with dropout, r ≈ −0.44 to −0.46).
- Strongest non-academic predictor: **Stress Index** (r ≈ +0.25).
- **Logistic Regression** achieved the best F1-score (0.585) and ROC-AUC (0.821); **Random Forest** achieved the highest accuracy (78.0%).
- Demographic factors (Gender, Internet Access, Parental Education, Family Income) showed minimal influence relative to academic and well-being indicators.

Full methodology, tables, and discussion are documented in `outputs/Student_Dropout_Research_Paper.docx`.

---

## 📚 Reference Links

- University of Kigali — https://uok.ac.rw
- UoK Online Learning Portal (LMS) — https://learn.uok.ac.rw/my/
- Admissions — https://admissions.uok.ac.rw
- Apply Online — https://apply.uok.ac.rw
- Student Portal (MIS) — https://mis.uok.ac.rw
- Student Portal (MyCampus) — https://mycampus.uok.ac.rw
- Digital/E-Library — http://ebooks.uok.ac.rw
- Institutional Repository — https://repository.uok.ac.rw
- Academic Calendar — https://uok.ac.rw/academic-calendar/
- scikit-learn documentation — https://scikit-learn.org/stable/documentation.html
- pandas documentation — https://pandas.pydata.org/docs/
- Apache Spark documentation — https://spark.apache.org/docs/latest/
- Moodle Web Services documentation — https://docs.moodle.org/dev/Web_services
- Ntambara Rukaka Steven Github — https://github.com/Ramjan0487/Student-Dropout-Prediction

---

## 📄 License

MIT — see `LICENSE`.

---

<p align="center"><i>Not an official University of Kigali product. Built for academic/research purposes (Big Data Analytics coursework) using publicly available UoK URLs and assets as of July 2026.</i></p>
