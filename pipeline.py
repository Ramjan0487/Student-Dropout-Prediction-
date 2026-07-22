import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, roc_curve, classification_report)
import json
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
FIG = "/home/claude/analysis/figures"

RAW_PATH = "/mnt/user-data/uploads/student_dropout_dataset_v3.csv"
df = pd.read_csv(RAW_PATH)

report = {}
report["raw_shape"] = df.shape
report["raw_missing"] = df.isna().sum().to_dict()
report["dropout_counts_raw"] = df["Dropout"].value_counts().to_dict()

# ---------------------------------------------------------------
# 1. DATA CLEANING / PREPROCESSING
# ---------------------------------------------------------------
df_clean = df.copy()

# Duplicate check
n_dupes = df_clean.duplicated(subset=[c for c in df_clean.columns if c != "Student_ID"]).sum()
report["duplicate_rows"] = int(n_dupes)
df_clean = df_clean.drop_duplicates(subset=[c for c in df_clean.columns if c != "Student_ID"])

# Impute numeric missing values with median (robust to outliers)
numeric_missing = ["Family_Income", "Study_Hours_per_Day", "Stress_Index"]
medians = {}
for col in numeric_missing:
    med = df_clean[col].median()
    medians[col] = float(med)
    df_clean[col] = df_clean[col].fillna(med)

# Impute categorical missing with mode
mode_val = df_clean["Parental_Education"].mode()[0]
df_clean["Parental_Education"] = df_clean["Parental_Education"].fillna(mode_val)
report["imputation"] = {"numeric_medians": medians, "categorical_mode_Parental_Education": mode_val}

# Outlier handling via IQR capping on key continuous features
outlier_cols = ["Age", "Family_Income", "Study_Hours_per_Day", "Attendance_Rate",
                "Travel_Time_Minutes", "Stress_Index", "GPA", "Semester_GPA", "CGPA"]
outlier_report = {}
for col in outlier_cols:
    q1, q3 = df_clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    outlier_report[col] = int(n_out)
    df_clean[col] = df_clean[col].clip(lower, upper)
report["outliers_capped"] = outlier_report

# Feature engineering
df_clean["GPA_Trend"] = df_clean["Semester_GPA"] - df_clean["CGPA"]
df_clean["Engagement_Score"] = (
    (df_clean["Attendance_Rate"] / 100) * 0.5
    + (df_clean["Study_Hours_per_Day"] / df_clean["Study_Hours_per_Day"].max()) * 0.5
)
df_clean["Semester_Num"] = df_clean["Semester"].str.extract(r"(\d+)").astype(int)

df_clean.to_csv("/home/claude/analysis/cleaned_dataset.csv", index=False)
report["clean_shape"] = df_clean.shape

# ---------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------

# Dropout rate overall
dropout_rate = df_clean["Dropout"].mean()
report["overall_dropout_rate"] = float(dropout_rate)

# Fig 1: Dropout distribution
fig, ax = plt.subplots(figsize=(5, 4))
counts = df_clean["Dropout"].value_counts().sort_index()
ax.bar(["Retained (0)", "Dropped Out (1)"], counts.values, color=["#4C72B0", "#C44E52"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, str(v), ha="center", fontweight="bold")
ax.set_title("Class Distribution of Student Outcomes")
ax.set_ylabel("Number of Students")
plt.tight_layout()
plt.savefig(f"{FIG}/fig1_class_distribution.png")
plt.close()

# Fig 2: Dropout rate by department
fig, ax = plt.subplots(figsize=(7, 4.5))
dept_rate = df_clean.groupby("Department")["Dropout"].mean().sort_values(ascending=False)
sns.barplot(x=dept_rate.index, y=dept_rate.values, ax=ax, palette="Reds_r")
ax.set_title("Dropout Rate by Department")
ax.set_ylabel("Dropout Rate")
ax.set_xlabel("Department")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{FIG}/fig2_dropout_by_department.png")
plt.close()
report["dropout_by_department"] = dept_rate.to_dict()

# Fig 3: Dropout rate by semester/year
fig, ax = plt.subplots(figsize=(6, 4))
sem_rate = df_clean.groupby("Semester")["Dropout"].mean().sort_index()
sns.barplot(x=sem_rate.index, y=sem_rate.values, ax=ax, palette="Blues_r")
ax.set_title("Dropout Rate by Academic Year")
ax.set_ylabel("Dropout Rate")
plt.tight_layout()
plt.savefig(f"{FIG}/fig3_dropout_by_semester.png")
plt.close()
report["dropout_by_semester"] = sem_rate.to_dict()

# Fig 4: CGPA distribution by dropout status
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.kdeplot(data=df_clean, x="CGPA", hue="Dropout", fill=True, common_norm=False, ax=ax, palette=["#4C72B0", "#C44E52"])
ax.set_title("CGPA Distribution: Retained vs. Dropped Out")
plt.tight_layout()
plt.savefig(f"{FIG}/fig4_cgpa_distribution.png")
plt.close()

# Fig 5: Attendance vs dropout
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.boxplot(data=df_clean, x="Dropout", y="Attendance_Rate", ax=ax, palette=["#4C72B0", "#C44E52"])
ax.set_xticklabels(["Retained", "Dropped Out"])
ax.set_title("Attendance Rate vs. Dropout Status")
plt.tight_layout()
plt.savefig(f"{FIG}/fig5_attendance_boxplot.png")
plt.close()

# Fig 6: Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
numeric_cols = ["Age", "Family_Income", "Study_Hours_per_Day", "Attendance_Rate",
                "Assignment_Delay_Days", "Travel_Time_Minutes", "Stress_Index",
                "GPA", "Semester_GPA", "CGPA", "GPA_Trend", "Engagement_Score", "Dropout"]
corr = df_clean[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.savefig(f"{FIG}/fig6_correlation_heatmap.png")
plt.close()
report["correlation_with_dropout"] = corr["Dropout"].drop("Dropout").sort_values(ascending=False).to_dict()

# Fig 7: Stress index vs dropout
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.violinplot(data=df_clean, x="Dropout", y="Stress_Index", ax=ax, palette=["#4C72B0", "#C44E52"])
ax.set_xticklabels(["Retained", "Dropped Out"])
ax.set_title("Stress Index Distribution by Dropout Status")
plt.tight_layout()
plt.savefig(f"{FIG}/fig7_stress_violin.png")
plt.close()

# Fig 8: Part-time job & scholarship effect
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
pt_rate = df_clean.groupby("Part_Time_Job")["Dropout"].mean()
sc_rate = df_clean.groupby("Scholarship")["Dropout"].mean()
axes[0].bar(pt_rate.index, pt_rate.values, color=["#55A868", "#C44E52"])
axes[0].set_title("Dropout Rate by Part-Time Job Status")
axes[0].set_ylabel("Dropout Rate")
axes[1].bar(sc_rate.index, sc_rate.values, color=["#55A868", "#C44E52"])
axes[1].set_title("Dropout Rate by Scholarship Status")
plt.tight_layout()
plt.savefig(f"{FIG}/fig8_job_scholarship.png")
plt.close()

# ---------------------------------------------------------------
# 3. MODELING
# ---------------------------------------------------------------
model_df = df_clean.copy()
cat_cols = ["Gender", "Internet_Access", "Part_Time_Job", "Scholarship",
            "Department", "Parental_Education"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    encoders[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

feature_cols = ["Age", "Gender", "Family_Income", "Internet_Access", "Study_Hours_per_Day",
                "Attendance_Rate", "Assignment_Delay_Days", "Travel_Time_Minutes",
                "Part_Time_Job", "Scholarship", "Stress_Index", "GPA", "Semester_GPA",
                "CGPA", "Semester_Num", "Department", "Parental_Education",
                "GPA_Trend", "Engagement_Score"]

X = model_df[feature_cols]
y = model_df["Dropout"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
}

results = {}
roc_data = {}
for name, model in models.items():
    if name in ["Logistic Regression", "SVM (RBF)"]:
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_data[name] = (fpr.tolist(), tpr.tolist())

    if name == "Random Forest":
        rf_model = model
        rf_preds = preds
        rf_proba = proba

report["model_results"] = results

# Fig 9: Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
metrics_df = pd.DataFrame(results).T
metrics_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar", ax=ax, colormap="viridis")
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG}/fig9_model_comparison.png")
plt.close()

# Fig 10: ROC curves
fig, ax = plt.subplots(figsize=(6, 5.5))
for name, (fpr, tpr) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Model Comparison")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG}/fig10_roc_curves.png")
plt.close()

# Fig 11: Confusion matrix (Random Forest = best interpretable model)
fig, ax = plt.subplots(figsize=(5, 4.5))
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Retained", "Dropped Out"], yticklabels=["Retained", "Dropped Out"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Random Forest")
plt.tight_layout()
plt.savefig(f"{FIG}/fig11_confusion_matrix.png")
plt.close()
report["rf_confusion_matrix"] = cm.tolist()

# Fig 12: Feature importance (Random Forest)
fig, ax = plt.subplots(figsize=(7, 6))
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
importances.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Feature Importance — Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{FIG}/fig12_feature_importance.png")
plt.close()
report["feature_importance"] = importances.sort_values(ascending=False).to_dict()

# Best model summary
best_model_name = max(results, key=lambda k: results[k]["f1"])
report["best_model"] = best_model_name
report["best_model_metrics"] = results[best_model_name]

with open("/home/claude/analysis/results_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("DONE")
print(json.dumps(report["model_results"], indent=2))
print("Best model:", best_model_name)
print("Overall dropout rate:", report["overall_dropout_rate"])
