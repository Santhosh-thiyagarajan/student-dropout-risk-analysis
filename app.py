# ============================================
# Student Retention & Dropout Risk Analysis
# Frontend: Streamlit
# Backend: Python + Machine Learning
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Student Dropout Risk Analysis",
    layout="wide"
)

st.title("🎓 Student Retention & Dropout Risk Analysis")
st.write("Machine Learning based Early Warning System")

# ============================================
# Load Dataset
# ============================================

@st.cache_data
def load_data():
    return pd.read_csv(""C:\Users\santh\OneDrive - Rathinam Group Of Institutions\Desktop\Dropout Project\university_student_retention_dataset_2134.csv"")

df = load_data()

# ============================================
# Dataset Overview
# ============================================

st.subheader("📊 Dataset Overview")
st.write("Dataset Shape:", df.shape)
st.dataframe(df.head())

# ============================================
# Dropout Distribution
# ============================================

st.subheader("📈 Dropout Distribution")

fig1, ax1 = plt.subplots()
sns.countplot(x="next_semester_dropout", data=df, ax=ax1)
ax1.set_title("Dropout Distribution")
st.pyplot(fig1)

# ============================================
# Feature & Target Split
# ============================================

X = df.drop("next_semester_dropout", axis=1)
y = df["next_semester_dropout"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# ============================================
# Preprocessing Pipeline
# ============================================

numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ============================================
# Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================================
# Sidebar: Model Selection
# ============================================

st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest"]
)

# ============================================
# Model Building
# ============================================

if model_choice == "Logistic Regression":
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
else:
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        ))
    ])

model.fit(X_train, y_train)

# ============================================
# Predictions
# ============================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ============================================
# Evaluation Metrics
# ============================================

st.subheader("📌 Model Performance")

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
st.write("ROC-AUC Score:", roc_auc)

# Confusion Matrix
fig2, ax2 = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
ax2.set_title("Confusion Matrix")
st.pyplot(fig2)

# ============================================
# ROC Curve
# ============================================

fpr, tpr, _ = roc_curve(y_test, y_prob)

fig3, ax3 = plt.subplots()
ax3.plot(fpr, tpr, label="ROC Curve")
ax3.plot([0, 1], [0, 1], "--")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curve")
ax3.legend()
st.pyplot(fig3)

# ============================================
# Feature Importance (Random Forest)
# ============================================

if model_choice == "Random Forest":
    st.subheader("🔍 Feature Importance")

    rf_clf = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_clf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df.head(10))

# ============================================
# Risk Score Distribution
# ============================================

st.subheader("⚠️ Dropout Risk Score Distribution")

fig4, ax4 = plt.subplots()
sns.histplot(y_prob, bins=30, kde=True, ax=ax4)
ax4.set_title("Dropout Risk Scores")
ax4.set_xlabel("Probability of Dropout")
st.pyplot(fig4)

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("**Project:** Student Retention & Dropout Risk Analysis using Machine Learning")
