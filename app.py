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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, roc_curve, accuracy_score
)
import os

st.set_page_config(page_title="Student Dropout Risk Analytics", layout="wide", page_icon="🎓")

st.title("🎓 Student Retention / Dropout Risk Analytics System")
st.markdown("""
This system predicts the probability of student dropout using historical academic and demographic data. 
It helps institutions identify at-risk students and intervene before dropout occurs.
""")

# Data Upload
st.sidebar.header("Data Upload & Configuration")
st.sidebar.markdown("Upload a new dataset to generate risk scores for those students.")
uploaded_file = st.sidebar.file_uploader("Upload Student Dataset (CSV, Excel)", type=["csv", "xlsx", "xls"])

@st.cache_data
def load_default_data():
    if os.path.exists('university_student_retention_dataset_2134.csv'):
        return pd.read_csv('university_student_retention_dataset_2134.csv')
    return None

df_train = load_default_data()

if df_train is None:
    st.error("Critical Error: Core training file 'university_student_retention_dataset_2134.csv' is missing from the directory. The system cannot perform predictive modeling.")
    st.stop()

# Determine analysis dataset
df_analyze = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_analyze = pd.read_csv(uploaded_file)
        else:
            df_analyze = pd.read_excel(uploaded_file)
        st.sidebar.success(f"Custom dataset '{uploaded_file.name}' loaded for Analysis!")
    except Exception as e:
        st.sidebar.error(f"Error loading analysis file: {e}")

if df_analyze is None:
    st.sidebar.info("No external file uploaded for analysis. The risk scoring section will analyze the training dataset.")
    df_analyze = df_train.copy()

st.header("1. Core Training Dataset Overview")
st.write(f"The model forms its logic securely from the verified historical dataset: **{df_train.shape[0]} Records**")
st.dataframe(df_train.head(5))

# Dynamic Configuration
st.sidebar.subheader("Feature Configuration")

# Target Selection based on the TRAINING data
default_target = "next_semester_dropout" if "next_semester_dropout" in df_train.columns else df_train.columns[-1]
target_idx = df_train.columns.get_loc(default_target) if default_target in df_train.columns else len(df_train.columns)-1
target_col = st.sidebar.selectbox("Select Target Variable (Dropout) from Training Data", df_train.columns, index=target_idx)

# Features to drop (e.g., student_id)
default_drops = [col for col in df_train.columns if 'id' in col.lower() and col != target_col]
cols_to_drop = st.sidebar.multiselect("Select columns to ignore (e.g., IDs)", df_train.columns, default=default_drops)

# Preprocessing setup on TRAINING DATA
X_train_full = df_train.drop(columns=[col for col in [target_col] + cols_to_drop if col in df_train.columns])
y_train_full = df_train[target_col] if target_col in df_train.columns else pd.Series(np.zeros(len(df_train)))

# Drop columns that are completely empty
X_train_full = X_train_full.dropna(axis=1, how='all')

# Drop rows where target is missing
valid_idx = y_train_full.dropna().index
X_train_full = X_train_full.loc[valid_idx]
y_train_full = y_train_full.loc[valid_idx]

# Encode target if it is not numeric
if y_train_full.dtype == 'object' or str(y_train_full.dtype) == 'category' or y_train_full.dtype == 'bool':
    le = LabelEncoder()
    y_train_full = le.fit_transform(y_train_full)
else:
    # Try to ensure it's integer for classification
    y_train_full = y_train_full.astype(int)

# Dynamically find numerical and categorical features based on training data
numerical_features = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train_full.select_dtypes(exclude=[np.number]).columns.tolist()

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])

# Check for minimum number of classes
unique_classes = np.unique(y_train_full)
if len(unique_classes) < 2:
    st.error(f"Cannot train models: The historical training dataset's target variable '{target_col}' contains only one class ({unique_classes[0]}). "
             "Please ensure 'university_student_retention_dataset_2134.csv' contains both positive and negative dropout examples.")
    st.stop()

# Split training data for internal evaluation
X_tr, X_te, y_tr, y_te = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full)

st.header("2. Model Configuration & Training")

col1, col2 = st.columns([1, 2])
with col1:
    model_choice = st.selectbox("Select Machine Learning Model:", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(random_state=42)
else:
    model = RandomForestClassifier(random_state=42, n_estimators=100)

# Build Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

with st.spinner("Training model on historical data..."):
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    y_pred_proba = pipeline.predict_proba(X_te)[:, 1]

st.header("3. Model Performance Evaluation (Validation Set)")

# Calculate metrics
precision = precision_score(y_te, y_pred, zero_division=0)
recall = recall_score(y_te, y_pred, zero_division=0)
f1 = f1_score(y_te, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_te, y_pred_proba)
accuracy = accuracy_score(y_te, y_pred)

mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
mcol1.metric("Accuracy", f"{accuracy:.2f}")
mcol2.metric("Precision", f"{precision:.2f}")
mcol3.metric("Recall", f"{recall:.2f}")
mcol4.metric("F1 Score", f"{f1:.2f}")
mcol5.metric("ROC-AUC", f"{roc_auc:.2f}")

st.header("4. Visualizations & Analytics")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Continue', 'Dropout'], yticklabels=['Continue', 'Dropout'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    st.pyplot(fig)

with viz_col2:
    st.subheader("ROC Curve")
    fig, ax = plt.subplots(figsize=(6, 4))
    fpr, tpr, _ = roc_curve(y_te, y_pred_proba)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    st.pyplot(fig)

viz_col3, viz_col4 = st.columns(2)

with viz_col3:
    st.subheader("Dropout Risk Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(y_pred_proba, bins=20, kde=True, color='purple', ax=ax)
    ax.axvline(0.3, color='green', linestyle='--', label='Low/Moderate threshold (0.3)')
    ax.axvline(0.7, color='red', linestyle='--', label='Moderate/High threshold (0.7)')
    ax.set_xlabel('Predicted Dropout Probability')
    ax.set_ylabel('Frequency (Test Set)')
    ax.legend(loc="upper right", fontsize='small')
    st.pyplot(fig)

with viz_col4:
    st.subheader("Feature Importance")
    try:
        if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        else:
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names()
        clean_names = [name.split('__')[-1] for name in feature_names]
    except Exception:
        # Fallback if feature names fail to extract
        clean_names = [f"Feature {i}" for i in range(X_tr.shape[1])]
        
    if model_choice == "Random Forest":
        importances = pipeline.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[-10:] # Top 10 features
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(range(len(indices)), importances[indices], color='teal')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([clean_names[i] for i in indices])
        ax.set_xlabel('Relative Importance')
        st.pyplot(fig)
        
    elif model_choice == "Logistic Regression":
        coefficients = pipeline.named_steps['classifier'].coef_[0]
        abs_coef = np.abs(coefficients)
        indices = np.argsort(abs_coef)[-10:] # Top 10 features by absolute weight
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(range(len(indices)), coefficients[indices], color='teal')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([clean_names[i] for i in indices])
        ax.set_xlabel('Coefficient Value (Impact direction & magnitude)')
        st.pyplot(fig)

st.header("5. New Dataset Risk Scoring & Early Warning")
if uploaded_file is not None:
    st.write(f"Analyzing students from: **{uploaded_file.name}**")
else:
    st.write("Retrospectively analyzing students from the training set since no new data was uploaded.")

# Prepare analysis data properly matching training structure
X_analyze = df_analyze.drop(columns=[col for col in cols_to_drop + [target_col] if col in df_analyze.columns]).copy()

# Add missing columns with NaNs if the uploaded dataset doesn't perfectly match training (e.g. they didn't upload target or some fields)
for col in X_train_full.columns:
    if col not in X_analyze.columns:
        X_analyze[col] = np.nan
        
# Ensure order matches
X_analyze = X_analyze[X_train_full.columns]

# Calculate risk scores for full dataset to display a sample
df_preview = df_analyze.copy()

# Handle the risk calculation robustly
try:
    df_preview['risk_score'] = pipeline.predict_proba(X_analyze)[:, 1]
except Exception as e:
    df_preview['risk_score'] = 0.0 # Fallback
    st.sidebar.warning(f"Could not calculate risk scores for all rows: {e}")

def get_risk_level(score):
    if score < 0.3: return "Low Risk"
    elif score < 0.7: return "Moderate Risk"
    else: return "High Risk"

df_preview['risk_level'] = df_preview['risk_score'].apply(get_risk_level)

# Sort by risk score descending
df_preview = df_preview.sort_values('risk_score', ascending=False)

# Select columns to display: ID column (if any), top features (up to 4), and risk metrics
id_cols = [col for col in cols_to_drop if 'id' in col.lower() and col in df_preview.columns]
display_cols = id_cols + [col for col in numerical_features[:3] + categorical_features[:1] if col in df_preview.columns] + ['risk_score', 'risk_level']
# deduplicate just in case
display_cols = list(dict.fromkeys(display_cols))

st.dataframe(
    df_preview[display_cols]
    .head(50)
    .style.background_gradient(subset=['risk_score'], cmap='Reds')
)

st.info("💡 **Recommendation:** Prioritize mentoring and financial support programs for students categorized under 'High Risk'.")
