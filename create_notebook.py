import nbformat as nbf

nb = nbf.v4.new_notebook()

# Define the cells
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("# Customer Churn Analysis & Prediction\n\nThis notebook covers Data Cleaning, Exploratory Data Analysis (EDA), and Machine Learning modeling to predict customer churn."))

# Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
"""))

# Load Data
cells.append(nbf.v4.new_markdown_cell("## 1. Load Dataset"))
cells.append(nbf.v4.new_code_cell("""data_path = '../data/customer_churn.csv'
df = pd.read_csv(data_path)
print("Shape:", df.shape)
display(df.head())
"""))
cells.append(nbf.v4.new_code_cell("""print("Data Info:")
df.info()
print("\\nMissing Values:\\n", df.isnull().sum())
print("\\nDuplicates:", df.duplicated().sum())
"""))

# Data Cleaning
cells.append(nbf.v4.new_markdown_cell("## 2. Data Cleaning & Feature Engineering"))
cells.append(nbf.v4.new_code_cell("""# TotalCharges is object, need to convert to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Fill missing TotalCharges with 0 or drop
df['TotalCharges'].fillna(0, inplace=True)
# Drop customerID as it's not a feature
df.drop('customerID', axis=1, inplace=True)

print("Missing values after cleaning:\\n", df.isnull().sum())
"""))

# EDA
cells.append(nbf.v4.new_markdown_cell("## 3. Exploratory Data Analysis"))
cells.append(nbf.v4.new_code_cell("""# 1. Overall Churn Rate
fig = px.pie(df, names='Churn', title='Overall Churn Rate', hole=0.4, color_discrete_sequence=['#3b82f6', '#ef4444'])
fig.show()
"""))
cells.append(nbf.v4.new_code_cell("""# 2. Churn by Contract Type
fig = px.histogram(df, x='Contract', color='Churn', barmode='group', title='Churn by Contract Type')
fig.show()
"""))
cells.append(nbf.v4.new_code_cell("""# 3. Monthly Charges vs Churn
fig = px.box(df, x='Churn', y='MonthlyCharges', color='Churn', title='Monthly Charges vs Churn')
fig.show()
"""))
cells.append(nbf.v4.new_code_cell("""# Correlation Matrix for Numeric Features
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numeric_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
"""))

# ML Prep
cells.append(nbf.v4.new_markdown_cell("## 4. Machine Learning Data Prep"))
cells.append(nbf.v4.new_code_cell("""# Encoding Categorical Variables
categorical_cols = df.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('Churn', axis=1)
y = df['Churn']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""))

# Modeling
cells.append(nbf.v4.new_markdown_cell("## 5. Machine Learning Models"))
cells.append(nbf.v4.new_code_cell("""models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

results = []
best_model = None
best_auc = 0

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if any(y_prob) else 0
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": auc
    })
    
    if auc > best_auc:
        best_auc = auc
        best_model = model

results_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
display(results_df)
"""))

cells.append(nbf.v4.new_code_cell("""# Feature Importance (Using Best Model if tree-based)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), features[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Export Artifacts"))
cells.append(nbf.v4.new_code_cell("""os.makedirs('../models', exist_ok=True)
joblib.dump(best_model, '../models/best_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
joblib.dump(label_encoders, '../models/label_encoders.pkl')
print("Model, scaler, and encoders exported successfully!")
"""))

nb['cells'] = cells

with open('notebooks/churn_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook successfully generated!")
