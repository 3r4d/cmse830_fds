import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# -----------------------
# Streamlit Page Setup
# -----------------------
st.set_page_config(page_title="Stroke & Diabetes Analysis", layout="wide")
st.title("Stroke & Diabetes Dataset Analysis")
st.write("This app balances datasets, shows correlation heatmaps, and displays feature importance using Random Forests.")

# -----------------------
# Load Datasets
# -----------------------
st.header("1️⃣ Load Datasets")

# Stroke dataset
stroke_path = "C:/Users/brad_/MSU/Year 1/CMSE 830/Project/healthcare-dataset-stroke-data.csv"
df_stroke = pd.read_csv(stroke_path)
df_stroke = df_stroke.drop(columns=['id', 'ever_married', 'work_type', 'Residence_type'])
df_stroke = df_stroke[df_stroke['gender'] != 'Other']
st.subheader("Stroke Dataset")
st.dataframe(df_stroke.head())

# Diabetes dataset
diabetes_path = "C:/Users/brad_/MSU/Year 1/CMSE 830/Project/diabetes_prediction_dataset.csv"
new_order = ['gender', 'age', 'hypertension', 'heart_disease', 'blood_glucose_level',
             'bmi', 'smoking_history', 'HbA1c_level', 'diabetes']
df_diabetes = pd.read_csv(diabetes_path)
df_diabetes = df_diabetes[new_order]
st.subheader("Diabetes Dataset")
st.dataframe(df_diabetes.head())

# -----------------------
# Balance Stroke Dataset with SMOTE
# -----------------------
st.header("2️⃣ Balance Stroke Dataset (SMOTE)")

# Encode categorical variables
label_encoder = LabelEncoder()
df_stroke['gender'] = label_encoder.fit_transform(df_stroke['gender'])
if 'smoking_status' in df_stroke.columns:
    df_stroke['smoking_status'] = label_encoder.fit_transform(df_stroke['smoking_status'])

# Separate features and target
X_stroke = df_stroke.drop('stroke', axis=1)
y_stroke = df_stroke['stroke']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_stroke_imputed = pd.DataFrame(imputer.fit_transform(X_stroke), columns=X_stroke.columns)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_stroke_imputed, y_stroke)
df_stroke_smote = pd.concat([X_smote, y_smote], axis=1)

st.write("Balanced Stroke Dataset shape:", df_stroke_smote.shape)
st.write(df_stroke_smote['stroke'].value_counts())

# -----------------------
# Balance Diabetes Dataset with Undersampling
# -----------------------
st.header("3️⃣ Balance Diabetes Dataset (Undersampling)")

df_majority = df_diabetes[df_diabetes.diabetes == 0]
df_minority = df_diabetes[df_diabetes.diabetes == 1]

df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=len(df_minority),
                                   random_state=42)

df_balanced_diabetes = pd.concat([df_majority_downsampled, df_minority])
df_balanced_diabetes = df_balanced_diabetes.sample(frac=1, random_state=42).reset_index(drop=True)

st.write("Balanced Diabetes Dataset shape:", df_balanced_diabetes.shape)
st.write(df_balanced_diabetes['diabetes'].value_counts())

# -----------------------
# Correlation Heatmaps
# -----------------------
st.header("4️⃣ Correlation Heatmaps")

st.subheader("Stroke Dataset (SMOTE)")
fig1, ax1 = plt.subplots(figsize=(12,10))
sns.heatmap(df_stroke_smote.corr(numeric_only=True), annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, fmt=".2f", linewidths=0.5, ax=ax1)
ax1.set_title("Correlation Heatmap - Stroke Dataset (SMOTE)", fontsize=16, pad=15)
st.pyplot(fig1)

st.subheader("Diabetes Dataset (Balanced)")
fig2, ax2 = plt.subplots(figsize=(10,8))
sns.heatmap(df_balanced_diabetes.corr(numeric_only=True), annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, fmt=".2f", linewidths=0.5, ax=ax2)
ax2.set_title("Correlation Heatmap - Diabetes Dataset", fontsize=16, pad=15)
st.pyplot(fig2)

# -----------------------
# Feature Importance with Random Forest
# -----------------------
st.header("5️⃣ Feature Importance (Random Forest)")

st.subheader("Stroke Dataset")
rf_stroke = RandomForestClassifier(n_estimators=200, random_state=42)
rf_stroke.fit(X_stroke, y_stroke)
feat_imp_stroke = pd.DataFrame({
    'Feature': X_stroke.columns,
    'Importance': rf_stroke.feature_importances_
}).sort_values(by='Importance', ascending=False)

fig3, ax3 = plt.subplots(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_stroke, palette='viridis', ax=ax3)
ax3.set_title("Stroke Dataset - Feature Importance (Random Forest)")
st.pyplot(fig3)

st.subheader("Diabetes Dataset")
X_diabetes = df_diabetes.drop('diabetes', axis=1)
y_diabetes = df_diabetes['diabetes']
for col in ['gender','smoking_history']:
    X_diabetes[col] = label_encoder.fit_transform(X_diabetes[col])

rf_diabetes = RandomForestClassifier(n_estimators=200, random_state=42)
rf_diabetes.fit(X_diabetes, y_diabetes)
feat_imp_diabetes = pd.DataFrame({
    'Feature': X_diabetes.columns,
    'Importance': rf_diabetes.feature_importances_
}).sort_values(by='Importance', ascending=False)

fig4, ax4 = plt.subplots(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_diabetes, palette='magma', ax=ax4)
ax4.set_title("Diabetes Dataset - Feature Importance (Random Forest)")
st.pyplot(fig4)
