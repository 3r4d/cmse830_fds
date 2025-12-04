import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# -----------------------
# Initial Data Loading and Setup (Run on every launch)
# -----------------------

# Original Stroke dataset
stroke_path = "data/healthcare-dataset-stroke-data.csv"
df_stroke = pd.read_csv(stroke_path)
df_stroke = df_stroke.drop(columns=['id', 'ever_married', 'work_type', 'Residence_type'])
df_stroke = df_stroke[df_stroke['gender'] != 'Other']

# Original Diabetes dataset
diabetes_path = "data/diabetes_prediction_dataset.csv"
new_order = ['gender', 'age', 'hypertension', 'heart_disease', 'blood_glucose_level',
             'bmi', 'smoking_history', 'HbA1c_level', 'diabetes']
df_diabetes = pd.read_csv(diabetes_path)
df_diabetes = df_diabetes[new_order]

# -----------------------
# Balance Stroke Dataset with SMOTE
# -----------------------
# Encode categorical variables
label_encoder = LabelEncoder()
df_stroke['gender'] = label_encoder.fit_transform(df_stroke['gender'])
# Handle missing smoking status (if any) or encode it
if 'smoking_status' in df_stroke.columns:
    df_stroke['smoking_status'] = df_stroke['smoking_status'].fillna('Unknown')
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

# -----------------------
# Balance Diabetes Dataset with Undersampling (FIXED)
# -----------------------
le = LabelEncoder()
df_diabetes['gender'] = le.fit_transform(df_diabetes['gender'])
df_diabetes['smoking_history'] = le.fit_transform(df_diabetes['smoking_history'])

df_majority = df_diabetes[df_diabetes.diabetes == 0]
df_minority = df_diabetes[df_diabetes.diabetes == 1]

df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=len(df_minority),
                                   random_state=42)

df_balanced_diabetes = pd.concat([df_majority_downsampled, df_minority])
df_balanced_diabetes = df_balanced_diabetes.sample(frac=1, random_state=42).reset_index(drop=True)


# -----------------------
# Load and Balance Heart Disease Dataset (NEW/HARDENED)
# -----------------------
@st.cache_data
def load_datasets(stroke_path, diabetes_path, heart_path):
    # ... (stroke and diabetes loading kept for completeness, though redefined globally)
    df_stroke = pd.read_csv(stroke_path)
    df_stroke = df_stroke.drop(columns=['id', 'ever_married', 'work_type', 'Residence_type'])
    df_stroke = df_stroke[df_stroke['gender'] != 'Other']

    df_diabetes = pd.read_csv(diabetes_path)
    new_order = ['gender', 'age', 'hypertension', 'heart_disease', 'blood_glucose_level',
                 'bmi', 'smoking_history', 'HbA1c_level', 'diabetes']
    df_diabetes = df_diabetes[new_order]

    df_heart = pd.read_csv(heart_path)
    # Drop features not used in the model
    df_heart = df_heart.drop(
        columns=['PhysActivity', 'Fruits', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
                 'DiffWalk', 'Education', 'Income', 'Veggies', 'HvyAlcoholConsump', 'CholCheck'])

    df_heart_new_order = ['Sex', 'Age', 'HighBP', 'HeartDiseaseorAttack', 'BMI', 'Smoker', 'HighChol', 'Diabetes',
                          'Stroke']
    df_heart = df_heart[df_heart_new_order]

    return df_stroke, df_diabetes, df_heart


stroke_path = "data/healthcare-dataset-stroke-data.csv"
diabetes_path = "data/diabetes_prediction_dataset.csv"
heart_path = "data/heart_disease_health_indicators_BRFSS2015.csv"

# Load dataframes again using the cached function for consistency
df_stroke_cached, df_diabetes_original, df_heart = load_datasets(stroke_path, diabetes_path, heart_path)


def balance_heart(df):
    df_copy = df.copy()
    le = LabelEncoder()

    # 1. Encode Sex
    df_copy['Sex'] = le.fit_transform(df_copy['Sex'])

    # 2. Ensure all other features are explicitly numeric (critical for StandardScaler)
    for col in ['HighBP', 'Smoker', 'HighChol', 'Diabetes', 'Stroke', 'Age', 'BMI']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')  # 'coerce' converts non-numeric values to NaN

    # --- Separate features and target ---
    X3 = df_copy.drop('HeartDiseaseorAttack', axis=1)
    y3 = df_copy['HeartDiseaseorAttack']

    # --- Handle missing values (for any NaNs introduced by 'coerce' or original data) ---
    imputer3 = SimpleImputer(strategy='mean')
    X_imputed3 = pd.DataFrame(imputer3.fit_transform(X3), columns=X3.columns)

    # --- Apply SMOTE ---
    smote = SMOTE(random_state=42)
    X_smote3, y_smote3 = smote.fit_resample(X_imputed3, y3)

    # --- Combine back into a single balanced DataFrame ---
    df_bal = pd.concat([X_smote3, y_smote3], axis=1)
    return df_bal


df_balanced_heart = balance_heart(df_heart)

# ------------------------------------------------------------
# Train Calibrated Stroke Model
# ------------------------------------------------------------
X_stroke = df_stroke_smote.drop('stroke', axis=1)
y_stroke = df_stroke_smote['stroke']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_stroke, y_stroke, test_size=0.2, random_state=42)

scaler_s = StandardScaler()
X_train_s_scaled = scaler_s.fit_transform(X_train_s)
X_test_s_scaled = scaler_s.transform(X_test_s)

stroke_model_raw = LogisticRegression(max_iter=1000, random_state=42)
stroke_model = CalibratedClassifierCV(stroke_model_raw, cv=5)
stroke_model.fit(X_train_s_scaled, y_train_s)

y_prob_uncal_s = stroke_model_raw.fit(X_train_s_scaled, y_train_s).predict_proba(X_test_s_scaled)[:, 1]
y_prob_cal_s = stroke_model.predict_proba(X_test_s_scaled)[:, 1]

print("Stroke Brier score (uncalibrated):", brier_score_loss(y_test_s, y_prob_uncal_s))
print("Stroke Brier score (calibrated):", brier_score_loss(y_test_s, y_prob_cal_s))

# ------------------------------------------------------------
# Train Calibrated Diabetes Model
# ------------------------------------------------------------
X_diab = df_balanced_diabetes.drop('diabetes', axis=1)
y_diab = df_balanced_diabetes['diabetes']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diab, y_diab, test_size=0.2, random_state=42)

scaler_d = StandardScaler()
X_train_d_scaled = scaler_d.fit_transform(X_train_d)
X_test_d_scaled = scaler_d.transform(X_test_d)

diab_model_raw = LogisticRegression(max_iter=1000, random_state=42)
diab_model = CalibratedClassifierCV(diab_model_raw, cv=5)
diab_model.fit(X_train_d_scaled, y_train_d)

y_prob_uncal_d = diab_model_raw.fit(X_train_d_scaled, y_train_d).predict_proba(X_test_d_scaled)[:, 1]
y_prob_cal_d = diab_model.predict_proba(X_test_d_scaled)[:, 1]

print("Diabetes Brier score (uncalibrated):", brier_score_loss(y_test_d, y_prob_uncal_d))
print("Diabetes Brier score (calibrated):", brier_score_loss(y_test_d, y_prob_cal_d))

# ------------------------------------------------------------
# Train Calibrated Heart Disease Model (FIXED)
# ------------------------------------------------------------
X_heart = df_balanced_heart.drop('HeartDiseaseorAttack', axis=1)
y_heart = df_balanced_heart['HeartDiseaseorAttack']

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

scaler_h = StandardScaler()
X_train_h_scaled = scaler_h.fit_transform(X_train_h)
X_test_h_scaled = scaler_h.transform(X_test_h)

heart_model_raw = LogisticRegression(max_iter=1000, random_state=42)
heart_model = CalibratedClassifierCV(heart_model_raw, cv=5)
heart_model.fit(X_train_h_scaled, y_train_h)

y_prob_uncal_h = heart_model_raw.fit(X_train_h_scaled, y_train_h).predict_proba(X_test_h_scaled)[:, 1]
y_prob_cal_h = heart_model.predict_proba(X_test_h_scaled)[:, 1]

print("Heart Disease Brier score (uncalibrated):", brier_score_loss(y_test_h, y_prob_uncal_h))
print("Heart Disease Brier score (calibrated):", brier_score_loss(y_test_h, y_prob_cal_h))

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("🧠 Health Risk Predictor (Calibrated Model)")
st.write(
    "Interactively explore how age, BMI, and health factors influence your probability of **stroke**, **diabetes**, or **heart disease**.")

# Choose model type
model_choice = st.radio("Select which condition to predict:", ["Stroke", "Diabetes", "Heart Disease"])

# Common sliders
age = st.slider("Age", 0, 100, 45)
bmi = st.slider("BMI", 10.0, 60.0, 25.0)
hypertension = st.checkbox("Hypertension (High Blood Pressure)", value=False)

if model_choice == "Stroke":
    glucose = st.slider("Average Glucose Level", 50.0, 300.0, 100.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])
    heart_disease = st.checkbox("Heart Disease", value=False)

    # Input encoding consistent with Stroke model training
    input_data = pd.DataFrame({
        'gender': [1],
        'age': [age],
        'hypertension': [int(hypertension)],
        'heart_disease': [int(heart_disease)],
        'avg_glucose_level': [glucose],
        'bmi': [bmi],
        'smoking_status': [0 if smoking_status == "never smoked"
                           else 1 if smoking_status == "formerly smoked"
        else 2 if smoking_status == "smokes"
        else 3]
    })

    input_scaled = scaler_s.transform(input_data)
    prob = stroke_model.predict_proba(input_scaled)[:, 1][0]

    # Re-anchor to real-world prevalence (~5%)
    real_prevalence = 0.05
    scaling_factor = real_prevalence / np.mean(y_prob_cal_s)
    prob_realistic = min(prob * scaling_factor, 1.0)

    st.metric("Predicted Stroke Probability", f"{prob_realistic * 100:.2f}%")

elif model_choice == "Diabetes":
    # Diabetes-specific inputs
    hba1c = st.slider("HbA1c Level", 3.0, 10.0, 5.7)
    glucose = st.slider("Blood Glucose Level", 50.0, 300.0, 100.0)
    smoking_history = st.selectbox("Smoking History", ["never", "former", "current", "not current", "ever", "no info"])
    heart_disease = st.checkbox("Heart Disease (Self-Reported)", value=False)  # Feature is in diabetes dataset

    # Input encoding consistent with Diabetes model training
    input_data = pd.DataFrame({
        'gender': [0],  # Assuming Female=0 for diabetes model
        'age': [age],
        'hypertension': [int(hypertension)],
        'heart_disease': [int(heart_disease)],
        'blood_glucose_level': [glucose],
        'bmi': [bmi],
        'smoking_history': [0 if smoking_history == "never"
                            else 1 if smoking_history == "former"
        else 2 if smoking_history == "current"
        else 3 if smoking_history == "not current"
        else 4 if smoking_history == "ever"
        else 5],
        'HbA1c_level': [hba1c]
    })

    input_scaled = scaler_d.transform(input_data)
    prob = diab_model.predict_proba(input_scaled)[:, 1][0]

    # Re-anchor to real-world prevalence (~8.5%)
    real_prevalence = 0.085
    scaling_factor = real_prevalence / np.mean(y_prob_cal_d)
    prob_realistic = min(prob * scaling_factor, 1.0)

    st.metric("Predicted Diabetes Probability", f"{prob_realistic * 100:.2f}%")

else:  # model_choice == "Heart Disease" (FIXED)
    # Heart Disease-specific inputs
    sex = st.selectbox("Sex", ["Female", "Male"])
    smoker = st.checkbox("Smoker", value=False)
    high_chol = st.checkbox("High Cholesterol", value=False)
    diabetes_history = st.checkbox("Diabetes (Self-Reported)", value=False)
    stroke_history = st.checkbox("History of Stroke", value=False)

    # Prepare input - MUST match df_balanced_heart features (Sex, Age, HighBP, BMI, Smoker, HighChol, Diabetes, Stroke)
    # Encoding for Sex: Female=0, Male=1
    input_data = pd.DataFrame({
        'Sex': [0 if sex == "Female" else 1],
        'Age': [age],
        'HighBP': [int(hypertension)],  # Using hypertension for HighBP
        'BMI': [bmi],
        'Smoker': [int(smoker)],
        'HighChol': [int(high_chol)],
        'Diabetes': [int(diabetes_history)],  # Using new diabetes_history variable
        'Stroke': [int(stroke_history)]
    })

    input_scaled_h = scaler_h.transform(input_data)
    prob_h = heart_model.predict_proba(input_scaled_h)[:, 1][0]

    # Re-anchor to real-world prevalence (~10%)
    real_prevalence_h = 0.10
    scaling_factor_h = real_prevalence_h / np.mean(y_prob_cal_h)
    prob_realistic_h = min(prob_h * scaling_factor_h, 1.0)

    st.metric("Predicted Heart Disease Probability", f"{prob_realistic_h * 100:.2f}%")