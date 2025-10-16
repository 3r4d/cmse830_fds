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


# ------------------------------------------------------------
# Assume you already have the following loaded in memory:
# df_stroke_smote  -> balanced stroke dataset
# df_balanced_diabetes -> balanced diabetes dataset
# ------------------------------------------------------------

# Original Stroke dataset
stroke_path = "../Data/healthcare-dataset-stroke-data.csv"
df_stroke = pd.read_csv(stroke_path)
df_stroke = df_stroke.drop(columns=['id', 'ever_married', 'work_type', 'Residence_type'])
df_stroke = df_stroke[df_stroke['gender'] != 'Other']


# Original Diabetes dataset
diabetes_path = "../Data/diabetes_prediction_dataset.csv"
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


# -----------------------
# Balance Diabetes Dataset with Undersampling
# -----------------------


df_majority = df_diabetes[df_diabetes.diabetes == 0]
df_minority = df_diabetes[df_diabetes.diabetes == 1]

df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=len(df_minority),
                                   random_state=42)

df_balanced_diabetes = pd.concat([df_majority_downsampled, df_minority])
df_balanced_diabetes = df_balanced_diabetes.sample(frac=1, random_state=42).reset_index(drop=True)



# Encode any non-numeric columns (safeguard)
for df in [df_stroke_smote, df_balanced_diabetes]:
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

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
# Streamlit UI
# ------------------------------------------------------------
st.title("🧠 Health Risk Predictor (Calibrated Model)")
st.write("Interactively explore how age, BMI, and health factors influence your probability of **stroke** or **diabetes**.")

# Choose model type
model_choice = st.radio("Select which condition to predict:", ["Stroke", "Diabetes"])

# Common sliders
age = st.slider("Age", 0, 100, 45)
bmi = st.slider("BMI", 10.0, 60.0, 25.0)
hypertension = st.checkbox("Hypertension (High Blood Pressure)", value=False)
heart_disease = st.checkbox("Heart Disease", value=False)

if model_choice == "Stroke":
    glucose = st.slider("Average Glucose Level", 50.0, 300.0, 100.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    # Prepare input
    input_data = pd.DataFrame({
        'gender': [1],  # assume female=1 for now
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

    st.metric("Predicted Stroke Probability", f"{prob_realistic*100:.2f}%")

else:
    glucose = st.slider("Blood Glucose Level", 50.0, 300.0, 100.0)
    hba1c = st.slider("HbA1c Level", 3.0, 14.0, 5.5)
    smoking = st.selectbox("Smoking History", ["never", "former", "current", "no info"])

    # Prepare input
    input_data = pd.DataFrame({
        'gender': [1],
        'age': [age],
        'hypertension': [int(hypertension)],
        'heart_disease': [int(heart_disease)],
        'blood_glucose_level': [glucose],
        'bmi': [bmi],
        'smoking_history': [0 if smoking == "never"
                            else 1 if smoking == "former"
                            else 2 if smoking == "current"
                            else 3],
        'HbA1c_level': [hba1c]
    })

    input_scaled = scaler_d.transform(input_data)
    prob = diab_model.predict_proba(input_scaled)[:, 1][0]

    # Re-anchor to real-world prevalence (~10%)
    real_prevalence = 0.10
    scaling_factor = real_prevalence / np.mean(y_prob_cal_d)
    prob_realistic = min(prob * scaling_factor, 1.0)

    st.metric("Predicted Diabetes Probability", f"{prob_realistic*100:.2f}%")
