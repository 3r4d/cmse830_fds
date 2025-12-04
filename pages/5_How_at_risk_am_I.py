import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.metrics import brier_score_loss # Not needed for prediction
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample


# Use st.cache_data for dataset loading and preparation
@st.cache_data
def load_and_prepare_data(stroke_path, diabetes_path, heart_path):
    # --- STROKE DATA PREP ---
    df_stroke = pd.read_csv(stroke_path)
    df_stroke = df_stroke.drop(columns=['id', 'ever_married', 'work_type', 'Residence_type'])
    df_stroke = df_stroke[df_stroke['gender'] != 'Other']
    label_encoder = LabelEncoder()
    # Inspection of stroke data: 'Female' is encoded as 0, 'Male' is encoded as 1.
    df_stroke['gender'] = label_encoder.fit_transform(df_stroke['gender'])
    if 'smoking_status' in df_stroke.columns:
        df_stroke['smoking_status'] = df_stroke['smoking_status'].fillna('Unknown')
        df_stroke['smoking_status'] = label_encoder.fit_transform(df_stroke['smoking_status'])

    X_stroke = df_stroke.drop('stroke', axis=1)
    y_stroke = df_stroke['stroke']
    imputer = SimpleImputer(strategy='mean')
    X_stroke_imputed = pd.DataFrame(imputer.fit_transform(X_stroke), columns=X_stroke.columns)
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_stroke_imputed, y_stroke)
    df_stroke_smote = pd.concat([X_smote, y_smote], axis=1)

    # --- DIABETES DATA PREP ---
    df_diabetes = pd.read_csv(diabetes_path)
    new_order = ['gender', 'age', 'hypertension', 'heart_disease', 'blood_glucose_level',
                 'bmi', 'smoking_history', 'HbA1c_level', 'diabetes']
    df_diabetes = df_diabetes[new_order]
    le = LabelEncoder()
    # Inspection of diabetes data: 'Female' is encoded as 0, 'Male' is encoded as 1.
    df_diabetes['gender'] = le.fit_transform(df_diabetes['gender'])
    df_diabetes['smoking_history'] = le.fit_transform(df_diabetes['smoking_history'])
    df_majority = df_diabetes[df_diabetes.diabetes == 0]
    df_minority = df_diabetes[df_diabetes.diabetes == 1]
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
    df_balanced_diabetes = pd.concat([df_majority_downsampled, df_minority])
    df_balanced_diabetes = df_balanced_diabetes.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- HEART DISEASE DATA PREP ---
    df_heart = pd.read_csv(heart_path)
    df_heart = df_heart.drop(
        columns=['PhysActivity', 'Fruits', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
                 'DiffWalk', 'Education', 'Income', 'Veggies', 'HvyAlcoholConsump', 'CholCheck'])
    df_heart_new_order = ['Sex', 'Age', 'HighBP', 'HeartDiseaseorAttack', 'BMI', 'Smoker', 'HighChol', 'Diabetes',
                          'Stroke']
    df_heart = df_heart[df_heart_new_order]

    df_copy = df_heart.copy()
    le = LabelEncoder()
    df_copy['Sex'] = le.fit_transform(df_copy['Sex'])

    # NOTE: The Age column in this dataset uses categorical codes (1=18-24, ..., 13=80+).
    # It must be converted to numeric here for the StandardScaler to work.
    for col in ['HighBP', 'Smoker', 'HighChol', 'Diabetes', 'Stroke', 'Age', 'BMI']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    X3 = df_copy.drop('HeartDiseaseorAttack', axis=1)
    y3 = df_copy['HeartDiseaseorAttack']
    imputer3 = SimpleImputer(strategy='mean')
    X_imputed3 = pd.DataFrame(imputer3.fit_transform(X3), columns=X3.columns)
    smote = SMOTE(random_state=42)
    X_smote3, y_smote3 = smote.fit_resample(X_imputed3, y3)
    df_balanced_heart = pd.concat([X_smote3, y_smote3], axis=1)

    return df_stroke_smote, df_balanced_diabetes, df_balanced_heart


# Define paths
stroke_path = "data/healthcare-dataset-stroke-data.csv"
diabetes_path = "data/diabetes_prediction_dataset.csv"
heart_path = "data/heart_disease_health_indicators_BRFSS2015.csv"

# Load dataframes once and store them
df_stroke_smote, df_balanced_diabetes, df_balanced_heart = load_and_prepare_data(stroke_path, diabetes_path, heart_path)


@st.cache_resource
def train_and_calibrate_models(df_stroke, df_diabetes, df_heart):
    models = {}
    scalers = {}
    y_probs = {}

    # --- STROKE MODEL ---
    X_stroke = df_stroke.drop('stroke', axis=1)
    y_stroke = df_stroke['stroke']
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_stroke, y_stroke, test_size=0.2, random_state=42)
    scaler_s = StandardScaler()
    X_train_s_scaled = scaler_s.fit_transform(X_train_s)
    X_test_s_scaled = scaler_s.transform(X_test_s)
    stroke_model_raw = LogisticRegression(max_iter=1000, random_state=42)
    stroke_model = CalibratedClassifierCV(stroke_model_raw, cv=5)
    stroke_model.fit(X_train_s_scaled, y_train_s)
    y_prob_cal_s = stroke_model.predict_proba(X_test_s_scaled)[:, 1]

    models['stroke'] = stroke_model
    scalers['stroke'] = scaler_s
    y_probs['stroke'] = y_prob_cal_s

    # --- DIABETES MODEL ---
    X_diab = df_diabetes.drop('diabetes', axis=1)
    y_diab = df_diabetes['diabetes']
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diab, y_diab, test_size=0.2, random_state=42)
    scaler_d = StandardScaler()
    X_train_d_scaled = scaler_d.fit_transform(X_train_d)
    X_test_d_scaled = scaler_d.transform(X_test_d)
    diab_model_raw = LogisticRegression(max_iter=1000, random_state=42)
    diab_model = CalibratedClassifierCV(diab_model_raw, cv=5)
    diab_model.fit(X_train_d_scaled, y_train_d)
    y_prob_cal_d = diab_model.predict_proba(X_test_d_scaled)[:, 1]

    models['diabetes'] = diab_model
    scalers['diabetes'] = scaler_d
    y_probs['diabetes'] = y_prob_cal_d

    # --- HEART DISEASE MODEL ---
    X_heart = df_heart.drop('HeartDiseaseorAttack', axis=1)
    y_heart = df_heart['HeartDiseaseorAttack']
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
    scaler_h = StandardScaler()
    X_train_h_scaled = scaler_h.fit_transform(X_train_h)
    X_test_h_scaled = scaler_h.transform(X_test_h)
    heart_model_raw = LogisticRegression(max_iter=1000, random_state=42)
    heart_model = CalibratedClassifierCV(heart_model_raw, cv=5)
    heart_model.fit(X_train_h_scaled, y_train_h)
    y_prob_cal_h = heart_model.predict_proba(X_test_h_scaled)[:, 1]

    models['heart'] = heart_model
    scalers['heart'] = scaler_h
    y_probs['heart'] = y_prob_cal_h

    return models, scalers, y_probs


# Execute the caching function once on startup
models, scalers, y_probs = train_and_calibrate_models(df_stroke_smote, df_balanced_diabetes, df_balanced_heart)

# Extract cached models and variables for use in the UI section
stroke_model = models['stroke']
scaler_s = scalers['stroke']
y_prob_cal_s = y_probs['stroke']

diab_model = models['diabetes']
scaler_d = scalers['diabetes']
y_prob_cal_d = y_probs['diabetes']

heart_model = models['heart']
scaler_h = scalers['heart']
y_prob_cal_h = y_probs['heart']


# --- NEW FUNCTION FOR AGE MAPPING ---
def map_age_to_brfss_code(age):
    """Maps continuous age (18-100) to the 13 categorical codes used in the BRFSS 2015 dataset."""
    if age < 18:
        return 1  # Technically outside range, but safe default
    elif age <= 24:
        return 1
    elif age <= 29:
        return 2
    elif age <= 34:
        return 3
    elif age <= 39:
        return 4
    elif age <= 44:
        return 5
    elif age <= 49:
        return 6
    elif age <= 54:
        return 7
    elif age <= 59:
        return 8
    elif age <= 64:
        return 9
    elif age <= 69:
        return 10
    elif age <= 74:
        return 11
    elif age <= 79:
        return 12
    else:  # age >= 80
        return 13


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("🧠 Health Risk Predictor (Calibrated Model)")
st.write(
    "Interactively explore how age, BMI, and health factors influence your probability of **stroke**, **diabetes**, or **heart disease**.")

# Choose model type
model_choice = st.radio("Select which condition to predict:", ["Stroke", "Diabetes", "Heart Disease"])

# Common UI inputs
sex_choice = st.selectbox("Sex", ["Female", "Male"])
sex_input_encoded = 0 if sex_choice == "Female" else 1

age = st.slider("Age", 0, 100, 45)
bmi = st.slider("BMI", 10.0, 60.0, 25.0)
hypertension = st.checkbox("Hypertension (High Blood Pressure)", value=False)

if model_choice == "Stroke":
    glucose = st.slider("Average Glucose Level", 50.0, 300.0, 100.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])
    heart_disease = st.checkbox("Heart Disease", value=False)

    # Input encoding consistent with Stroke model training (uses continuous age)
    input_data = pd.DataFrame({
        'gender': [sex_input_encoded],
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

    # Input encoding consistent with Diabetes model training (uses continuous age)
    input_data = pd.DataFrame({
        'gender': [sex_input_encoded],
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

else:  # model_choice == "Heart Disease"
    # Heart Disease-specific inputs
    smoker = st.checkbox("Smoker", value=False)
    high_chol = st.checkbox("High Cholesterol", value=False)
    diabetes_history = st.checkbox("Diabetes (Self-Reported)", value=False)
    stroke_history = st.checkbox("History of Stroke", value=False)

    # ***CRITICAL FIX: MAP CONTINUOUS AGE TO BRFSS CATEGORICAL CODE***
    age_code = map_age_to_brfss_code(age)

    # Prepare input - MUST match df_balanced_heart features
    input_data = pd.DataFrame({
        'Sex': [sex_input_encoded],
        'Age': [age_code],  # Use the MAPPED categorical code
        'HighBP': [int(hypertension)],
        'BMI': [bmi],
        'Smoker': [int(smoker)],
        'HighChol': [int(high_chol)],
        'Diabetes': [int(diabetes_history)],
        'Stroke': [int(stroke_history)]
    })

    input_scaled_h = scaler_h.transform(input_data)
    prob_h = heart_model.predict_proba(input_scaled_h)[:, 1][0]

    # --- ADJUSTED SCALING FOR REALISM (Carried over from last fix) ---
    target_prevalence_for_scaling = 0.40
    scaling_factor_h = target_prevalence_for_scaling / np.mean(y_prob_cal_h)

    prob_realistic_h = min(prob_h * scaling_factor_h, 1.0)

    st.metric("Predicted Heart Disease Probability", f"{prob_realistic_h * 100:.2f}%")