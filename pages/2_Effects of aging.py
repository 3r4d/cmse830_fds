import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import altair as alt
from sklearn.linear_model import LogisticRegression


# -----------------------
# Load Datasets
# -----------------------
@st.cache_data
def load_datasets(stroke_path, diabetes_path, heart_path):
    df_stroke = pd.read_csv(stroke_path)
    df_stroke = df_stroke.drop(columns=['id', 'ever_married', 'work_type', 'Residence_type'])
    df_stroke = df_stroke[df_stroke['gender'] != 'Other']

    df_diabetes = pd.read_csv(diabetes_path)
    new_order = ['gender', 'age', 'hypertension', 'heart_disease', 'blood_glucose_level',
                 'bmi', 'smoking_history', 'HbA1c_level', 'diabetes']
    df_diabetes = df_diabetes[new_order]

    df_heart = pd.read_csv(heart_path)
    df_heart = df_heart.drop(columns=['PhysActivity', 'Fruits', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
                            'DiffWalk', 'Education', 'Income', 'Veggies', 'HvyAlcoholConsump', 'CholCheck'])
    df_heart_new_order = ['Sex', 'Age', 'HighBP', 'HeartDiseaseorAttack', 'BMI', 'Smoker', 'HighChol', 'Diabetes', 'Stroke']
    df_heart = df_heart[df_heart_new_order]


    return df_stroke, df_diabetes, df_heart

stroke_path = "data/healthcare-dataset-stroke-data.csv"
diabetes_path = "data/diabetes_prediction_dataset.csv"
heart_path = "data/heart_disease_health_indicators_BRFSS2015.csv"

df_stroke, df_diabetes, df_heart = load_datasets(stroke_path, diabetes_path, heart_path)


# -----------------------
# Balance Stroke Dataset with SMOTE
# -----------------------
@st.cache_data
def balance_stroke(df):
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    if 'smoking_status' in df.columns:
        df['smoking_status'] = le.fit_transform(df['smoking_status'])

    X = df.drop('stroke', axis=1)
    y = df['stroke']
    X_imputed = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)

    X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_imputed, y)
    df_bal = pd.concat([X_smote, y_smote], axis=1)
    return df_bal

df_stroke_smote = balance_stroke(df_stroke.copy()) # Use a copy to prevent SettingWithCopyWarning


# -----------------------
# Balance Diabetes Dataset with Undersampling
# -----------------------
@st.cache_data
def balance_diabetes(df):
    df_majority = df[df.diabetes == 0]
    df_minority = df[df.diabetes == 1]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=len(df_minority),
                                       random_state=42)
    df_bal = pd.concat([df_majority_downsampled, df_minority])
    df_bal = df_bal.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_bal

df_balanced_diabetes = balance_diabetes(df_diabetes.copy()) # Use a copy


# -----------------------
# Balance Heart Dataset with SMOTE (Using a copy of the dataframe)
# -----------------------
@st.cache_data
def balance_heart(df):
    df_heart_copy = df.copy()
    label_encoder = LabelEncoder()
    df_heart_copy['Sex'] = label_encoder.fit_transform(df_heart_copy['Sex'])

    if 'Smoker' in df_heart_copy.columns:
        # Note: assuming 'Smoker' column contains strings that need encoding if it wasn't numeric already
        # Based on the BRFSS data, this column is usually 0/1 binary, so this encoding might be redundant if the CSV already used 0/1.
        # But we'll keep it for robustness, assuming the input CSV for heart disease is mixed string/int.
        df_heart_copy['Smoker'] = pd.to_numeric(df_heart_copy['Smoker'], errors='coerce')
        df_heart_copy['Smoker'] = df_heart_copy['Smoker'].fillna(df_heart_copy['Smoker'].mode()[0])

    # Ensure Age is numeric for imputation if needed, although it should be categorical codes (1-13)
    for col in ['HighBP', 'Smoker', 'HighChol', 'Diabetes', 'Stroke', 'Age', 'BMI']:
        df_heart_copy[col] = pd.to_numeric(df_heart_copy[col], errors='coerce')


    # --- Separate features and target ---
    X3 = df_heart_copy.drop('HeartDiseaseorAttack', axis=1)
    y3 = df_heart_copy['HeartDiseaseorAttack']

    # --- Handle missing values ---
    imputer3 = SimpleImputer(strategy='mean')
    X_imputed3 = pd.DataFrame(imputer3.fit_transform(X3), columns=X3.columns)

    # --- Apply SMOTE ---
    smote = SMOTE(random_state=42)
    X_smote3, y_smote3 = smote.fit_resample(X_imputed3, y3)

    # --- Combine back into a single balanced DataFrame ---
    df_bal = pd.concat([X_smote3, y_smote3], axis=1)
    return df_bal

df_balanced_heart = balance_heart(df_heart.copy())


# -----------------------
# Cached function for correlation heatmaps
# -----------------------
@st.cache_data
def plot_correlation_heatmap(df, title, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title(title, fontsize=16, pad=15)
    st.pyplot(fig)


# -----------------------
# NEW: Age Code to Midpoint Age Mapping
# -----------------------
AGE_CODE_TO_MIDPOINT = {
    1: 21,  # 18-24
    2: 27,  # 25-29
    3: 32,  # 30-34
    4: 37,  # 35-39
    5: 42,  # 40-44
    6: 47,  # 45-49
    7: 52,  # 50-54
    8: 57,  # 55-59
    9: 62,  # 60-64
    10: 67, # 65-69
    11: 72, # 70-74
    12: 77, # 75-79
    13: 82  # 80+ (Using 82 as a rough midpoint)
}



st.title("👵Effects of aging 👴")
st.write("Let's start by taking a look at the relationship between age and the likelihood you'll develop stroke, diabetes, or heart disease.")
# -----------------------
# Prepare age vs mean stroke probability
# -----------------------
@st.cache_data
def prepare_age_stroke_data(df):
    # Group by age and calculate mean stroke probability
    age_prob = df.groupby('age')['stroke'].mean().reset_index()
    age_prob.rename(columns={'stroke': 'mean_stroke_probability'}, inplace=True)
    return age_prob

age_stroke_df = prepare_age_stroke_data(df_stroke_smote)

# -----------------------
# Streamlit interactive plot - Stroke
# -----------------------

# Fit logistic regression for stroke
X = df_stroke['age'].values.reshape(-1, 1)
y = df_stroke['stroke'].values
model = LogisticRegression()
model.fit(X, y)

# Generate age range and predicted probabilities
age_range = np.arange(df_stroke['age'].min(), df_stroke['age'].max())
prob = model.predict_proba(age_range.reshape(-1,1))[:,1]

# Create DataFrame for Altair
age_prob_df = pd.DataFrame({
    'age': age_range,
    'predicted_stroke_probability': prob
})

# Altair interactive plot
chart = alt.Chart(age_prob_df).mark_line(point=True).encode(
    x=alt.X('age', title='Age'),
    y=alt.Y('predicted_stroke_probability', title='Predicted Stroke Probability'),
    tooltip=['age', 'predicted_stroke_probability']
).properties(
    title='Predicted Probability of Stroke risk overtime'
).interactive()

st.altair_chart(chart, use_container_width=True)


#Diabetes plot
# Fit logistic regression for diabetes
X_diabetes = df_diabetes['age'].values.reshape(-1, 1)
y_diabetes = df_diabetes['diabetes'].values
model_diabetes = LogisticRegression()
model_diabetes.fit(X_diabetes, y_diabetes)

# Generate age range and predicted probabilities
age_range_diabetes = np.arange(df_diabetes['age'].min(), df_diabetes['age'].max())
prob_diabetes = model_diabetes.predict_proba(age_range_diabetes.reshape(-1,1))[:,1]

# Create DataFrame for Altair
age_prob_diabetes_df = pd.DataFrame({
    'age': age_range_diabetes,
    'predicted_diabetes_probability': prob_diabetes
})

# Altair interactive plot
chart_diabetes = alt.Chart(age_prob_diabetes_df).mark_line(point=True).encode(
    x=alt.X('age', title='Age'),
    y=alt.Y('predicted_diabetes_probability', title='Predicted Diabetes Probability'),
    tooltip=['age', 'predicted_diabetes_probability']
).properties(
    title='Predicted Probability of risk of Diabetes overtime'
).interactive()

st.altair_chart(chart_diabetes, use_container_width=True)


#------------------
#Heart disease plot (MODIFIED TO USE MIDPOINT AGE)
#------------------
# Fit logistic regression for heart disease
X_heart = df_heart['Age'].values.reshape(-1, 1)
y_heart = df_heart['HeartDiseaseorAttack'].values
model_heart = LogisticRegression()
model_heart.fit(X_heart, y_heart)

# Generate age range and predicted probabilities
# Use +1 to include the max code (13)
age_code_range = np.arange(df_heart['Age'].min(), df_heart['Age'].max() + 1)
prob_heart = model_heart.predict_proba(age_code_range.reshape(-1,1))[:,1]

# Create DataFrame for Altair
age_prob_heart_df = pd.DataFrame({
    'Age Code': age_code_range,
    'Predicted Probability': prob_heart
})

# CRITICAL STEP: Map the Age Code to the Midpoint Age for plotting
age_prob_heart_df['Age'] = age_prob_heart_df['Age Code'].map(AGE_CODE_TO_MIDPOINT)


# Altair interactive plot
chart_heart = alt.Chart(age_prob_heart_df).mark_line(point=True).encode(
    # Use the new 'Age' column (midpoint age) for the X-axis
    x=alt.X('Age', title='Age (Midpoint of Group)'),
    y=alt.Y('Predicted Probability', title='Predicted Heart Disease Probability'),
    # Use the original Age Code for better tooltip context
    tooltip=[alt.Tooltip('Age', title='Midpoint Age'), alt.Tooltip('Age Code', title='BRFSS Code'), 'Predicted Probability']
).properties(
    title='Predicted Probability of Heart Disease by Age Group Midpoint'
).interactive()

st.altair_chart(chart_heart, use_container_width=True)


st.write("Now that we can visualize the risk overtime, hop around the app and discover some ways in which we can control our probability.")


