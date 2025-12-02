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

df_stroke_smote = balance_stroke(df_stroke)





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

df_balanced_diabetes = balance_diabetes(df_diabetes)

# -----------------------
# Balance Diabetes Dataset with Undersampling
# -----------------------
def balance_heart(df):
    label_encoder = LabelEncoder()
    df_heart['Sex'] = label_encoder.fit_transform(df_heart['Sex'])

    if 'smoker' in df_heart.columns:
        df_heart['smoker'] = label_encoder.fit_transform(df_heart['smoker'])

    # --- Separate features and target ---
    X3 = df_heart.drop('HeartDiseaseorAttack', axis=1)
    y3 = df_heart['HeartDiseaseorAttack']

    # --- Handle missing values ---
    # Use mean for numeric columns (you could also use median or mode)
    imputer3 = SimpleImputer(strategy='mean')
    X_imputed3 = pd.DataFrame(imputer3.fit_transform(X3), columns=X3.columns)

    # --- Apply SMOTE ---
    smote = SMOTE(random_state=42)
    X_smote3, y_smote3 = smote.fit_resample(X_imputed3, y3)

    # --- Combine back into a single balanced DataFrame ---
    df_heart_smote = pd.concat([X_smote3, y_smote3], axis=1)



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








st.title("👵Effects of aging 👴")
st.write("Let's start by taking a look at the relationship between your age and the likelihood you'll develop stroke/diabetes.")
# -----------------------
# Prepare age vs mean stroke probability
#help from chat GPT to convert original code into functions in order to utilize the cache_data
# -----------------------
@st.cache_data
def prepare_age_stroke_data(df):
    # Group by age and calculate mean stroke probability
    age_prob = df.groupby('age')['stroke'].mean().reset_index()
    age_prob.rename(columns={'stroke': 'mean_stroke_probability'}, inplace=True)
    return age_prob

age_stroke_df = prepare_age_stroke_data(df_stroke_smote)

# -----------------------
# Streamlit interactive plot
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
#Heart disease plot
#------------------
# Fit logistic regression for stroke
X_heart = df_heart['Age'].values.reshape(-1, 1)
y_heart = df_heart['HeartDiseaseorAttack'].values
model_heart = LogisticRegression()
model_heart.fit(X_heart, y_heart)

# Generate age range and predicted probabilities
age_range_heart = np.arange(df_heart['Age'].min(), df_heart['Age'].max())
prob_heart = model_heart.predict_proba(age_range_heart.reshape(-1,1))[:,1]

# Create DataFrame for Altair
age_prob_heart_df = pd.DataFrame({
    'Age': age_range_heart,
    'HeartDiseaseorAttack': prob_heart
})

# Altair interactive plot
chart_heart = alt.Chart(age_prob_heart_df).mark_line(point=True).encode(
    x=alt.X('Age', title='Age'),
    y=alt.Y('HeartDiseaseorAttack', title='Heart Disease or Heart Attack'),
    tooltip=['Age', 'HeartDiseaseorAttack']
).properties(
    title='Predicted Probability of Heart Disease or Heart Attack'
).interactive()

st.altair_chart(chart, use_container_width=True)



st.write("Now that we can visualize the risk overtime, hop around the app and discover some ways in which we can control our probability.")
