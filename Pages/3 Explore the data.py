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



# -----------------------
# Load Datasets
# -----------------------
@st.cache_data
def load_datasets(stroke_path, diabetes_path):
    df_stroke = pd.read_csv(stroke_path)
    df_stroke = df_stroke.drop(columns=['id', 'ever_married', 'work_type', 'Residence_type'])
    df_stroke = df_stroke[df_stroke['gender'] != 'Other']

    df_diabetes = pd.read_csv(diabetes_path)
    new_order = ['gender', 'age', 'hypertension', 'heart_disease', 'blood_glucose_level',
                 'bmi', 'smoking_history', 'HbA1c_level', 'diabetes']
    df_diabetes = df_diabetes[new_order]

    return df_stroke, df_diabetes

stroke_path = "C:/Users/brad_/MSU/Year 1/CMSE 830/Project/healthcare-dataset-stroke-data.csv"
diabetes_path = "C:/Users/brad_/MSU/Year 1/CMSE 830/Project/diabetes_prediction_dataset.csv"

df_stroke, df_diabetes = load_datasets(stroke_path, diabetes_path)

st.write("now that we know that overlaps exist, lets take a look at what factors influence these disease the most.")
st.write("Below are two charts showing the most important factors found in our data that influence the likelihood of having a stroke or developing diabetes.")
st.write("According to these charts the factors are:")
with st.expander("Stroke"):
    st.write("age, smoking status, blood glucose (blood sugar), and hypertension (high blood pressure)")

with st.expander("Diabetes"):
    st.write("A1c, blood glucose (blood sugar), age, and BMI")


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
# Cached function for Random Forest Feature Importance
# -----------------------
@st.cache_data
def train_rf_feature_importance(X, y, n_estimators=200, random_state=42):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X, y)
    feat_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    return feat_imp_df, rf_model

@st.cache_data
def plot_feature_importance(feat_imp_df, title="Feature Importance", palette="magma", figsize=(10,6)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette=palette, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# -----------------------
# Stroke Random Forest
# -----------------------
st.header("Risk factors for stroke and diabetes")

st.subheader("Stroke Dataset")
X_stroke = df_stroke_smote.drop('stroke', axis=1)
y_stroke = df_stroke_smote['stroke']
feat_imp_stroke, rf_stroke = train_rf_feature_importance(X_stroke, y_stroke)
plot_feature_importance(feat_imp_stroke, "Stroke Dataset - Feature Importance (Random Forest)", palette='viridis')

# -----------------------
# Diabetes Random Forest
# -----------------------
st.subheader("Diabetes Dataset")
X_diabetes = df_balanced_diabetes.drop('diabetes', axis=1)
y_diabetes = df_balanced_diabetes['diabetes']
# Encode categorical columns
for col in ['gender','smoking_history']:
    X_diabetes[col] = LabelEncoder().fit_transform(X_diabetes[col])

feat_imp_diabetes, rf_diabetes = train_rf_feature_importance(X_diabetes, y_diabetes)
plot_feature_importance(feat_imp_diabetes, "Diabetes Dataset - Feature Importance (Random Forest)")
