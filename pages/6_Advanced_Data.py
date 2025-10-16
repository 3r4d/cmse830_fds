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


stroke_path = "data/healthcare-dataset-stroke-data.csv"
diabetes_path = "data/diabetes_prediction_dataset.csv"

df_stroke, df_diabetes = load_datasets(stroke_path, diabetes_path)
df_stroke1 = df_stroke.copy()



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

#==================================================
#Class imbalance drop down
#==================================================

with st.expander("Class imbalance"):
    st.write("When working with the original datasets you'll notice that the amount of patients with diabetes/stroke is HEAVILY outnumbered by those who do not have diabetes/stroke.")
    st.subheader("Stroke Dataset")
    st.dataframe(df_stroke1.head())
    st.write("Here we see stroke data that looks at factors such as: age, high blood pressure(hypertension), heart disease, average blood glucose level, BMI, smoking history, and whether or not the person has had a stroke before.")
    st.write("But if we take a look at the number of patients with stroke or diabetes vs those with, there is a big descrepency. See below:")
    st.write("The number of people with stroke is: " + str(sum(df_stroke1['stroke'])) + ". Compare that to the total number of people: " + str(len(df_stroke1['stroke'])) + ".")
    st.write("This difference is called a bias. In order to obtain accurate data we need to make the total number of positives the same as the negatives. The 'SMOTE' technique was applied in order to eliminate the gap in the positive and negative groups.")
    st.write("Balanced Stroke Dataset:")
    st.write(df_stroke_smote['stroke'].value_counts())



    st.subheader("Diabetes Dataset")
    st.dataframe(df_diabetes.head())
    st.write(
        "Here we see data for diabetes that looks at similar factors as the stroke dataset with a couple extras: age, high blood pressure(hypertension), heart disease, average blood glucose level, BMI, smoking history, and A1c levels.")

    st.write("We run into the same issue here as the stroke dataset however. The imbalance of those who have diabetes is significantly less than those who do.")
    st.write("The number of people with diabetes is: " + str(sum(df_diabetes['diabetes'])) + ". Compare that to the total number of people: " + str(len(df_diabetes['diabetes'])) + ".")
    st.write("For the diabetes dataset a resampling technique was used. By randomly selecting an equal number of representatives from each category we can define a new dataset based off the original. This helps us obtain equal representation from each group")
    st.write("Balanced Diabetes Dataset shape:")
    st.write(df_balanced_diabetes['diabetes'].value_counts())



# -----------------------
# Cached function for correlation heatmaps
# -----------------------

@st.cache_data
def plot_correlation_heatmap(df, title, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title(title, fontsize=16, pad=15)
    st.pyplot(fig)

#==================================================
#Heat map issues drop down
#==================================================
with st.expander("Visualization issues"):
    st.write("One common practice when dealing with data analysis is the visualization of data. Heat maps are a common practice, showing patterns and trend between several different variables within the dataset.")
    st.write("Sometimes with real-world data the correlations can be opposite of what we know to be true however. Take note of the relationship between stroke and BMI in the stroke dataset.")
    st.write("The value below is 0.05 which is almost no correlation at all, but according to several studies (such as this one: https://pubmed.ncbi.nlm.nih.gov/35971008/) BMI and stroke are heavily correlated and even have a causal relationship.")

    with st.expander("Stoke Heat Map (balanced data)"):
        st.header("Correlation Heatmaps")
        st.subheader("Stroke Dataset (SMOTE)")
        plot_correlation_heatmap(df_stroke_smote, "Correlation Heatmap - Stroke Dataset (SMOTE)", figsize=(12, 10))
    st.write("Is this a fluke? Let's check on the diabetes dataset and see if this pops up again or not.")
    with st.expander("Diabetes Dataset (balanced data)"):
        st.subheader("Diabetes Dataset (Balanced)")
        plot_correlation_heatmap(df_balanced_diabetes, "Correlation Heatmap - Diabetes Dataset", figsize=(10, 8))
        st.write(
    "When taking a look at the diabetes dataset the correlations seem a little more realistic, but there are some subtle issues as well.")
    st.write(
    "Notice the correlation between high blood pressure and heart disease? It's only 0.10 indicating very little correlation. But according to several studies (one such study: https://pmc.ncbi.nlm.nih.gov/articles/PMC10243231/) hypertension and heart disease are heavily correlated and have a causal relationship. While this is not the main focus of the diabetes dataset, we can see there is another instance in which data correlations are not seeming to make sense.")
    st.write("So what could be going on? Let's take a look and compare the original data with the synthetic to see if maybe the synthetic data has anything to do with it or not.")
    with st.expander("Stroke heat map (original data vs SMOTE)"):
        st.subheader("Stroke Dataset (original)")
        plot_correlation_heatmap(df_stroke1, "Correlation Heatmap - Stroke Dataset (Original data)", figsize=(10, 8))
        st.subheader("Stroke Dataset (SMOTE)")
        plot_correlation_heatmap(df_stroke_smote, "Correlation Heatmap - Stroke Dataset (SMOTE)", figsize=(12, 10))
    st.write("It's almost harder to identify correlation with the original (this shows cleaning data works!).")
    st.write(
        "But what's really happening is the heat map is a nice tool to identify linear relationships with not a lot of noise. Unfortunately real world data is filled with noise and often not linear.")
    st.write("This is why random forest was utilized.")


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
def plot_feature_importance(feat_imp_df, title="Feature Importance", palette="magma", figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette=palette, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)


# # -----------------------
# # Stroke Random Forest
# # -----------------------
# st.header("Feature Importance (Random Forest)")
#
# st.subheader("Stroke Dataset")
# X_stroke = df_stroke_smote.drop('stroke', axis=1)
# y_stroke = df_stroke_smote['stroke']
# feat_imp_stroke, rf_stroke = train_rf_feature_importance(X_stroke, y_stroke)
# plot_feature_importance(feat_imp_stroke, "Stroke Dataset - Feature Importance (Random Forest)", palette='viridis')
#
# # -----------------------
# # Diabetes Random Forest
# # -----------------------
# st.subheader("Diabetes Dataset")
# X_diabetes = df_balanced_diabetes.drop('diabetes', axis=1)
# y_diabetes = df_balanced_diabetes['diabetes']
# # Encode categorical columns
# for col in ['gender', 'smoking_history']:
#     X_diabetes[col] = LabelEncoder().fit_transform(X_diabetes[col])
#
# feat_imp_diabetes, rf_diabetes = train_rf_feature_importance(X_diabetes, y_diabetes)
# plot_feature_importance(feat_imp_diabetes, "Diabetes Dataset - Feature Importance (Random Forest)")
