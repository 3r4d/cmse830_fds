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
st.set_page_config(page_title="Stroke, Diabetes, & Heart Disease Analysis", layout="wide")
st.title("Stroke & Diabetes Dataset Analysis")



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
#help from chatGPT to choose the best method of balancing with large datasets
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
#help from chatGPT to help choose best way to balance large datasets
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
# Balance Heart Dataset with Undersampling
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
    df_bal = pd.concat([X_smote3, y_smote3], axis=1)
    return df_bal

df_heart_smote = balance_heart(df_heart)

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



st.write("Now that we know that overlaps exist, lets take a look at what factors influence these diseases the most.")
st.write("Below are three charts showing the most important factors found in our data that influence the likelihood of having a stroke, developing diabetes, or developing heart disease.")
st.write("According to these charts the factors are:")
with st.expander("Stroke"):
    st.write("Age, Smoking Status, Blood Glucose (Blood Sugar), and Hypertension (High Blood Pressure)")
    # -----------------------
    # Stroke Random Forest
    # -----------------------
    st.header("Risk Factors for Stroke, Diabetes, and Heart disease")

    st.subheader("Stroke Dataset")
    X_stroke = df_stroke_smote.drop('stroke', axis=1)
    y_stroke = df_stroke_smote['stroke']
    feat_imp_stroke, rf_stroke = train_rf_feature_importance(X_stroke, y_stroke)
    plot_feature_importance(feat_imp_stroke, "Stroke Dataset - Feature Importance (Random Forest)", palette='viridis')

with st.expander("Diabetes"):
    st.write("A1c, Blood Glucose (Blood Sugar), Age, and BMI")
    # -----------------------
    # Diabetes Random Forest
    # -----------------------
    st.subheader("Diabetes Dataset")
    X_diabetes = df_balanced_diabetes.drop('diabetes', axis=1)
    y_diabetes = df_balanced_diabetes['diabetes']
    # Encode categorical columns
    for col in ['gender', 'smoking_history']:
        X_diabetes[col] = LabelEncoder().fit_transform(X_diabetes[col])

    feat_imp_diabetes, rf_diabetes = train_rf_feature_importance(X_diabetes, y_diabetes)
    plot_feature_importance(feat_imp_diabetes, "Diabetes Dataset - Feature Importance (Random Forest)")

with st.expander("Heart Disease"):
    st.write("Age, High Blood Pressure, BMI, and High Cholesterol")
    # -----------------------
    # Heart Random Forest
    # -----------------------
    st.subheader("Heart Dataset")
    X_heart = df_heart_smote.drop('HeartDiseaseorAttack', axis=1)
    y_heart = df_heart_smote['HeartDiseaseorAttack']
    feat_imp_heart, rf_heart = train_rf_feature_importance(X_heart, y_heart)
    plot_feature_importance(feat_imp_heart, "Heart Dataset - Feature Importance (Random Forest)", palette='viridis')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


# --- 1. PCA Function Definition ---
def perform_pca(df, target_col, n_components=2):
    """
    Performs PCA on a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column to be excluded from PCA.
        n_components (int): The number of principal components to return.

    Returns:
        tuple: (df_pca, pca_model, X_scaled) - DataFrame with components, the PCA model, and the scaled features.
    """
    st.subheader(f"PCA on {target_col.capitalize()} Dataset")

    # 1. Separate Features and Target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 2. Standardize the Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # 3. Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    # Create a DataFrame for the principal components
    column_names = [f'PC{i + 1}' for i in range(n_components)]
    df_pca = pd.DataFrame(data=principal_components, columns=column_names)
    df_pca[target_col] = y.reset_index(drop=True)

    st.write(f"**Explained Variance Ratio for {n_components} components:**")
    st.code(pca.explained_variance_ratio_)
    st.write(f"**Total Explained Variance:** {np.sum(pca.explained_variance_ratio_):.2f}")

    return df_pca, pca, X_scaled_df


# --- 2. Scree Plot Function for Optimal Component Selection ---
def plot_scree_plot(pca_model, title):
    """Plots the explained variance ratio to help select the number of components."""
    fig, ax = plt.subplots(figsize=(8, 5))
    explained_variance = pca_model.explained_variance_ratio_
    components = range(1, len(explained_variance) + 1)

    ax.bar(components, explained_variance)
    ax.plot(components, explained_variance.cumsum(), marker='o', linestyle='--', color='red')

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title(f'Scree Plot for {title}')
    ax.set_xticks(components)
    ax.grid(True)

    st.pyplot(fig)


# --- 3. 2D PCA Visualization Function ---
def plot_pca_2d(df_pca, target_col, title):
    """Plots the first two principal components."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Ensure the target column is categorical for distinct colors
    sns.scatterplot(
        x='PC1',
        y='PC2',
        hue=df_pca[target_col].astype(str),
        palette='viridis',
        data=df_pca,
        ax=ax,
        s=50,
        alpha=0.6
    )
    ax.set_title(f'2D PCA of {title}')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    st.pyplot(fig)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


# --- 1. PCA Function Definition ---
def perform_pca(df, target_col, n_components=2):
    """
    Performs PCA on a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column to be excluded from PCA.
        n_components (int): The number of principal components to return.

    Returns:
        tuple: (df_pca, pca_model, X_scaled) - DataFrame with components, the PCA model, and the scaled features.
    """
    st.subheader(f"PCA on {target_col.capitalize()} Dataset")

    # 1. Separate Features and Target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 2. Standardize the Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # 3. Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    # Create a DataFrame for the principal components
    column_names = [f'PC{i + 1}' for i in range(n_components)]
    df_pca = pd.DataFrame(data=principal_components, columns=column_names)
    df_pca[target_col] = y.reset_index(drop=True)

    st.write(f"**Explained Variance Ratio for {n_components} components:**")
    st.code(pca.explained_variance_ratio_)
    st.write(f"**Total Explained Variance:** {np.sum(pca.explained_variance_ratio_):.2f}")

    return df_pca, pca, X_scaled_df


# --- 2. Scree Plot Function for Optimal Component Selection ---
def plot_scree_plot(pca_model, title):
    """Plots the explained variance ratio to help select the number of components."""
    fig, ax = plt.subplots(figsize=(8, 5))
    explained_variance = pca_model.explained_variance_ratio_
    components = range(1, len(explained_variance) + 1)

    ax.bar(components, explained_variance)
    ax.plot(components, explained_variance.cumsum(), marker='o', linestyle='--', color='red')

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title(f'Scree Plot for {title}')
    ax.set_xticks(components)
    ax.grid(True)

    st.pyplot(fig)


# --- 3. 2D PCA Visualization Function ---
def plot_pca_2d(df_pca, target_col, title):
    """Plots the first two principal components."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Ensure the target column is categorical for distinct colors
    sns.scatterplot(
        x='PC1',
        y='PC2',
        hue=df_pca[target_col].astype(str),
        palette='viridis',
        data=df_pca,
        ax=ax,
        s=50,
        alpha=0.6
    )
    ax.set_title(f'2D PCA of {title}')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    st.pyplot(fig)


# -----------------------------------------------------------
# Perform PCA on the Three Balanced Datasets
# -----------------------------------------------------------

st.header("✨ Principal Component Analysis (PCA)")
st.write(
    "PCA is used here to reduce the dimensionality of the datasets while retaining as much of the original variance as possible. This also helps in visualizing the data.")

# --- PCA for Stroke Dataset ---
df_stroke_pca, pca_stroke, X_stroke_scaled = perform_pca(df_stroke_smote, 'stroke')
plot_pca_2d(df_stroke_pca, 'stroke', "Stroke Dataset")

# Optional: Scree Plot to determine optimal components for Stroke
pca_full_stroke = PCA().fit(X_stroke_scaled)
plot_scree_plot(pca_full_stroke, "Stroke Dataset (Full PCA)")


# --- PCA for Diabetes Dataset ---
# Need to ensure 'gender' and 'smoking_history' are numerically encoded for df_balanced_diabetes
# before standardizing and applying PCA, as done in the feature importance section.

df_diabetes_pca_data = df_balanced_diabetes.copy()
# Re-apply Label Encoding just in case the original function didn't save the changes globally
le = LabelEncoder()
df_diabetes_pca_data['gender'] = le.fit_transform(df_diabetes_pca_data['gender'])
df_diabetes_pca_data['smoking_history'] = le.fit_transform(df_diabetes_pca_data['smoking_history'])

df_diabetes_pca, pca_diabetes, X_diabetes_scaled = perform_pca(df_diabetes_pca_data, 'diabetes')
plot_pca_2d(df_diabetes_pca, 'diabetes', "Diabetes Dataset")

# Optional: Scree Plot to determine optimal components for Diabetes
pca_full_diabetes = PCA().fit(X_diabetes_scaled)
plot_scree_plot(pca_full_diabetes, "Diabetes Dataset (Full PCA)")


# --- PCA for Heart Disease Dataset ---
df_heart_pca, pca_heart, X_heart_scaled = perform_pca(df_heart_smote, 'HeartDiseaseorAttack')
plot_pca_2d(df_heart_pca, 'HeartDiseaseorAttack', "Heart Disease Dataset")

# Optional: Scree Plot to determine optimal components for Heart Disease
pca_full_heart = PCA().fit(X_heart_scaled)
plot_scree_plot(pca_full_heart, "Heart Disease Dataset (Full PCA)")
