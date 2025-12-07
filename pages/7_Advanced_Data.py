import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# -----------------------
# Streamlit Page Setup
# -----------------------
st.set_page_config(page_title="Stroke & Diabetes Analysis", layout="wide")
st.title("Stroke & Diabetes Dataset Analysis")
st.write("Here you can explore the specifics behind the data and see the reasoning behind data that was and was not used")

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

df_balanced_heart = balance_heart(df_heart.copy())

def balance_heart(df):
    label_encoder = LabelEncoder()
    # FIX 1: Change global 'df_heart' to local 'df'
    df['Sex'] = label_encoder.fit_transform(df['Sex'])

    if 'smoker' in df.columns:
        # FIX 2: Change global 'df_heart' to local 'df'
        df['smoker'] = label_encoder.fit_transform(df['smoker'])

    # --- Separate features and target ---
    X3 = df.drop('HeartDiseaseorAttack', axis=1)
    y3 = df['HeartDiseaseorAttack']

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

#==================================================
#Class imbalance drop down
#==================================================

with st.expander("Class imbalance"):
    st.write("When working with the original datasets you'll notice that the amount of patients with stroke, diabetes, and heart disease is HEAVILY outnumbered by those who do not have diabetes/stroke.")
    st.subheader("Stroke Dataset")
    st.dataframe(df_stroke1.head())
    st.write("Here we see stroke data that looks at factors such as: age, high blood pressure(hypertension), heart disease, average blood glucose level, BMI, smoking history, and whether or not the person has had a stroke before.")
    st.write("But if we take a look at the number of patients with stroke or diabetes vs those with, there is a big descrepency. See below:")
    st.write("The number of people with stroke is: " + str(sum(df_stroke1['stroke'])) + ". Compare that to the total number of people: " + str(len(df_stroke1['stroke'])) + ".")
    st.write("This difference is called a bias. In order to obtain accurate data we need to make the total number of positives the same as the negatives. The 'SMOTE' technique was applied in order to eliminate the gap in the positive and negative groups.")
    st.write("Initial Stroke Dataset:")
    st.write(df_stroke['stroke'].value_counts())
    st.write("Balanced Stroke Dataset (SMOTE):")
    st.write(df_stroke_smote['stroke'].value_counts())



    st.subheader("Diabetes Dataset")
    st.dataframe(df_diabetes.head())
    st.write(
        "Here we see data for diabetes that looks at similar factors as the stroke dataset with a couple extras: age, high blood pressure(hypertension), heart disease, average blood glucose level, BMI, smoking history, and A1c levels.")

    st.write("We run into the same issue here as the stroke dataset however. The imbalance of those who have diabetes is significantly less than those who do.")
    st.write("The number of people with diabetes is: " + str(sum(df_diabetes['diabetes'])) + ". Compare that to the total number of people: " + str(len(df_diabetes['diabetes'])) + ".")
    st.write("For the diabetes dataset a resampling technique was used. By randomly selecting an equal number of representatives from each category we can define a new dataset based off the original. This helps us obtain equal representation from each group")
    st.write("Initial Diabetes Dataset:")
    st.write(df_diabetes['diabetes'].value_counts())
    st.write("Balanced Diabetes Dataset (Undersampling):")
    st.write(df_balanced_diabetes['diabetes'].value_counts())

    st.subheader("Heart Disease Dataset")
    st.dataframe(df_heart.head())
    st.write(
        "As with the stroke and diabetes dataset, the heart dataset share very similar factors.")
    st.write("When looking at the imbalance, we notice the heart disease dataset is no differet."
             "The number of people with Heart disease or Heart attack is " + str(sum(df_heart['Heartdiseaseorattack'])) + ". Compare that to the total number of people: " + str(len(df_heart['Heartdiseaseorattack'])) + ".")

    st.write("Initial Heart Disease Dataset:")
    st.write(df_heart['HeartDiseaseorAttack'].value_counts())
    st.write("Balanced Heart disease Dataset (SMOTE):")
    st.write(df_balanced_heart['HeartDiseaseorAttack'].value_counts())


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

    with st.expander("Heart Disease Heat Map (balanced data)"):
        st.header("Correlation Heatmaps")
        st.subheader("Heart Disease Dataset (SMOTE)")
        plot_correlation_heatmap(df_balanced_heart, "Correlation Heatmap - Heart Disease Dataset (SMOTE)", figsize=(12, 10))
    st.write("Finally bringing in the Heart Disease dataset we notice even more issues. This cannot be a fluke.")

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



with st.expander("Failed Feature Engineering"):
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

    st.header("Principal Component Analysis (PCA)")
    st.write(
        "PCA is used here to reduce the dimensionality of the datasets while retaining as much of the original variance as possible. This is supposed to help in visualizing the data.")
    st.write("However, as you'll see in the plots below all the data is grouped and there is no clear distinction between the target groups and the non-target groups. This means that this data is not suitable for linear dimensionality reduction.")
    st.write("You'll also see in the scree plot there is no obvious elbow. The first PC covers roughly 70% of the variance, but it takes several more PC to cover 90% of the variance in each model.")
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


with st.expander("CMSE830 Rubric info used on this page for ease of grading (to be removed after grading) "):
    st.write("Data Collection: Three datasets used, checked for missing values, several columns dropped from all datasets to match variables across the datasets, Column reordering.")
    st.write("Exploratory Data Analysis and Visualization: 3 visualizations on effects of aging page, 3 random forest visuals on Explore the data, 6 heat maps on advanced data.   ")
    st.write("Data Processing and Feature Engineering: One hot encoding used during data cleaning, scaling used in calibrated model in how at risk am i page, PCA used in failed feature engineering tab in advanced data, SVM used in explore the data  ")
    st.write("Model Development and Evaluation: Random forest and SVM used in explore the data page, Model comparison used in advanced data page and explore the data SVM to linear SVM")
    st.write("Streamlit App Development: 3 interactive plots in effects of aging, three interactive models in how at risk am i, drop down tabs everywhere, detailed info where applicable, caching used for random forest plots and SVM")
    st.write("Github Repository and Documentation: Github repository on https://github.com/3r4d/cmse830_fds")
    st.write("Advanced Modeling Techniques: None implemented")
    st.write("Specialized Data Science Applications: None implemented ")
    st.write("High-Performance Computing: None Implemented")
    st.write("Real-world Application and Impact: Showing importance of risk factors of these diseases, real-world sources, and help resources for managing the most impactful variables that were determined through the data analysis of the app")
    st.write("Exceptional Presentation and visualization: None Implemented. ")