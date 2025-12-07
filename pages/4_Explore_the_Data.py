import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler # Consolidated import
from sklearn.decomposition import PCA # Consolidated import
from sklearn.model_selection import train_test_split # Consolidated import
from sklearn.svm import SVC # Consolidated import
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # Consolidated import
import numpy as np # Consolidated import
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


# -----------------------
# Streamlit Page Setup
# -----------------------
st.set_page_config(page_title="Stroke, Diabetes, & Heart Disease Analysis", layout="wide")
st.title("Stroke, Diabetes, & Heart Disease Analysis")



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


@st.cache_data
# --- Helper Function for SVM ---
def perform_svm_classification(df, target_col, dataset_name):
    """
    Performs data splitting, scaling, SVC training, and evaluation.
    """
    st.subheader(f"{dataset_name} - Support Vector Machine (SVM)")
    st.write("Below you'll see two support vector models showing an 87% accuracy and a 77% accuracy respetively. The first model while more successful cannot be directly tied to variable while the lower accuracy can be. You'll see several factors along with their weight/importance")
    # 1. Separate Features (X) and Target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 2. Handle Categorical Features (Necessary for Diabetes dataset)
    # The Stroke and Heart datasets already have numerical features from your setup.
    le = LabelEncoder()
    # Note: These checks ensure the columns exist before encoding
    if 'gender' in X.columns and X['gender'].dtype == 'object':
        X['gender'] = le.fit_transform(X['gender'])
    if 'smoking_history' in X.columns and X['smoking_history'].dtype == 'object':
        X['smoking_history'] = le.fit_transform(X['smoking_history'])
    # Heart disease columns
    if 'Sex' in X.columns and X['Sex'].dtype == 'object':
        X['Sex'] = le.fit_transform(X['Sex'])
    if 'smoker' in X.columns and X['smoker'].dtype == 'object':
        X['smoker'] = le.fit_transform(X['smoker'])


    # 3. Split the Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4. Standardize the Features (Critical for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Training (Using a radial basis function (RBF) kernel)
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # 6. Evaluation
    y_pred = svm_model.predict(X_test_scaled)

    st.text(f"Classification Report for {dataset_name}:")
    st.code(classification_report(y_test, y_pred))

    st.text(f"Confusion Matrix for {dataset_name}:")
    st.code(confusion_matrix(y_test, y_pred))
    st.markdown("---")


# --- Helper function for Linear SVM ---
@st.cache_data
def train_linear_svm_and_show_coefficients(df, target_col, random_state=42):
    """
    Trains a Linear SVM model, evaluates it, and displays feature coefficients.

    Args:
        df (pd.DataFrame): The input balanced DataFrame.
        target_col (str): The name of the target column.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (accuracy, report, coefficients_df)
    """
    st.header(f"Linear SVM Analysis for {target_col.replace('_', ' ').title()}")

    # 1. Separate Features (X) and Target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Handle categorical features for Diabetes (ensure numerical encoding)
    if target_col == 'diabetes':
        le = LabelEncoder()
        for col in ['gender', 'smoking_history']:
            # Added check for dtype=='object' to prevent error if already encoded
            if col in X.columns and X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col])

    feature_names = X.columns  # Store original feature names

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    # 3. Standardize Features (CRUCIAL for Linear SVM and Coefficient Interpretation)
    # The coefficients are only comparable if features are scaled first.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train the Linear SVM Model
    # Use kernel='linear' and increase C for less regularization if needed,
    # but C=1.0 is a good starting point.
    svm_model = SVC(kernel='linear', random_state=random_state, C=1.0)
    svm_model.fit(X_train_scaled, y_train)

    # 5. Evaluation (Optional, but good practice)
    y_pred = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False, zero_division=0)

    st.subheader("Model Performance")
    st.write(f"**Accuracy on Test Set:** **{accuracy:.4f}**")
    st.code(report)

    # --------------------------------------------------------
    # 6. Extract and Display Coefficients (The core request)
    # --------------------------------------------------------

    # Linear SVM coefficients are stored in the 'coef_' attribute
    # Since we have binary classification, coef_ will be a 1xN array (N=features)
    coefficients = svm_model.coef_[0]

    # Create DataFrame for easy sorting and display
    coefficients_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Magnitude': np.abs(coefficients)  # Absolute value for importance ranking
    }).sort_values(by='Magnitude', ascending=False)

    st.subheader("Feature Importance from Linear SVM Coefficients")
    st.markdown(
        "The **Magnitude** indicates the importance, and the **Coefficient** sign (+/-) indicates the direction of influence on the likelihood of the positive class (1)."
    )
    st.dataframe(coefficients_df)

    return accuracy, report, coefficients_df


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

    # --------------------------------------------------------------------
    # A. SVM for Stroke Dataset
    # --------------------------------------------------------------------
    perform_svm_classification(df_stroke_smote, 'stroke', "Stroke Dataset")

    # --- 1. Linear SVM on Stroke Dataset ---
    acc_stroke_lin, report_stroke_lin, coef_stroke = train_linear_svm_and_show_coefficients(
        df_stroke_smote.copy(), 'stroke'
    )

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

    # --------------------------------------------------------------------
    # B. SVM for Diabetes Dataset
    # --------------------------------------------------------------------
    # We need a copy of the balanced diabetes data to ensure 'gender' and
    # 'smoking_history' are encoded numerically before scaling for the model.
    df_diabetes_svm = df_balanced_diabetes.copy()
    perform_svm_classification(df_diabetes_svm, 'diabetes', "Diabetes Dataset")

    # --- 2. Linear SVM on Diabetes Dataset ---
    acc_diabetes_lin, report_diabetes_lin, coef_diabetes = train_linear_svm_and_show_coefficients(
        df_balanced_diabetes.copy(), 'diabetes'
    )

with st.expander("Heart Disease (can take about 90 seconds to load all data)"):
    st.write("Age, High Blood Pressure, BMI, and High Cholesterol")
    # -----------------------
    # Heart Random Forest (Kept on Full Data for high fidelity features)
    # -----------------------
    st.subheader("Heart Dataset")
    X_heart = df_heart_smote.drop('HeartDiseaseorAttack', axis=1)
    y_heart = df_heart_smote['HeartDiseaseorAttack']
    feat_imp_heart, rf_heart = train_rf_feature_importance(X_heart, y_heart)
    plot_feature_importance(feat_imp_heart, "Heart Dataset - Feature Importance (Random Forest)", palette='viridis')

    # --------------------------------------------------------------------
    # C. SVM for Heart Disease Dataset (Downsampled RBF Kernel)
    # --------------------------------------------------------------------
    st.subheader("RBF Kernel SVM Analysis (Downsampled)")
    st.markdown("**Downsampling Note:** The RBF kernel SVM is a complex and timely, therefore the training data was limited to 5,000 samples to reduce comuptation time.")

    # 1. Define the target sample size for the fast RBF model
    N_SAMPLES_RBF = 5000

    # 2. Downsample the balanced data
    if len(df_heart_smote) > N_SAMPLES_RBF:
        df_heart_rbf_downsampled = resample(
            df_heart_smote,
            replace=False,
            n_samples=N_SAMPLES_RBF,
            random_state=42,
            stratify=df_heart_smote['HeartDiseaseorAttack']
        )
    else:
        df_heart_rbf_downsampled = df_heart_smote.copy()

    # 3. Perform RBF SVM classification on the smaller dataset
    perform_svm_classification(
        df_heart_rbf_downsampled,
        'HeartDiseaseorAttack',
        f"Heart Disease Dataset (Downsampled N={len(df_heart_rbf_downsampled)})"
    )

    # --------------------------------------------------------------------
    # D. Linear SVM for Heart Disease Dataset (Downsampled to 30000)
    # --------------------------------------------------------------------
    # We will use the already downsampled data (df_heart_rbf_downsampled)
    # to maintain a consistent comparison size with the RBF kernel model.
    st.subheader(f"Linear Kernel SVM Analysis (Downsampled N={len(df_heart_rbf_downsampled)})")
    acc_heart_lin, report_heart_lin, coef_heart = train_linear_svm_and_show_coefficients(
        df_heart_rbf_downsampled.copy(),  # Use the downsampled data
        'HeartDiseaseorAttack'
    )