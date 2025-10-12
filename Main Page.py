import pandas as pd
from matplotlib.pyplot import figure, tight_layout, show, title
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


#load stroke dataframe
df1 = pd.read_csv("C:/Users/brad_/MSU/Year 1/CMSE 830/Project/healthcare-dataset-stroke-data.csv")

#drops:
#drop columns: married, work type, and residence type
df1 = df1.drop(columns=['id', 'ever_married', 'work_type', 'Residence_type'])

#drop 1 datum for "other gender"
index_to_drop = df1[df1['gender'] == 'Other'].index
df1 = df1.drop(index_to_drop)

print(df1.head())


#load diabetes datafram
df2 = pd.read_csv("C:/Users/brad_/MSU/Year 1/CMSE 830/Project/diabetes_prediction_dataset.csv")
new_order = ['gender', 'age', 'hypertension', 'heart_disease', 'blood_glucose_level', 'bmi', 'smoking_history', 'HbA1c_level', 'diabetes']
df2=df2[new_order]
print(df2.head())



#most code for balancing pulled from chat GPT
#stroke dataset balance
# --- Load and clean dataset ---
# --- Encode categorical variables ---
label_encoder = LabelEncoder()
df1['gender'] = label_encoder.fit_transform(df1['gender'])

if 'smoking_status' in df1.columns:
    df1['smoking_status'] = label_encoder.fit_transform(df1['smoking_status'])

# --- Separate features and target ---
X = df1.drop('stroke', axis=1)
y = df1['stroke']

# --- Handle missing values ---
# Use mean for numeric columns (you could also use median or mode)
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# --- Apply SMOTE ---
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_imputed, y)

# --- Combine back into a single balanced DataFrame ---
df_stroke_smote = pd.concat([X_smote, y_smote], axis=1)

# --- Verify results ---
print("Balanced dataset shape:", df_stroke_smote.shape)
print(df_stroke_smote['stroke'].value_counts())
print(df_stroke_smote.isnull().sum())  # confirm no NaNs

#--------------------------------------------------------
#diabetes dataset balance
#--------------------------------------------------------

# Load and reorder
df2 = pd.read_csv("C:/Users/brad_/MSU/Year 1/CMSE 830/Project/diabetes_prediction_dataset.csv")
new_order = ['gender', 'age', 'hypertension', 'heart_disease', 'blood_glucose_level',
             'bmi', 'smoking_history', 'HbA1c_level', 'diabetes']
df2 = df2[new_order]

# Separate majority and minority classes
df_majority = df2[df2.diabetes == 0]
df_minority = df2[df2.diabetes == 1]

# Downsample majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,     # no replacement
                                   n_samples=len(df_minority),  # match minority count
                                   random_state=42)

# Combine minority and downsampled majority
df_balanced_diabetes = pd.concat([df_majority_downsampled, df_minority])

# Shuffle dataset
df_balanced_diabetes = df_balanced_diabetes.sample(frac=1, random_state=42).reset_index(drop=True)

print("Balanced diabetes dataset shape:", df_balanced_diabetes.shape)
print(df_balanced_diabetes['diabetes'].value_counts())


#heatmap
#stroke SMOTE
figure(figsize=(12, 10))
sns.heatmap(df_stroke_smote.corr(numeric_only=True),
            annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
title("Correlation Heatmap - Stroke-SMOTE", fontsize=16, pad=15)
tight_layout()
show()



#diabetes
figure(figsize=(8, 6))  # Adjust figure size if needed
sns.heatmap(df_balanced_diabetes.corr(numeric_only=True),
            annot=True,
            cmap='coolwarm',
            vmin=-1, vmax=1,
            fmt=".2f",
            linewidths=0.5)

title("Correlation Heatmap - Diabetes", fontsize=16, pad=15)
tight_layout()
show()


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: suppress warnings
import warnings
warnings.filterwarnings('ignore')


# Load and clean
df_stroke = pd.read_csv("C:/Users/brad_/MSU/Year 1/CMSE 830/Project/healthcare-dataset-stroke-data.csv")
df_stroke = df_stroke.drop(columns=['id', 'ever_married', 'work_type', 'Residence_type'])
df_stroke = df_stroke[df_stroke['gender'] != 'Other']

# Encode categorical features
label_encoder = LabelEncoder()
df_stroke['gender'] = label_encoder.fit_transform(df_stroke['gender'])
if 'smoking_status' in df_stroke.columns:
    df_stroke['smoking_status'] = label_encoder.fit_transform(df_stroke['smoking_status'])

# Separate features and target
X_stroke = df_stroke.drop('stroke', axis=1)
y_stroke = df_stroke['stroke']

# Train Random Forest
rf_stroke = RandomForestClassifier(n_estimators=200, random_state=42)
rf_stroke.fit(X_stroke, y_stroke)

# Feature importance
feat_importance_stroke = pd.DataFrame({
    'Feature': X_stroke.columns,
    'Importance': rf_stroke.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot
figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importance_stroke, palette='viridis')
title("Stroke Dataset - Feature Importance (Random Forest)")
show()


# Load and reorder
df_diabetes = pd.read_csv("C:/Users/brad_/MSU/Year 1/CMSE 830/Project/diabetes_prediction_dataset.csv")
new_order = ['gender', 'age', 'hypertension', 'heart_disease', 'blood_glucose_level',
             'bmi', 'smoking_history', 'HbA1c_level', 'diabetes']
df_diabetes = df_diabetes[new_order]

# Encode categorical features
df_diabetes['gender'] = label_encoder.fit_transform(df_diabetes['gender'])
df_diabetes['smoking_history'] = label_encoder.fit_transform(df_diabetes['smoking_history'])

# Separate features and target
X_diabetes = df_diabetes.drop('diabetes', axis=1)
y_diabetes = df_diabetes['diabetes']

# Train Random Forest
rf_diabetes = RandomForestClassifier(n_estimators=200, random_state=42)
rf_diabetes.fit(X_diabetes, y_diabetes)

# Feature importance
feat_importance_diabetes = pd.DataFrame({
    'Feature': X_diabetes.columns,
    'Importance': rf_diabetes.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot
figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importance_diabetes, palette='magma')
title("Diabetes Dataset - Feature Importance (Random Forest)")
show()
