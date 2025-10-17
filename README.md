Stroke and Diabetes comparison

Goal: The goal of this project is to look at the stroke and diabetes dataset, clean the data, find correlations of risk factors between the two, and provide visualizations and interactive plots for users to better understand the risk factors. Also to provide information for helping manage the risk factors. 
After finding correlative risk factors, allow users to see how the managable (changable) risk factors increase or decrease the likelihood of them developing the disease. 

How it's made: This app is coded in python and adapted to run with the streamlit application creator. Several python libraries were utilized and two Kaggle datasets.
Datasets: 
Stroke: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
Diabetes: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

The requirements are: 
streamlit
seaborn
pandas
numpy
matplotlib
scikit-learn
imblearn
altair

Lessons learned: 
Technical side: How to work with streamlit and many of it's interactive features. I also learned more about data set imbalance and how to work with that, using SMOTE and undersampling. 
Analysis side: Blood glucose is a large risk factor for both stroke and diabetes. I did not expect blood glucose to be such a large factor for stroke. 
