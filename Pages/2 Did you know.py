import streamlit as st
st.title("Did you know? 🤔")


st.write("Did you know that stroke, diabetes, and heart disease have several overlaps in their risk factors? This is for multiple reasons. Mainly the risk factors for diabetes overlaps with stroke and heart disease. Furhtermore diabetes damages blood vessels which further increases the risk for stroke and heart disease.")
st.write("If you are to develop any one of these three pathologies, your risk for any of the other two increase. We are currently going to focus on the relationship between stroke and diabetes (heart disease to come later!).")
st.write("Explore below to find the risk factors for stroke and diabetes")

with st.expander("Stroke risk factors"):
    st.write("High blood pressure, High Cholestrol, Heart disease, Diabetes, Obesity, Sickle cell disease, Smoking, ")

with st.expander("Diabetes risk factors"):
    st.write("High blood pressure, factors from heart disease, Obesity, Smoking")

with st.expander("Overlapping risk factors"):
    st.write("High blood pressure (manageable, Obesity (manageable), Smoking (manageable)")
    st.write("We can see there is several manageable factors that overlap. It may be difficult, but these are in our control.")
    st.write("If you feel like you need help, please see the information sites to see where to get help in managing these factors.")

with st.expander("Sources"):
    st.write("https://www.cdc.gov/stroke/risk-factors/index.html")
    st.write("https://www.cdc.gov/diabetes/risk-factors/index.html")
    st.write("https://health.umms.org/2024/02/02/relationship-between-diabetes-and-stroke/")
