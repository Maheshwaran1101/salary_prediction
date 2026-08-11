import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Salary_Data.csv")

# Train model
X = df[['YearsExperience']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("Salary Prediction App")

st.write("Enter years of experience to predict salary.")

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.1
)

if st.button("Predict Salary"):
    predicted_salary = model.predict([[experience]])[0]

    st.success(
        f"Predicted Salary for {experience} years of experience is ₹{predicted_salary:,.2f}"
    )