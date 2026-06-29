import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("Titanic Survival Prediction")

st.write("Developed by Vishwa Hiremath")

st.write("Enter passenger details to predict survival.")

# User Inputs

passenger_id = st.number_input("Passenger ID", min_value=1, value=1)

pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Sex", ["Female", "Male"])

age = st.number_input("Age", min_value=0, max_value=100, value=25)

sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, value=0)

parch = st.number_input("Number of Parents/Children", min_value=0, value=0)

fare = st.number_input("Fare", min_value=0.0, value=50.0)

embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert inputs to match training data
sex = 1 if sex == "Male" else 0

embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create input dataframe
input_data = pd.DataFrame({
    "PassengerId": [passenger_id],
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked_Q": [embarked_Q],
    "Embarked_S": [embarked_S]
})

# Prediction
if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger is likely to Survive")
    else:
        st.error("Passenger is unlikely to Survive")
