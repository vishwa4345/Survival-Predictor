import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Load Trained Model
# -------------------------------
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📌 About Project")

st.sidebar.info("""
**Titanic Survival Prediction**

This application Developed by "VISHWA HIREMATH", predicts whether a passenger is likely to survive the Titanic disaster using a trained **Logistic Regression** model.

**Technologies Used**
- Python
- Pandas
- Scikit-learn
- Streamlit
- Pickle

**Developed By**
Vishwa Hiremath
""")

# -------------------------------
# Main Title
# -------------------------------
st.title("🚢 Titanic Survival Prediction")

st.write(""
by VISHWA HIREMATH.
"")

st.write("""
Enter the passenger details below and click **Predict** to estimate the survival probability.
""")

st.markdown("---")

# -------------------------------
# User Inputs
# -------------------------------

passenger_id = st.number_input(
    "Passenger ID",
    min_value=1,
    value=1
)

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=25
)

sibsp = st.number_input(
    "Siblings/Spouses Aboard",
    min_value=0,
    value=0
)

parch = st.number_input(
    "Parents/Children Aboard",
    min_value=0,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=50.0
)

embarked = st.selectbox(
    "Embarked Port",
    ["C", "Q", "S"]
)

# -------------------------------
# Data Preprocessing
# -------------------------------

sex = 1 if sex == "Male" else 0

embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

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

# -------------------------------
# Prediction
# -------------------------------

if st.button("🔍 Predict"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    survival_probability = probability[0][1] * 100

    st.markdown("---")

    if prediction[0] == 1:
        st.success("✅ Passenger is likely to Survive")
    else:
        st.error("❌ Passenger is unlikely to Survive")

    st.subheader("Prediction Probability")

    st.write(f"**Survival Probability:** {survival_probability:.2f}%")

    st.progress(min(survival_probability / 100, 1.0))

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")

st.caption(
    "Developed using Python, Scikit-learn and Streamlit | "
    "Machine Learning Deployment Project"
)
