import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the models
with open('classification_model.pkl', 'rb') as file:
    classification_model = pickle.load(file)
with open('regression_model.pkl', 'rb') as file:
    regression_model = pickle.load(file)
with open('clustering_model.pkl', 'rb') as file:
    clustering_model = pickle.load(file)

# App title
st.title("🩺 Diabetes Prediction and Analysis")

# Sidebar model selection
model_type = st.sidebar.selectbox("Select Model", ["Classification", "Regression", "Clustering"])

# --------------------- CLASSIFICATION ---------------------
if model_type == "Classification":
    st.header("🔍 Diabetes Classification")

    int_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Age']
    float_features = ['BMI', 'DiabetesPedigreeFunction']

    input_data = []
    st.subheader("Enter the following patient details:")
    for feature in int_features:
        input_data.append(st.number_input(f"{feature} (int):", min_value=0, value=0, format="%d"))
    for feature in float_features:
        input_data.append(st.number_input(f"{feature} (float):", min_value=0.0, value=0.0, format="%.2f"))

    if st.button("Predict Diabetes"):
        input_data_reshaped = [input_data]
        prediction = classification_model.predict(input_data_reshaped)[0]
        st.success(f"**Prediction:** {'🧬 Diabetes Detected' if prediction == 0 else '✅ No Diabetes Detected'}")

# --------------------- REGRESSION ---------------------
elif model_type == "Regression":
    st.header("📊 Insulin Level Regression")

    int_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Age', 'Outcome']
    float_features = ['BMI', 'DiabetesPedigreeFunction']

    input_data = []
    st.subheader("Enter the following patient details:")
    for feature in int_features:
        input_data.append(st.number_input(f"{feature} (int):", min_value=0, value=0, format="%d"))
    for feature in float_features:
        input_data.append(st.number_input(f"{feature} (float):", min_value=0.0, value=0.0, format="%.2f"))

    if st.button("Predict Insulin Level"):
        input_data_reshaped = [input_data]
        prediction = regression_model.predict(input_data_reshaped)[0]
        st.success(f"**Predicted Insulin Level:** {prediction:.2f}")

# --------------------- CLUSTERING ---------------------
elif model_type == "Clustering":
    st.header("📈 Age-Based Clustering")

    age = st.number_input("Enter Age (int):", min_value=0, value=0, format="%d")

    if st.button("Determine Age Cluster"):
        prediction = clustering_model.predict([[age]])[0]
        st.success(f"**Age Cluster:** {prediction}")
        
        if prediction == 0:
            st.info(" This individual belongs to the younger age group.\nRisk of diabetes is lower, but maintaining a healthy lifestyle is important.")
        elif prediction == 1:
            st.warning(" This individual belongs to the older age group.\nHigher risk of diabetes. Regular check-ups are recommended.")
        elif prediction == 2:
            st.info("This individual belongs to the middle-aged group.\nModerate diabetes risk. Preventive care is advised.")


