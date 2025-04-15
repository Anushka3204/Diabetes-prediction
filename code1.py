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

# Create the Streamlit app
st.title("Diabetes Prediction and Analysis")

# Sidebar for model selection
model_type = st.sidebar.selectbox("Select Model", ["Classification", "Regression", "Clustering"])

# Classification UI
if model_type == "Classification":
    st.header("Diabetes Prediction")
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data = []
    for feature in features:
        input_data.append(st.number_input(f"Enter value for {feature}:", value=0.0))
    
    if st.button("Predict"):
        input_data_reshaped = [input_data]
        # Assuming you have the scaler saved as well, load it here
        # with open('scaler.pkl', 'rb') as file:
        #     scaler = pickle.load(file)
        # input_data_scaled = scaler.transform(input_data_reshaped)
        prediction = classification_model.predict(input_data_reshaped)[0]
        st.write(f"**Prediction:** {'Diabetes' if prediction == 1 else 'No Diabetes'}")

# Regression UI
elif model_type == "Regression":
    st.header("Insulin Level Prediction")
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    input_data = []
    for feature in features:
        input_data.append(st.number_input(f"Enter value for {feature}:", value=0.0))
    
    if st.button("Predict"):
        input_data_reshaped = [input_data]
        # Assuming you have the scaler saved as well, load it here
        # with open('scaler.pkl', 'rb') as file:
        #     scaler = pickle.load(file)
        # input_data_scaled = scaler.transform(input_data_reshaped)
        prediction = regression_model.predict(input_data_reshaped)[0]
        st.write(f"**Predicted Insulin Level:** {prediction:.2f}")

# Clustering UI
elif model_type == "Clustering":
    st.header("Age Cluster Prediction")
    age = st.number_input("Enter the age:", value=0.0)
    
    if st.button("Predict"):
        prediction = clustering_model.predict([[age]])[0]
        st.write(f"**Age Cluster:** {prediction}")
        if prediction == 0:
            st.write("This individual belongs to the younger age group.")
            st.write("Individuals in this cluster have a lower risk of diabetes but should still maintain a healthy lifestyle.")
        elif prediction == 1:
            st.write("This individual belongs to the older age group.")
            st.write("Individuals in this cluster have a higher risk of diabetes and should be monitored regularly.")
        elif prediction == 2:
            st.write("This individual belongs to the middle-aged group.")
            st.write("Individuals in this cluster have a moderate risk of diabetes and should focus on preventive measures.")