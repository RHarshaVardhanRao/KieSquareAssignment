import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_iris


# Load Iris dataset
data = load_iris(as_frame=True)
df = data.frame  # Iris dataset in DataFrame format
X = df.drop('target', axis=1)
y = df['target']

# Load the saved model pipeline
@st.cache_resource
def load_model():
    with open(r'B:\Harsha\Praxis\Kie Square Assigment\2.)Datascience Life Cycle\xgb_model_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

# Function to make predictions
def predict(features, model):
    # Create a DataFrame with the correct column names expected by the model
    feature_names = X.columns.to_list()
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Predict using the model
    prediction = model.predict(features_df)
    return prediction

# Streamlit App UI
st.title("Iris Species Prediction App")

st.write("""
### Input the features of the Iris flower below to get a prediction:
""")

# Input fields for feature values
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Load the model
model = load_model()

# When the user clicks the 'Predict' button
if st.button('Predict'):
    # Get the input feature values as a list
    features = [sepal_length, sepal_width, petal_length, petal_width]

    # Make predictions
    prediction = predict(features, model)

    # Map prediction to the corresponding Iris species
    iris_species = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}
    predicted_species = iris_species[prediction[0]]

    # Display the result
    st.write(f"### Predicted Iris Species: {predicted_species}")
