import pandas as pd
import pickle
from gradio import Interface, Number

# Load the pre-trained model
with open("diabetes_classifier.pkl", "rb") as file:
    loaded_model = pickle.load(file)
diabetes_classifier = loaded_model['model']
columns = loaded_model['columns']

# Function to make predictions
def predict_diabetes_func(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    input_df = pd.DataFrame([input_data], columns=columns)
    prediction = diabetes_classifier.predict(input_df)
    return "Positive" if prediction[0] == 1 else "Negative"

# Define Gradio interface
iface = Interface(
    fn=predict_diabetes_func,  # Updated function name
    inputs=[
        Number(label="Pregnancies"),
        Number(label="Glucose"),
        Number(label="BloodPressure"),
        Number(label="SkinThickness"),
        Number(label="Insulin"),
        Number(label="BMI"),
        Number(label="DiabetesPedigreeFunction"),
        Number(label="Age"),
    ],
    outputs="text",
    live=True,
)

# Check if the interface is running, close it, and launch a new one
# check the status of gradio
# launch the interface
iface.launch(server_port=7871, server_name="0.0.0.0")
