import os
import psycopg2
import pandas as pd
import pickle
from gradio import Interface, Number
from datetime import datetime

# Database connection
DB_HOST = os.environ['DB_HOST']
DB_PORT = os.environ['DB_PORT']
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']

connection = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

cursor = connection.cursor()
# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    timestamp TIMESTAMP,
    Pregnancies INT,
    Glucose INT,
    BloodPressure INT,
    SkinThickness INT,
    Insulin INT,
    BMI REAL,
    DiabetesPedigreeFunction REAL,
    Age INT,
    prediction TEXT
)
""")

# Load the pre-trained model
with open("diabetes_classifier.pkl", "rb") as file:
    loaded_model = pickle.load(file)
diabetes_classifier = loaded_model['model']
columns = loaded_model['columns']

# Function to make predictions
def predict_diabetes_func(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    timestamp = datetime.now()
    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    input_df = pd.DataFrame([input_data], columns=columns)
    prediction = diabetes_classifier.predict(input_df)
    result = "Positive" if prediction[0] == 1 else "Negative"
    
    # Insert into the database
    cursor.execute("""
    INSERT INTO predictions (timestamp, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, prediction)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (timestamp, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, result))
    connection.commit()

    return result

# Define Gradio interface
iface = Interface(
    fn=predict_diabetes_func,
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

# Launch the interface
iface.launch(server_port=7860, server_name="0.0.0.0")
