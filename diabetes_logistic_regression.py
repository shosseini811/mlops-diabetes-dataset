"""
Module for predicting diabetes using logistic regression.
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
DATA_PATH = "diabetes.csv"
diabetes_data = pd.read_csv(DATA_PATH)

# Split the data into training and testing sets
X = diabetes_data.drop("Outcome", axis=1)
y = diabetes_data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
logreg = LogisticRegression(max_iter=1000, C=0.01)
logreg.fit(X_train, y_train)

# Predict and evaluate the model's performance
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
