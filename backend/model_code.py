# model_code.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Step 1: Load the dataset
def load_data():
    # Replace with the correct path to your dataset
    df = pd.read_csv('creditcard.csv')
    return df

# Step 2: Preprocess the data
def preprocess_data(df):
    X = df.drop('Class', axis=1)  # Features
    y = df['Class']  # Target variable (fraud or not)
    return X, y

# Step 3: Train-Test Split
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 4: Train the Model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Step 5: Save the Model
def save_model(model):
    joblib.dump(model, 'fraud_detection_model.pkl')
    print("Model saved as 'fraud_detection_model.pkl'")

# Main function to execute the pipeline
def main():
    # Load and preprocess data
    df = load_data()
    X, y = preprocess_data(df)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = train_model(X_train, y_train)

    # Save the trained model
    save_model(model)

if __name__ == "__main__":
    main()
