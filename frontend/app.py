import numpy as np
import joblib
from flask import Flask, request, render_template

# Load model
model = joblib.load('fraud_detection_model.pkl')

# Create app
app = Flask(__name__)

# Function to generate 30 features (Time + V1–V28 + Amount)
def generate_full_input(user_time, user_amount):
    v_features = np.random.normal(0, 1, 28)
    features = np.concatenate([[user_time], v_features, [user_amount]])
    return features.reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        time = float(request.form['time'])
        amount = float(request.form['amount'])

        # Generate input features
        features = generate_full_input(time, amount)

        # Make prediction
        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "⚠️ Fraudulent Transaction"
        else:
            result = "✅ Non-Fraudulent Transaction"

        return render_template('index.html', prediction_result=result)
    
    except Exception as e:
        return render_template('index.html', prediction_result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
