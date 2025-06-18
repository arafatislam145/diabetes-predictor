import joblib
import numpy as np

# Load trained model
model = joblib.load('model/model.pkl')

# Input data: 8 features (must be in correct order)
# Format: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

sample = np.array([[2, 120, 70, 20, 79, 25.0, 0.5, 30]])  # You can change these values

# Predict
prediction = model.predict(sample)[0]

# Output result
if prediction == 1:
    print("🔴 The person is likely to have diabetes.")
else:
    print("🟢 The person is not likely to have diabetes.")
