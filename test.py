# testing saved model

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# import data
Heart_DF = pd.read_csv('dataset.csv')
df = Heart_DF.copy()

# ML READY
# Split your data to train and test. Choose Splitting ratio wisely.
# split the data into X and Y
X = df.drop('target', axis=1)
y = df['target']

# split the data in training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% data for testing
    random_state=42,    # for reproducibility
    stratify=y
    )

print(f"Training set size: {X_train.shape} samples")    # Training size   (Rows, col) 
print(f"Testing set size: {X_test.shape} samples")      # Testing size    (Rows, col)  
print(f"Training set size: {y_train.shape} samples")    # Training size    
print(f"Testing set size: {y_test.shape} samples")      # Testing size

# using scalar on train and test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the saved model
model = joblib.load('best_heart_disease_model.joblib')

# Create a sample input DataFrame
sample_input = X_test.iloc[[0]] 

print("\n--- Sample Patient Data ---")
print(sample_input)

# CRITICAL STEP: We must scale this single sample using the scaler we fitted earlier
sample_scaled = scaler.transform(sample_input)

# Make Prediction
prediction = model.predict(sample_scaled)

# Make Probability Prediction (shows confidence percentage)
# Note: specific to algorithms like Logistic Regression, Random Forest, etc.
if hasattr(model, "predict_proba"):
    probability = model.predict_proba(sample_scaled)
    confidence = probability[0][prediction[0]] * 100
else:
    confidence = None

# Output the single result
print("\n--- Prediction Result ---")
if prediction[0] == 1:
    print(f"Prediction: Heart Disease DETECTED")
else:
    print(f"Prediction: NO Heart Disease Detected")

if confidence:
    print(f"Confidence: {confidence:.2f}%")


print("\n--- Overall Model Validation ---")
# Use the model to predict on the full scaled test set
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy of loaded model on Test Data: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# 1. PRINT COLUMNS FIRST (To be absolutely sure)
# This will show you the exact spelling needed for the dictionary keys
print("Your Dataset Columns:", list(X.columns))

# 2. UPDATED DICTIONARY 
# I have renamed the keys to match the error log you provided.
new_patient_data = {
    'age': 57,
    'sex': 1,                    
    'chest pain type': 0,        # Renamed from 'cp'
    'resting bp s': 140,         # Renamed from 'trestbps'
    'cholesterol': 241,          # Renamed from 'chol'
    'fasting blood sugar': 0,    # Renamed from 'fbs'
    'resting ecg': 1,            # Renamed from 'restecg'
    'max heart rate': 123,       # Renamed from 'thalach'
    'exercise angina': 1,        # Renamed from 'exang'
    'oldpeak': 0.2,              # seemingly correct
    'ST slope': 1,               # Renamed from 'slope'
    # Note: If your dataset also has 'ca' or 'thal', keep them. 
    # If your dataset DOES NOT have them, remove the two lines below:
    'ca': 0,           
    'thal': 3          
}

# 3. CONVERT TO DATAFRAME
input_df = pd.DataFrame([new_patient_data])

# 4. ALIGN COLUMNS
# This line caused the error before, but now the names should match
input_df = input_df[X.columns]

print("\n--- Input DataFrame (Aligned) ---")
print(input_df)

# # 5. SCALE AND PREDICT
# input_scaled = scaler.transform(input_df)
# prediction = model.predict(input_scaled)

# print(f"\nPrediction: {prediction[0]}")

# 5. SCALE AND PREDICT
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

# Calculate Confidence (Probability)
if hasattr(model, "predict_proba"):
    probability = model.predict_proba(input_scaled)
    confidence = probability[0][prediction[0]] * 100
else:
    confidence = None

print("\n--- Final Manual Prediction Result ---")
if prediction[0] == 1:
    print(f"Result: Heart Disease DETECTED (Class 1)")
    print("Action: Consult a doctor immediately.")
else:
    print(f"Result: NO Heart Disease Detected (Class 0)")
    print("Action: Patient appears healthy based on model parameters.")

if confidence:
    print(f"Confidence: {confidence:.2f}%")