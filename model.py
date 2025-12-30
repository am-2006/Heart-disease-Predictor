import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. LOAD DATA
# Ensure your CSV has 13 feature columns and 1 target column
try:
    data = pd.read_csv('Heart_Disease_Prediction.csv')
except FileNotFoundError:
    print("Error: Heart_Disease_Prediction.csv not found.")
    exit()

# 2. PREPROCESSING
# Identify target. Adjust 'Heart Disease_Presence' if your column name differs
target_column = 'Heart Disease' 
if 'Heart Disease' not in data.columns:
    # Fallback to look for similar names if exact match fails
    target_column = [col for col in data.columns if 'Heart' in col or 'Presence' in col][0]

X = data.drop(target_column, axis=1)
y = data[target_column]

# If target is text (e.g., 'Presence', 'Absence'), convert to 0 and 1
if y.dtype == 'object':
    y = y.map({'Absence': 0, 'Presence': 1})

# 3. SPLIT AND SCALE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# This scaler is the "Key" to fixing your "Always High Risk" error
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. MODEL ARCHITECTURE (Matching your uploaded .h5 file)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(13,)), # Exactly 13 features
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# 5. TRAINING
print("Starting training...")
model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=16, 
    validation_split=0.1,
    verbose=1
)

# 6. SAVE ASSETS
model.save('heart_disease_model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("\n" + "="*30)
print("✅ SUCCESS: Model and Scaler saved.")
print("Files generated: 'heart_disease_model.h5' and 'scaler.pkl'")
print("="*30)

# 7. EVALUATION
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
