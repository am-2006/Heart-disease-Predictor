# Created by Aman Kumar Choudhary
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Load dataset
data = pd.read_csv('Heart_Disease_Prediction.csv')

# Check for non-numeric columns
print(data.dtypes)

# Encode categorical columns (target variable becomes "Heart Disease_Presence")
data = pd.get_dummies(data, drop_first=True)

# Verify column names after encoding
print(data.columns)

# Ensure correct target column name
target_column = 'Heart Disease_Presence'

# Define features (X) and target (y)
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained model and scaler
model.save('model.h5')
print("✅ Model saved as 'model.h5'")

joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved as 'scaler.pkl'")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Predictions and metrics
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
