📊 Heart Disease Prediction App
An AI-powered web application for predicting the likelihood of heart disease using a machine-learning model built with TensorFlow and Streamlit.

📌 Features
Predicts the presence of heart disease based on medical parameters.
User-friendly web interface powered by Streamlit.
Accurate classification using a Deep Learning model.
Visualizes prediction results and model performance.
🛠️ Tech Stack
Python 3.11.9
TensorFlow – Deep Learning Model
Streamlit – Web Application
Pandas, NumPy, Scikit-learn – Data Processing
📂 Project Structure
bash
Copy
Edit
📦 Heart_Disease_Prediction
 ├── 📊 Heart_Disease_Prediction.csv  # Dataset
 ├── 📜 model.py                      # Model Training Script
 ├── 📜 app.py                        # Streamlit App
 ├── 📜 requirements.txt              # Python Packages
 └── 📜 README.md                     # Project Documentation
🚀 Getting Started
1. Clone the Repository:
bash
Copy
Edit
git clone https://github.com/yourusername/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
2. Set Up the Environment:
Ensure you have Python 3.11.9 and virtualenv installed.

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Train the Model:
bash
Copy
Edit
python model.py
4. Run the Streamlit App:
bash
Copy
Edit
streamlit run app.py
📊 Input Features
The model predicts heart disease based on the following parameters:

Age: Age of the patient
Sex: Gender (1 = Male, 0 = Female)
Chest Pain Type: (0–3, different types of pain)
BP: Blood Pressure
Cholesterol: Cholesterol Level
FBS over 120: Fasting Blood Sugar (1 = Yes, 0 = No)
EKG Results: Electrocardiographic results (0–2)
Max HR: Maximum Heart Rate
Exercise Angina: (1 = Yes, 0 = No)
ST Depression: Depression induced by exercise
Slope of ST: Slope of the ST segment (0–2)
Number of vessels fluro: Number of major vessels (0–3)
Thallium: Thallium stress test result (1–3)
📈 Model Performance
Accuracy: 88.89%
Precision/Recall/F1-Score: See model evaluation output.
🤝 Contributions
Contributions are welcome! Feel free to fork the repo and submit a pull request.

📧 Contact
For questions or support, contact Aman Kumar Choudhary.  krchoudharyaman@gmail.com
