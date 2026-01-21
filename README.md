# Heart Disease Prediction Web Application using Machine Learning

## 📌 Project Overview
This project is an end-to-end **Machine Learning–based Heart Disease Prediction Web Application** developed using **Python, Flask, and K-Nearest Neighbors (KNN)** algorithm.  
The application predicts the presence of heart disease based on selected clinical attributes provided by the user through a web interface.

---

## 🎯 Objective
The main objective of this project is to:
- Analyze key cardiovascular health parameters
- Build and optimize a machine learning classification model
- Deploy the trained model using Flask for real-time predictions
- Assist in early detection of heart disease risk

---

## 🧠 Machine Learning Model
- **Algorithm Used:** K-Nearest Neighbors (KNN)
- **Optimization:** Best value of *K* selected using 10-fold cross-validation
- **Reason for Selection:**
  - Simple and effective for medical classification tasks
  - Performs well on scaled numerical features
- **Library:** scikit-learn

---

## 📊 Dataset Description
The dataset (`heart_final.csv`) contains clinical attributes related to heart health.

### Selected Features:
- ST Slope
- Exercise Angina
- Chest Pain Type
- Maximum Heart Rate

### Target Variable:
- **target** (0 – No Heart Disease, 1 – Heart Disease)

---

## ⚙️ Data Preprocessing
- Feature scaling performed using **MinMaxScaler**
- Ensures all features contribute equally to distance-based KNN model
- Data split into training and testing sets (70:30 ratio)

---

## 🧪 Model Training and Evaluation
- Cross-validation performed to select optimal *K* value
- Model evaluated using:
  - Accuracy Score
  - Classification Report
  - Confusion Matrix
- Achieves reliable performance on unseen test data

---

## 🌐 Web Application (Flask)
The Flask application provides:
- User-friendly web interface for data input
- Real-time heart disease prediction
- Display of prediction results on the same page

### Flask Routes:
- `/` → Home page
- `/predict` → Handles prediction logic and form submission

---

## 🖥️ Tech Stack Used
- **Programming Language:** Python
- **Web Framework:** Flask
- **Machine Learning:** scikit-learn
- **Data Processing:** Pandas
- **Feature Scaling:** MinMaxScaler
- **Frontend:** HTML (Jinja2 Templates)

---

## 📁 Project Structure
- heart-disease-prediction-flask-ml/
- │
- ├── app.py # Flask application
- ├── heart_final.csv # Dataset
- ├── templates/
- │ └── webpage.html # Frontend HTML page
- └── README.md # Project documentation
