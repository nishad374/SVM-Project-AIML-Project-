# ❤️ Heart Disease Prediction using SVM

This project is a web application that predicts the probability of heart disease in a patient using a **Support Vector Machine (SVM)** model. It is built with **Streamlit** for the frontend and **scikit-learn** for machine learning.

## 🚀 Features

- Clean and interactive user interface with Streamlit
- Inputs include age, blood pressure, cholesterol, ECG, and more
- Predicts the **risk level (Low, Moderate, High)** using a trained SVM classifier
- Generates a **PDF medical report** for download
- Easy to deploy and customize

## 📦 Technologies Used

- Python
- Scikit-learn (SVM)
- Pandas & NumPy
- Streamlit
- FPDF (for PDF generation)
- Joblib (for model serialization)

## 🧠 Machine Learning

The model is trained using the **Cleveland Heart Disease dataset**. It uses a combination of numerical and categorical features, which are preprocessed using `StandardScaler` and `OneHotEncoder`.

## 📋 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
