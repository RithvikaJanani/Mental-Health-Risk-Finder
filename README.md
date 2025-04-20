# Mental Health Risk Predictor

A Streamlit web app that predicts the risk of mental health issues based on personal and workplace-related factors. The prediction is made using a trained machine learning model.

## Features
- Takes user inputs on age, gender, workplace conditions, mental health support, and more.
- Predicts mental health risk level (High / Low).
- Displays prediction confidence.
- Clean UI powered by Streamlit.

## Files in this repo
- `app.py` – main Streamlit app.
- `mental_health_model.pkl` – trained machine learning model.
- `encoders.pkl` – label encoders for categorical data.
- `cleaned.csv` & `survey.csv` – dataset files.
- `riskpredictor.ipynb` – model training and preprocessing.

## How to Run
Make sure you have `streamlit`, `pandas`, `joblib`, `matplotlib` installed.

```bash
pip install streamlit pandas joblib matplotlib
