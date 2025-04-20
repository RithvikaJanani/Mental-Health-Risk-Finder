import streamlit as st
import pandas as pd
import joblib

model = joblib.load('mental_health_model.pkl')
encoders = joblib.load('encoders.pkl')

st.set_page_config(page_title="Mental Health Risk Predictor", page_icon="🧠")
st.title("Mental Health Risk Predictor")
st.markdown("Use this tool to assess your potential **mental health risk** based on your work environment and personal factors.")

# ------------------- Personal Info -------------------
st.markdown("### 👤 Personal Information")
age = st.slider("Age", 18, 70)
gender = st.selectbox("Gender", ["male", "female", "other"])
self_employed = st.selectbox("Self-Employed?", ["Yes", "No"])
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

# ------------------- Work Info -------------------
st.markdown("### 💼 Workplace Environment")
work_interfere = st.selectbox("How often does mental health interfere with work?", ["Often", "Rarely", "Never", "Sometimes"])
no_employees = st.selectbox("Company Size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
tech_company = st.selectbox("Do you work in a tech company?", ["Yes", "No"])
leave = st.selectbox("Ease of leave for mental health reasons", 
                     ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"])

# ------------------- Support & Benefits -------------------
st.markdown("### 🧑‍⚕️ Support & Benefits")
benefits = st.selectbox("Mental Health Benefits Provided", ["Yes", "No", "Don't know"])
care_options = st.selectbox("Care Options Available", ["Yes", "No", "Not sure"])
wellness_program = st.selectbox("Wellness Program", ["Yes", "No", "Don't know"])
seek_help = st.selectbox("Encouraged to Seek Help", ["Yes", "No", "Don't know"])
anonymity = st.selectbox("Is anonymity guaranteed?", ["Yes", "No", "Don't know"])

# ------------------- Workplace Conversations -------------------
st.markdown("### 💬 Conversations & Disclosure")
mental_health_consequence = st.selectbox("Consequences for discussing mental health", ["Yes", "No", "Maybe"])
phys_health_consequence = st.selectbox("Consequences for discussing physical health", ["Yes", "No", "Maybe"])
coworkers = st.selectbox("Comfortable discussing with coworkers", ["Yes", "No", "Some of them"])
supervisor = st.selectbox("Comfortable discussing with supervisor", ["Yes", "No", "Some of them"])
mental_health_interview = st.selectbox("Would disclose mental health in interview?", ["Yes", "No", "Maybe"])
phys_health_interview = st.selectbox("Would disclose physical health in interview?", ["Yes", "No", "Maybe"])
mental_vs_physical = st.selectbox("Is mental health as important as physical health?", ["Yes", "No", "Don't know"])
obs_consequence = st.selectbox("Have you seen consequences for others?", ["Yes", "No"])

# ------------------- Preprocessing -------------------
input_df = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'self_employed': self_employed,
    'family_history': family_history,
    'work_interfere': work_interfere,
    'no_employees': no_employees,
    'remote_work': remote_work,
    'tech_company': tech_company,
    'benefits': benefits,
    'care_options': care_options,
    'wellness_program': wellness_program,
    'seek_help': seek_help,
    'anonymity': anonymity,
    'leave': leave,
    'mental_health_consequence': mental_health_consequence,
    'phys_health_consequence': phys_health_consequence,
    'coworkers': coworkers,
    'supervisor': supervisor,
    'mental_health_interview': mental_health_interview,
    'phys_health_interview': phys_health_interview,
    'mental_vs_physical': mental_vs_physical,
    'obs_consequence': obs_consequence
}])

# Convert employee range to approximate value
input_df['no_employees'] = input_df['no_employees'].replace({
    '1-5': 3,
    '6-25': 15,
    '26-100': 63,
    '100-500': 300,
    '500-1000': 750,
    'More than 1000': 1200
})

# Encode categorical features
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

# ------------------- Prediction -------------------
if st.button("🔍 Predict Mental Health Risk"):
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]  # [low_risk, high_risk]

        st.markdown("### 🧾 **Prediction Result**")
        if prediction == 1:
            st.error(f"""
            ## High Risk Detected  
            **Confidence:** {proba[1]:.2%}

            Based on your responses, there is a high probability of mental health risk.  
            It may be helpful to speak with a mental health professional or seek support.
            """)
        else:
            st.success(f"""
            ## Low Risk Detected  
            **Confidence:** {proba[0]:.2%}

            Great! Your responses indicate a low risk of mental health concerns.  
            Continue taking care of your mental well-being.
            """)

    except Exception as e:
        st.warning(f"⚠️ Something went wrong: {e}")
