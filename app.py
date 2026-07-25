import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Campus Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Campus Placement Prediction")
st.write("Predict whether a student is likely to get placed.")

st.divider()

cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

internships = st.number_input(
    "Number of Internships",
    min_value=0,
    max_value=10,
    value=1
)

aptitude = st.slider(
    "Aptitude Test Score",
    0,
    100,
    70
)

communication = st.slider(
    "Communication Skills (1-10)",
    1,
    10,
    7
)

projects = st.number_input(
    "Number of Projects",
    min_value=0,
    max_value=20,
    value=2
)

st.divider()

if st.button("Predict Placement"):

    features = np.array([[
        cgpa,
        internships,
        aptitude,
        communication,
        projects
    ]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("✅ Prediction: Student is likely to be Placed.")
    else:
        st.error("❌ Prediction: Student is Not Likely to be Placed.")

    st.subheader("Input Summary")

    st.write(f"CGPA: **{cgpa}**")
    st.write(f"Internships: **{internships}**")
    st.write(f"Aptitude Score: **{aptitude}**")
    st.write(f"Communication Skills: **{communication}/10**")
    st.write(f"Projects: **{projects}**")