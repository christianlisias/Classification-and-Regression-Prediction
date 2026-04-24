import streamlit as st
import joblib
import numpy as np
import pandas as pd

model_class = joblib.load('classification_pipeline.pkl')
model_reg = joblib.load('regression_pipeline.pkl')

def main():
    st.title('Placement & Salary Prediction Deployment')

    gender_raw = st.radio("Gender", ["Male", "Female"])
    ssc_percentage = st.number_input("SSC (10th) Percentage", 0.0, 100.0, 75.0)
    hsc_percentage = st.number_input("HSC (12th) Percentage", 0.0, 100.0, 75.0)
    degree_percentage = st.number_input("Degree Percentage", 0.0, 100.0, 70.0)
    cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0)
    entrance_exam_score = st.number_input("Entrance Exam Score", 0.0, 100.0, 80.0)
    technical_skill_score = st.number_input("Technical Skill Score", 0.0, 100.0, 75.0)
    soft_skill_score = st.number_input("Soft Skill Score", 0.0, 100.0, 75.0)
    internship_count = st.number_input("Internship Count", 0, 10, 1)
    live_projects = st.number_input("Live Projects", 0, 10, 1)
    work_experience_months = st.number_input("Work Experience (Months)", 0, 60, 6)
    certifications = st.number_input("Certifications Count", 0, 10, 2)
    attendance_percentage = st.number_input("Attendance Percentage", 0.0, 100.0, 85.0)
    backlogs = st.number_input("Backlogs Count", 0, 10, 0)
    extra_raw = st.radio("Extracurricular Activities", ["Yes", "No"])

    gender = 1 if gender_raw == "Male" else 0
    extracurricular_activities = 1 if extra_raw == "Yes" else 0

    data = {
        'gender': gender,
        'ssc_percentage': ssc_percentage,
        'hsc_percentage': hsc_percentage,
        'degree_percentage': degree_percentage,
        'cgpa': cgpa,
        'entrance_exam_score': entrance_exam_score,
        'technical_skill_score': technical_skill_score,
        'soft_skill_score': soft_skill_score,
        'internship_count': internship_count,
        'live_projects': live_projects,
        'work_experience_months': work_experience_months,
        'certifications': certifications,
        'attendance_percentage': attendance_percentage,
        'backlogs': backlogs,
        'extracurricular_activities': extracurricular_activities,
        'student_id': 0
    }

    df = pd.DataFrame([data])

    if st.button("Make Prediction"):
        # Classification
        prediction_class = model_class.predict(df)[0]
        status = "Placed" if int(prediction_class) == 1 else "Not Placed"
        
        # Regression
        prediction_reg = model_reg.predict(df)[0]
        
        st.markdown("---")
        st.subheader("Results:")
        st.success(f"Placement Status: **{status}**")
        
        if int(prediction_class) == 1:
            st.success(f"Estimated Salary Package: **{prediction_reg:.2f} LPA**")
        else:
            st.info("Estimated Salary Package: **0.00 LPA** (Not Placed)")

if __name__ == "__main__":
    main()