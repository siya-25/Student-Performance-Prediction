import streamlit as st
import pandas as pd
from student_performance_predictor import StudentPerformancePredictor
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Title and description
    st.title("📚 Student Performance Prediction System")
    st.markdown("""
    This application predicts student GPA based on various factors and provides a risk assessment.
    Fill in the form below to get a prediction.
    """)

    # Initialize predictor
    try:
        predictor = StudentPerformancePredictor(r"C:\Users\siyap\OneDrive\Desktop\FDS-3\student_performance_data.csv")
        predictor.load_and_preprocess_data()
        predictor.train_and_evaluate_models()
    except Exception as e:
        st.error(f"Error initializing the predictor: {str(e)}")
        return

    # Create columns for form layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Student Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=30, value=20)
        major = st.selectbox("Major", ["Arts", "Science", "Business", "Education"])
        
    with col2:
        st.subheader("Academic Information")
        study_hours = st.slider("Study Hours per Week", 0, 40, 20)
        attendance = st.slider("Attendance Rate (%)", 0, 100, 75)
        part_time_job = st.selectbox("Part-Time Job", ["Yes", "No"])
        extra_curricular = st.selectbox("Extra Curricular Activities", ["Yes", "No"])

    # Prediction button
    if st.button("Predict Performance", type="primary"):
        try:
            # Prepare features
            features = {
                'Gender': gender,
                'Age': age,
                'StudyHoursPerWeek': study_hours,
                'AttendanceRate': attendance,
                'Major': major,
                'PartTimeJob': part_time_job,
                'ExtraCurricularActivities': extra_curricular
            }

            # Get prediction
            result = predictor.predict(features)
            
            # Display results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted GPA", f"{result['prediction']:.2f}")
                
            with col2:
                risk_color = {
                    "low": "green",
                    "medium": "orange",
                    "high": "red"
                }[result['risk_level']]
                
                st.markdown(f"""
                <div style='background-color: {risk_color}20; padding: 20px; border-radius: 10px;'>
                    <h3 style='color: {risk_color}; margin:0;'>
                        {result['risk_level'].upper()} RISK
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.metric("Confidence", f"{result['confidence']:.1f}%")

            # Display model metrics
            st.subheader("Model Performance Metrics")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("R² Score", f"{result['metrics']['R2']:.4f}")
            with metrics_col2:
                st.metric("RMSE", f"{result['metrics']['RMSE']:.4f}")
            with metrics_col3:
                st.metric("MAE", f"{result['metrics']['MAE']:.4f}")

            # Add recommendations based on risk level
            st.subheader("Recommendations")
            if result['risk_level'] == "high":
                st.warning("""
                - Increase study hours significantly
                - Improve attendance rate
                - Consider reducing external commitments
                - Seek academic counseling
                """)
            elif result['risk_level'] == "medium":
                st.info("""
                - Maintain current study habits
                - Look for areas of improvement
                - Consider joining study groups
                - Regular attendance is important
                """)
            else:
                st.success("""
                - Excellent performance! Keep up the good work
                - Consider mentoring other students
                - Balance academics with other activities
                - Set higher academic goals
                """)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    # Add information about the model
    with st.expander("About the Model"):
        st.markdown("""
        This prediction system uses machine learning to estimate student GPA based on various factors:
        - Personal factors (Age, Gender)
        - Academic engagement (Study Hours, Attendance)
        - External activities (Part-Time Job, Extra Curricular Activities)
        - Field of study (Major)
        
        The system uses multiple models including Random Forest, Gradient Boosting, and Neural Networks,
        selecting the best performing model for predictions.
        """)

if __name__ == "__main__":
    main() 