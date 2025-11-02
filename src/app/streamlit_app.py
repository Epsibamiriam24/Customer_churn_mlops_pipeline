import streamlit as st
import pandas as pd
import joblib
import sys
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime

# Add the src directory to the path so we can import our modules
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from train import (
    prepare_target, convert_last_interaction, drop_identifier_columns,
    run_train, detect_feature_types, infer_target_column
)

# Set page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔄",
    layout="wide"
)

# Initialize MLflow
mlflow.set_tracking_uri("file:" + str(Path(src_dir).parent / "mlruns"))

# Title and Navigation
st.title("Customer Churn Prediction")
tab1, tab2, tab3 = st.tabs(["Predict", "Train Model", "Model Performance"])

with tab1:
    st.write("Upload customer data to predict churn probability")

# Load the model
@st.cache_resource
def load_model():
    model_path = Path(src_dir).parent / "artifacts" / "model.joblib"
    if not model_path.exists():
        st.error("Model not found! Please train the model first.")
        return None
    return joblib.load(model_path)

model_data = load_model()

if model_data is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load and display the data
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Process the data
    if st.button("Predict Churn"):
        try:
            # Preprocess the data
            df = drop_identifier_columns(df)
            df = convert_last_interaction(df)
            
            # Get predictions
            predictions = model_data["model"].predict(df)
            prob_predictions = model_data["model"].predict_proba(df)
            
            # Add predictions to dataframe
            results = pd.DataFrame({
                "Churn Probability": prob_predictions[:, 1],
                "Prediction": ["Churn" if p == 1 else "No Churn" for p in predictions]
            })
            
            # Display results
            st.subheader("Prediction Results")
            st.dataframe(results)
            
            # Show distribution of predictions
            st.subheader("Prediction Distribution")
            st.bar_chart(results["Prediction"].value_counts())
            
            # Download button for predictions
            st.download_button(
                label="Download Predictions",
                data=results.to_csv(index=False),
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Show model information
st.sidebar.header("Model Information")
if model_data.get("accuracy") is not None:
    st.sidebar.metric("Model Accuracy", f"{model_data['accuracy']:.2%}")

st.sidebar.subheader("Feature Information")
st.sidebar.write("Numeric Features:", model_data["numeric_cols"])
st.sidebar.write("Categorical Features:", model_data["categorical_cols"])

# Instructions
with st.expander("How to use"):
    st.write("""
    1. Upload a CSV file containing customer data
    2. The file should contain the same features used during training
    3. Click 'Predict Churn' to get predictions
    4. Download the predictions using the download button
    """)