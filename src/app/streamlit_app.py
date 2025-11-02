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
    page_title="Customer Churn Predictor - Auto Deploy Test",
    page_icon="✨",
    layout="wide"
)

# Initialize MLflow
mlflow.set_tracking_uri("file:" + str(Path(src_dir).parent / "mlruns"))

def get_recommendation(churn_prob, contract_length, total_spend, tenure):
    if churn_prob < 0.3:
        return "Customer appears stable. Consider upselling premium services."
    elif churn_prob < 0.6:
        recommendations = []
        if contract_length == "Monthly":
            recommendations.append("Offer annual contract with discount")
        if total_spend > 500:
            recommendations.append("Review current plan for cost-saving opportunities")
        if tenure < 12:
            recommendations.append("Provide early loyalty rewards")
        return " • " + "\n • ".join(recommendations) if recommendations else "Monitor customer satisfaction"
    else:
        return "⚠️ High churn risk! Immediate retention actions needed:\n • Contact customer for feedback\n • Offer personalized retention package\n • Consider service upgrades or discounts"

# Load the model first
@st.cache_resource
def load_model():
    try:
        # Try loading from artifacts directory first (Docker container)
        model_path = Path("artifacts") / "model.joblib"
        if model_path.exists():
            return joblib.load(model_path)
        
        # Fallback to development path
        model_path = Path(src_dir).parent / "artifacts" / "model.joblib"
        if model_path.exists():
            return joblib.load(model_path)
        
        st.error("Model not found! Please train the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model_data = load_model()

# Title and Navigation
st.title("Customer Churn Prediction")
tab1, tab2, tab3 = st.tabs(["Predict", "Train Model", "Model Performance"])

with tab1:
    st.write("Predict customer churn probability")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Single Customer Prediction")
        with st.form("customer_form"):
            # Customer details
            st.write("Enter Customer Details:")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)
            usage_frequency = st.number_input("Usage Frequency", min_value=0, max_value=100, value=10)
            support_calls = st.number_input("Support Calls", min_value=0, max_value=50, value=0)
            payment_delay = st.number_input("Payment Delay", min_value=0, max_value=90, value=0)
            
            # Subscription and Contract
            st.write("Subscription and Contract Details:")
            subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
            total_spend = st.number_input("Total Spend", min_value=0.0, value=100.0)
            days_since_last_interaction = st.number_input("Last Interaction", min_value=0, max_value=365, value=7)
            
            submitted = st.form_submit_button("Predict Churn")
            
            if submitted:
                try:
                    # Create a DataFrame with the input data
                    input_data = pd.DataFrame({
                        'CustomerID': [1],  # Dummy ID that will be dropped
                        'Age': [age],
                        'Gender': [gender],
                        'Tenure': [tenure],
                        'Usage Frequency': [usage_frequency],
                        'Support Calls': [support_calls],
                        'Payment Delay': [payment_delay],
                        'Subscription Type': [subscription_type],
                        'Contract Length': [contract_length],
                        'Total Spend': [total_spend],
                        'Last Interaction': [days_since_last_interaction]
                    })
                    
                    # Encode categorical variables using saved label encoders
                    for col, le in model_data.get("label_encoders", {}).items():
                        if col in input_data.columns:
                            input_data[col] = le.transform(input_data[col])
                    
                    # Make prediction
                    prob = model_data["model"].predict_proba(input_data)[0, 1]
                    prediction = "Likely to Churn" if prob > 0.5 else "Likely to Stay"
                    
                    # Show prediction with nice formatting
                    st.success("Prediction Complete!")
                    st.markdown(f"""
                    ### Results:
                    - **Prediction:** {prediction}
                    - **Churn Probability:** {prob:.1%}
                    
                    #### Recommendation:
                    {get_recommendation(prob, contract_length, total_spend, tenure)}
                    """)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    with col2:
        st.subheader("Batch Predictions")
        st.write("Upload a CSV file for batch predictions")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_predictions")
        
        if uploaded_file is not None:
            try:
                # Load and display the data
                df = pd.read_csv(uploaded_file)
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                if st.button("Predict Churn for All Customers"):
                    # Preprocess the data
                    df_processed = df.copy()
                    if 'CustomerID' in df_processed.columns:
                        customer_ids = df_processed['CustomerID']
                        df_processed = drop_identifier_columns(df_processed)
                    df_processed = convert_last_interaction(df_processed)
                    
                    # Get predictions
                    predictions = model_data["model"].predict(df_processed)
                    prob_predictions = model_data["model"].predict_proba(df_processed)
                    
                    # Create results DataFrame
                    results = pd.DataFrame({
                        "Churn Probability": prob_predictions[:, 1],
                        "Prediction": ["Likely to Churn" if p > 0.5 else "Likely to Stay" for p in prob_predictions[:, 1]]
                    })
                    
                    if 'CustomerID' in df.columns:
                        results.insert(0, 'CustomerID', customer_ids)
                    
                    # Display results
                    st.success(f"Processed {len(df)} customers")
                    st.markdown(f"""
                    ### Summary:
                    - Total Customers: {len(df)}
                    - Likely to Churn: {sum(prob_predictions[:, 1] > 0.5)}
                    - Likely to Stay: {sum(prob_predictions[:, 1] <= 0.5)}
                    """)
                    
                    # Display detailed results
                    st.subheader("Detailed Predictions")
                    st.dataframe(results)
                    
                    # Download button for predictions
                    st.download_button(
                        label="Download Predictions",
                        data=results.to_csv(index=False),
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Show distribution of predictions
                    st.subheader("Prediction Distribution")
                    st.bar_chart(results["Prediction"].value_counts())
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

with tab2:
    st.header("Train New Model")
    
    # File uploaders for training data
    train_file = st.file_uploader("Upload Training Data (CSV)", type="csv", key="train")
    test_file = st.file_uploader("Upload Test Data (CSV)", type="csv", key="test")
    
    if train_file and test_file:
        if st.button("Train New Model"):
            try:
                # Load data
                train_data = pd.read_csv(train_file)
                test_data = pd.read_csv(test_file)
                
                # Train model
                model_info = run_train(train_data, test_data)
                
                # Save model
                model_path = Path("artifacts") / "model.joblib"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model_info, model_path)
                
                st.success("Model trained successfully! You can now use it for predictions.")
                st.experimental_rerun()  # Reload the app to use new model
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

with tab3:
    st.header("Model Performance")
    model = load_model()
    
    if model is not None and isinstance(model, dict):
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{model.get('accuracy', 0):.2%}")
        with col2:
            st.metric("Precision", f"{model.get('precision', 0):.2%}")
        with col3:
            st.metric("Recall", f"{model.get('recall', 0):.2%}")
        
        # Display confusion matrix if available
        if 'confusion_matrix' in model:
            st.subheader("Confusion Matrix")
            cm = model['confusion_matrix']
            cm_df = pd.DataFrame(
                cm,
                index=['Actual No Churn', 'Actual Churn'],
                columns=['Predicted No Churn', 'Predicted Churn']
            )
            st.dataframe(cm_df)

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