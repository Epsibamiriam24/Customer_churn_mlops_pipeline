import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path
from datetime import datetime

# Optional mlflow import
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

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
    page_icon="🎯",
    layout="wide"
)

# Initialize MLflow (if available)
if MLFLOW_AVAILABLE:
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
st.title("Customer Churn Prediction 🎯")
st.caption("Version 2.0 - Updated UI with Complete Feature Set | Deployed: Nov 3, 2025")
tab1, tab2, tab3 = st.tabs(["Predict", "Train Model", "Model Performance"])

with tab1:
    st.write("### 🔮 Predict Customer Churn Probability")
    st.write("Enter customer details below to predict churn risk")
    
    # Show model accuracy warning if low
    if model_data and model_data.get("accuracy"):
        if model_data["accuracy"] < 0.70:
            st.warning(f"⚠️ Note: Current model accuracy is {model_data['accuracy']:.1%}. Consider retraining the model for better predictions.")
        else:
            st.info(f"✓ Model accuracy: {model_data['accuracy']:.1%}")
    
    if model_data is None:
        st.error("❌ Model not loaded! Please train the model first in the 'Train Model' tab.")
    else:
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Customer Information")
            with st.form("customer_form"):
                # Primary inputs (highlighted)
                st.markdown("#### 🎯 Key Financial Metrics")
                tenure = st.number_input(
                    "Tenure (months with company)", 
                    min_value=0, 
                    max_value=100, 
                    value=12,
                    help="Number of months the customer has been with the company"
                )
                
                # Calculate MonthlyCharges from Total Spend and Tenure
                total_spend = st.number_input(
                    "Total Spend ($)", 
                    min_value=0.0, 
                    value=500.0,
                    step=50.0,
                    help="Total amount spent by the customer"
                )
                
                # Display calculated Monthly Charges
                monthly_charges = total_spend / tenure if tenure > 0 else 0
                st.metric("Calculated Monthly Charges", f"${monthly_charges:.2f}")
                
                st.markdown("---")
                
                # Other customer details
                st.markdown("#### 👤 Demographics")
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                gender = st.selectbox("Gender", ["male", "female"])
                
                st.markdown("#### 📱 Usage & Support")
                usage_frequency = st.slider("Usage Frequency (per month)", min_value=0, max_value=50, value=10)
                support_calls = st.slider("Support Calls (last month)", min_value=0, max_value=20, value=2)
                payment_delay = st.slider("Payment Delay (days)", min_value=0, max_value=90, value=0)
                
                st.markdown("#### 📋 Subscription Details")
                subscription_type = st.selectbox("Subscription Type", ["basic", "standard", "premium"])
                contract_length = st.selectbox("Contract Length", ["monthly", "quarterly", "annual"])
                days_since_last_interaction = st.number_input(
                    "Days Since Last Interaction", 
                    min_value=0, 
                    max_value=30000, 
                    value=30,
                    help="Number of days since the last customer interaction"
                )
                
                submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)
                
        with col2:
            st.subheader("📈 Prediction Results")
            
            if submitted:
                try:
                    # Create a DataFrame with the input data in the exact order and format
                    # that the model expects based on the training data
                    input_data = pd.DataFrame({
                        'Age': [float(age)],
                        'Gender': [gender.lower()],  # Ensure lowercase
                        'Tenure': [float(tenure)],
                        'Usage Frequency': [float(usage_frequency)],
                        'Support Calls': [float(support_calls)],
                        'Payment Delay': [float(payment_delay)],
                        'Subscription Type': [subscription_type.lower()],  # Ensure lowercase
                        'Contract Length': [contract_length.lower()],  # Ensure lowercase
                        'Total Spend': [float(total_spend)],
                        'Last Interaction': [float(days_since_last_interaction)]
                    })
                    
                    # Debug: Show the input data
                    with st.expander("🔍 Debug: Input Data"):
                        st.write("**Input DataFrame:**")
                        st.dataframe(input_data)
                        st.write("**Data Types:**")
                        st.write(input_data.dtypes)
                        st.write("**Expected Features:**")
                        st.write(f"Numeric: {model_data['numeric_cols']}")
                        st.write(f"Categorical: {model_data['categorical_cols']}")
                    
                    # Make prediction
                    prediction = model_data["model"].predict(input_data)[0]
                    prob_array = model_data["model"].predict_proba(input_data)[0]
                    prob = prob_array[1]  # Probability of churn (class 1)
                    
                    # Debug: Show raw prediction
                    with st.expander("🔍 Debug: Prediction Details"):
                        st.write(f"**Raw Prediction:** {prediction}")
                        st.write(f"**Probability Array:** {prob_array}")
                        st.write(f"**P(No Churn):** {prob_array[0]:.4f}")
                        st.write(f"**P(Churn):** {prob_array[1]:.4f}")
                    
                    # Display results with visual indicators
                    # Prediction: 0 = No Churn (Negative), 1 = Churn (Positive)
                    if prediction == 0:
                        st.success("✅ CUSTOMER WILL LIKELY STAY (Predicted: No Churn)")
                        prediction_label = "No Churn (Negative)"
                    else:
                        st.error("⚠️ CUSTOMER WILL LIKELY CHURN (Predicted: Churn)")
                        prediction_label = "Churn (Positive)"
                    
                    # Risk level based on probability
                    if prob > 0.7:
                        risk_level = "High"
                        color = "red"
                    elif prob > 0.4:
                        risk_level = "Medium"
                        color = "orange"
                    else:
                        risk_level = "Low"
                        color = "green"
                    
                    # Show detailed metrics
                    st.markdown("---")
                    
                    # Create metric columns
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Prediction", prediction_label)
                    with metric_col2:
                        st.metric("Churn Probability", f"{prob:.1%}")
                    with metric_col3:
                        st.metric("Risk Level", risk_level)
                    
                    # Show prediction details
                    st.markdown("### 📊 Customer Summary")
                    summary_data = {
                        "Metric": [
                            "Tenure",
                            "Monthly Charges",
                            "Total Spend",
                            "Contract Type",
                            "Subscription",
                            "Usage Frequency",
                            "Support Calls"
                        ],
                        "Value": [
                            f"{tenure} months",
                            f"${monthly_charges:.2f}",
                            f"${total_spend:.2f}",
                            contract_length.capitalize(),
                            subscription_type.capitalize(),
                            f"{usage_frequency}/month",
                            f"{support_calls}"
                        ]
                    }
                    st.table(pd.DataFrame(summary_data))
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    recommendation = get_recommendation(prob, contract_length, total_spend, tenure)
                    st.info(recommendation)
                    
                    # Additional insights
                    st.markdown("### 🎯 Key Insights")
                    insights = []
                    
                    if tenure < 12:
                        insights.append("• Customer is new (tenure < 12 months) - Higher churn risk")
                    if monthly_charges > 80:
                        insights.append("• High monthly charges - Consider offering discounts")
                    if support_calls > 5:
                        insights.append("• High support calls - Indicates potential service issues")
                    if contract_length == "monthly":
                        insights.append("• Month-to-month contract - Consider offering annual contract incentives")
                    if payment_delay > 10:
                        insights.append("• Payment delays detected - Financial issues may indicate churn risk")
                    if usage_frequency < 5:
                        insights.append("• Low usage frequency - Customer may not be engaged")
                    
                    if insights:
                        for insight in insights:
                            st.write(insight)
                    else:
                        st.write("• Customer profile looks stable overall")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.write("Debug info:")
                    st.write(f"Model type: {type(model_data)}")
                    st.write(f"Model keys: {model_data.keys() if isinstance(model_data, dict) else 'N/A'}")
            else:
                st.info("👈 Fill in the customer details and click 'Predict Churn' to see results")
    
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
st.sidebar.header("ℹ️ Model Information")
if model_data and isinstance(model_data, dict):
    if model_data.get("accuracy") is not None:
        st.sidebar.metric("Model Accuracy", f"{model_data['accuracy']:.2%}")
    
    st.sidebar.subheader("📋 Required Features")
    st.sidebar.write("**Numeric:**")
    if "numeric_cols" in model_data:
        for col in model_data["numeric_cols"]:
            st.sidebar.write(f"• {col}")
    
    st.sidebar.write("**Categorical:**")
    if "categorical_cols" in model_data:
        for col in model_data["categorical_cols"]:
            st.sidebar.write(f"• {col}")
else:
    st.sidebar.warning("Model not loaded")

# Instructions
with st.expander("📖 How to use this app"):
    st.write("""
    1. **Enter customer details** in the form on the left
    2. Click **'Predict Churn'** button to get the prediction
    3. View the results including churn probability and risk level
    4. Check the **debug sections** to see detailed prediction information
    5. For batch predictions, use the file upload feature
    """)

# Input Data Interpretation Guide
with st.expander("📊 Input Data Interpretation Guide"):
    st.markdown("""
    ### Understanding Customer Input Features
    
    #### 🎯 **Key Financial Metrics** (Most Important)
    
    **1. Tenure (Months with Company)**
    - **What it means:** How long the customer has been with the company
    - **Impact on Churn:**
        - 🟢 **0-6 months:** HIGH RISK - New customers are most likely to churn
        - 🟡 **7-24 months:** MEDIUM RISK - Building loyalty
        - 🟢 **25+ months:** LOW RISK - Established, loyal customers
    - **Why it matters:** Longer tenure = stronger relationship = less likely to leave
    
    **2. Total Spend ($)**
    - **What it means:** Total amount the customer has spent with the company
    - **Impact on Churn:**
        - 🔴 **$0-$200:** HIGH RISK - Low investment in service
        - 🟡 **$201-$1000:** MEDIUM RISK - Moderate engagement
        - 🟢 **$1000+:** LOW RISK - High value customer
    - **Why it matters:** More invested = more to lose by leaving
    
    **3. Monthly Charges (Calculated)**
    - **What it means:** Average monthly cost = Total Spend ÷ Tenure
    - **Impact on Churn:**
        - 🟢 **$0-$50:** LOW RISK - Affordable for customer
        - 🟡 **$51-$100:** MEDIUM RISK - Moderate cost
        - 🔴 **$100+:** HIGH RISK - High monthly cost may drive churn
    - **Why it matters:** High costs can motivate customers to find cheaper alternatives
    
    ---
    
    #### 👤 **Demographics**
    
    **4. Age**
    - **What it means:** Customer's age in years
    - **Impact:** 
        - Older customers (40+) tend to be more stable
        - Younger customers (18-30) are more likely to switch services
    
    **5. Gender**
    - **What it means:** Customer's gender (male/female)
    - **Impact:** Minor impact on churn behavior
    
    ---
    
    #### 📱 **Usage & Support Patterns**
    
    **6. Usage Frequency (per month)**
    - **What it means:** How often the customer uses the service
    - **Impact on Churn:**
        - 🔴 **0-5:** HIGH RISK - Not engaged with service
        - 🟡 **6-15:** MEDIUM RISK - Moderate usage
        - 🟢 **16+:** LOW RISK - Active, engaged user
    - **Why it matters:** Active users are invested and less likely to leave
    
    **7. Support Calls (last month)**
    - **What it means:** Number of times customer contacted support
    - **Impact on Churn:**
        - 🟢 **0-2:** LOW RISK - Satisfied customer
        - 🟡 **3-5:** MEDIUM RISK - Some issues
        - 🔴 **6+:** HIGH RISK - Frustrated customer with problems
    - **Why it matters:** Many support calls indicate problems or dissatisfaction
    
    **8. Payment Delay (days)**
    - **What it means:** Average days late on payments
    - **Impact on Churn:**
        - 🟢 **0:** LOW RISK - Financially stable, on-time payer
        - 🟡 **1-15:** MEDIUM RISK - Occasional delays
        - 🔴 **16+:** HIGH RISK - Financial difficulties or losing interest
    - **Why it matters:** Payment issues may signal financial problems or disengagement
    
    ---
    
    #### 📋 **Subscription Details**
    
    **9. Subscription Type**
    - **What it means:** Service tier the customer subscribes to
    - **Impact on Churn:**
        - 🔴 **Basic:** HIGH RISK - Least committed tier
        - 🟡 **Standard:** MEDIUM RISK - Moderate commitment
        - 🟢 **Premium:** LOW RISK - Most invested customers
    - **Why it matters:** Premium customers have more invested and better service
    
    **10. Contract Length**
    - **What it means:** Duration of customer's contract commitment
    - **Impact on Churn:**
        - 🔴 **Monthly:** HIGH RISK - Can leave anytime, no commitment
        - 🟡 **Quarterly:** MEDIUM RISK - 3-month commitment
        - 🟢 **Annual:** LOW RISK - Locked in for 1 year
    - **Why it matters:** Longer contracts = stronger commitment = harder to leave
    
    **11. Days Since Last Interaction**
    - **What it means:** How many days since customer last engaged with service
    - **Impact on Churn:**
        - 🟢 **0-10:** LOW RISK - Recently active
        - 🟡 **11-30:** MEDIUM RISK - Moderate engagement
        - 🔴 **31+:** HIGH RISK - Disengaged, may have already left
    - **Why it matters:** Recent interaction = active customer = less likely to churn
    
    ---
    
    ### 🎯 Prediction Output Explained
    
    **Prediction Types:**
    - ✅ **No Churn (Negative/0):** Customer will likely STAY
        - This is a **True Negative** when the customer actually stays
        - Model confidence shown as probability < 50%
    
    - ⚠️ **Churn (Positive/1):** Customer will likely LEAVE
        - This is a **True Positive** when the customer actually churns
        - Model confidence shown as probability > 50%
    
    **Churn Probability:**
    - The percentage likelihood that a customer will churn
    - 0-40%: Low risk
    - 41-70%: Medium risk
    - 71-100%: High risk
    
    **Risk Levels:**
    - 🟢 **Low:** Customer is stable and unlikely to leave
    - 🟡 **Medium:** Customer shows some warning signs
    - 🔴 **High:** Customer is at serious risk of churning
    
    ---
    
    ### 💡 Example Interpretations
    
    **Scenario 1: Loyal Customer (Low Risk)**
    - Tenure: 48 months, Total Spend: $2400 (Monthly: $50)
    - Usage: 20/month, Support Calls: 1, Payment Delay: 0
    - Premium/Annual contract, Last Interaction: 3 days
    - **Interpretation:** Long-term, engaged customer with affordable costs and no issues
    
    **Scenario 2: At-Risk Customer (High Risk)**
    - Tenure: 2 months, Total Spend: $80 (Monthly: $40)
    - Usage: 3/month, Support Calls: 8, Payment Delay: 25
    - Basic/Monthly contract, Last Interaction: 60 days
    - **Interpretation:** New customer, not engaged, having problems, may have already left
    """)
