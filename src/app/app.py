import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# Add src to Python path
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load the trained model
@st.cache_resource
def load_model():
    model_path = SRC_DIR.parent / "artifacts/model.joblib"
    if not model_path.exists():
        st.error("Model file not found! Please train the model first.")
        return None
    return joblib.load(model_path)

def main():
    st.title("Customer Churn Prediction")
    st.write("""
    ### Upload your customer data for churn prediction
    This application predicts customer churn based on historical data.
    """)

    # Load model
    model_data = load_model()
    if model_data is None:
        return

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the input data
            input_df = pd.read_csv(uploaded_file)
            st.write("### Data Preview:")
            st.write(input_df.head())

            # Check if required columns are present
            required_cols = model_data['numeric_cols'] + model_data['categorical_cols']
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.write("Required columns:", required_cols)
                return

            if st.button("Predict Churn"):
                # Make predictions
                model = model_data['model']
                predictions = model.predict(input_df)
                
                # Add predictions to dataframe
                output_df = input_df.copy()
                output_df['Churn_Prediction'] = predictions
                output_df['Churn_Prediction'] = output_df['Churn_Prediction'].map({1: 'Yes', 0: 'No'})
                
                # Display results
                st.write("### Prediction Results:")
                st.write(output_df)
                
                # Download button for predictions
                st.download_button(
                    label="Download Predictions",
                    data=output_df.to_csv(index=False),
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
                
                # Show prediction statistics
                total = len(predictions)
                churn_count = sum(predictions)
                st.write(f"### Summary:")
                st.write(f"Total customers: {total}")
                st.write(f"Predicted to churn: {churn_count} ({(churn_count/total)*100:.1f}%)")
                st.write(f"Predicted to stay: {total-churn_count} ({((total-churn_count)/total)*100:.1f}%)")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()