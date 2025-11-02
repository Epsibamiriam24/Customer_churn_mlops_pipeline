import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def train_initial_model():
    """Train a simple initial model for deployment testing"""
    try:
        # Try to load actual training data
        df = pd.read_csv('Data/train.csv')
        if 'Churn' not in df.columns:
            raise ValueError("Training data doesn't have target column")
    except Exception as e:
        print(f"Using dummy data because: {str(e)}")
        # Create sample data if loading fails
        data = {
            'tenure': [12, 24, 36, 1, 6] * 20,
            'MonthlyCharges': [50.0, 70.0, 90.0, 45.0, 80.0] * 20,
            'TotalCharges': [600.0, 1680.0, 3240.0, 45.0, 480.0] * 20,
            'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No', 'Fiber optic'] * 20,
            'OnlineSecurity': ['Yes', 'No', 'Yes', 'No internet service', 'No'] * 20,
            'OnlineBackup': ['Yes', 'No', 'Yes', 'No internet service', 'No'] * 20,
            'TechSupport': ['Yes', 'No', 'Yes', 'No internet service', 'No'] * 20,
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'] * 20,
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Electronic check'] * 20,
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'] * 20,
            'Churn': ['No', 'Yes', 'No', 'Yes', 'No'] * 20
        }
        df = pd.DataFrame(data)
    
    try:
        print("Starting model training...")
        
        # Handle missing values
        numeric_imputer = SimpleImputer(strategy='mean')
        categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Impute missing values
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
        df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
        
        # Encode categorical variables
        le_dict = {}
        for col in categorical_cols:
            if col != 'Churn':  # Don't encode target yet
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                le_dict[col] = le
        
        # Encode target
        le_target = LabelEncoder()
        df['Churn'] = le_target.fit_transform(df['Churn'])
        
        # Split features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        print("Training model...")
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Get predictions for basic metrics
        y_pred = model.predict(X)
        
        # Calculate basic metrics
        accuracy = np.mean(y == y_pred)
        precision = np.sum((y == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
        recall = np.sum((y == 1) & (y_pred == 1)) / np.sum(y == 1)
        
        # Create confusion matrix
        conf_matrix = [[np.sum((y == 0) & (y_pred == 0)), np.sum((y == 0) & (y_pred == 1))],
                      [np.sum((y == 1) & (y_pred == 0)), np.sum((y == 1) & (y_pred == 1))]]
        
        print("Saving model...")
        # Save model with computed metrics
        model_info = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix,
            "label_encoders": le_dict,
            "target_encoder": le_target
        }
        
        # Create artifacts directory and save model
        import os
        os.makedirs('artifacts', exist_ok=True)
        joblib.dump(model_info, 'artifacts/model.joblib')
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_initial_model()