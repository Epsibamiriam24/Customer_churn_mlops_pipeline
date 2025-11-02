import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_initial_model():
    """Train a simple initial model for deployment testing"""
    # Create sample data
    data = {
        'tenure': [12, 24, 36, 1, 6],
        'MonthlyCharges': [50.0, 70.0, 90.0, 45.0, 80.0],
        'TotalCharges': [600.0, 1680.0, 3240.0, 45.0, 480.0],
        'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No', 'Yes', 'No internet service', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
        'Churn': ['No', 'Yes', 'No', 'Yes', 'No']
    }
    df = pd.DataFrame(data)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']):
        df[col] = le.fit_transform(df[col])
    
    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model with sample metrics
    model_info = {
        "model": model,
        "accuracy": 0.85,
        "precision": 0.83,
        "recall": 0.81,
        "confusion_matrix": [[2, 0], [1, 2]]
    }
    
    # Create artifacts directory and save model
    import os
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model_info, 'artifacts/model.joblib')

if __name__ == "__main__":
    train_initial_model()