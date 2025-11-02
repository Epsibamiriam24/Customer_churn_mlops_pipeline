import joblib
import pandas as pd

# Load the model
model_data = joblib.load("artifacts/model.joblib")

print("="*50)
print("MODEL INFORMATION")
print("="*50)
print(f"Model type: {type(model_data['model'])}")
print(f"\nTarget column: {model_data['target_col']}")
print(f"\nNumeric columns: {model_data['numeric_cols']}")
print(f"\nCategorical columns: {model_data['categorical_cols']}")
print(f"\nBest params: {model_data.get('best_params', 'None')}")
print(f"\nModel accuracy: {model_data.get('accuracy', 'None')}")

# Check the pipeline structure
print("\n" + "="*50)
print("PIPELINE STRUCTURE")
print("="*50)
pipeline = model_data['model']
print(f"Pipeline steps: {pipeline.named_steps.keys()}")

# Get feature names from the preprocessor
if 'preprocessor' in pipeline.named_steps:
    preprocessor = pipeline.named_steps['preprocessor']
    print(f"\nPreprocessor transformers:")
    for name, transformer, columns in preprocessor.transformers_:
        print(f"  {name}: {columns}")

# Test with sample data
print("\n" + "="*50)
print("TEST PREDICTION")
print("="*50)

# Create sample input matching the training data structure
sample_data = pd.DataFrame({
    'Age': [35.0],
    'Gender': ['male'],
    'Tenure': [24.0],
    'Usage Frequency': [15.0],
    'Support Calls': [3.0],
    'Payment Delay': [5.0],
    'Subscription Type': ['standard'],
    'Contract Length': ['annual'],
    'Total Spend': [1200.0],
    'Last Interaction': [30.0]
})

print("\nSample input data:")
print(sample_data)
print(f"\nData types:\n{sample_data.dtypes}")

try:
    prediction = pipeline.predict(sample_data)
    prob = pipeline.predict_proba(sample_data)
    print(f"\nPrediction: {prediction[0]}")
    print(f"Probabilities: {prob[0]}")
    print(f"Churn probability: {prob[0][1]:.2%}")
except Exception as e:
    print(f"\nError during prediction: {e}")

# Test with different values
print("\n" + "="*50)
print("TESTING DIFFERENT SCENARIOS")
print("="*50)

scenarios = [
    {
        'name': 'Low Risk Customer',
        'data': pd.DataFrame({
            'Age': [30.0], 'Gender': ['female'], 'Tenure': [60.0],
            'Usage Frequency': [25.0], 'Support Calls': [1.0], 'Payment Delay': [0.0],
            'Subscription Type': ['premium'], 'Contract Length': ['annual'],
            'Total Spend': [3000.0], 'Last Interaction': [5.0]
        })
    },
    {
        'name': 'High Risk Customer',
        'data': pd.DataFrame({
            'Age': [25.0], 'Gender': ['male'], 'Tenure': [3.0],
            'Usage Frequency': [2.0], 'Support Calls': [10.0], 'Payment Delay': [30.0],
            'Subscription Type': ['basic'], 'Contract Length': ['monthly'],
            'Total Spend': [100.0], 'Last Interaction': [90.0]
        })
    }
]

for scenario in scenarios:
    try:
        pred = pipeline.predict(scenario['data'])
        prob = pipeline.predict_proba(scenario['data'])
        print(f"\n{scenario['name']}:")
        print(f"  Prediction: {'Churn' if pred[0] == 1 else 'No Churn'}")
        print(f"  Churn probability: {prob[0][1]:.2%}")
    except Exception as e:
        print(f"\n{scenario['name']}: Error - {e}")
