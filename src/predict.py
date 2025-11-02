# src/predict.py
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

def predict_from_df(model_path, df):
    d = joblib.load(model_path)
    model = d["model"]
    
    # Handle 'Last Interaction' column if present
    if "Last Interaction" in df.columns and "days_since_last_interaction" not in df.columns:
        df["Last Interaction"] = pd.to_datetime(df["Last Interaction"], errors="coerce")
        df["days_since_last_interaction"] = (pd.Timestamp.today() - df["Last Interaction"]).dt.days
        df.drop(columns=["Last Interaction"], inplace=True)
    
    # assume df contains same feature columns as training (order doesn't have to match)
    preds = model.predict(df)
    return preds

def predict_from_csv(model_path, input_csv, out_csv=None):
    df = pd.read_csv(input_csv)
    preds = predict_from_df(model_path, df)
    pred_df = pd.DataFrame({"prediction": preds})
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(out_csv, index=False)
    return pred_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to saved model joblib")
    parser.add_argument("input_csv", help="Input CSV for predictions (features only)")
    parser.add_argument("--out", help="Optional output CSV to save predictions")
    args = parser.parse_args()

    pred_df = predict_from_csv(args.model, args.input_csv, args.out)
    print(pred_df.head(20))

if __name__ == "__main__":
    main()
