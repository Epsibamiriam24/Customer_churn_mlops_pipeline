# src/train.py
import argparse
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Ensure src/ is importable when running `python src/train.py`
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import build_pipeline

DEFAULT_OUT = "artifacts/model.joblib"
DEFAULT_TARGET = "Churn"
RANDOM_STATE = 42

def infer_target_column(df, target_name=None):
    if target_name:
        for c in df.columns:
            if c.lower() == target_name.lower():
                return c
    if "target" in df.columns:
        return "target"
    return df.columns[-1]

def detect_feature_types(df, target_col):
    df_features = df.drop(columns=[target_col])
    numeric_cols = df_features.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df_features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols

def prepare_target(series: pd.Series):
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return series.astype(int)
    s = series.astype(str).str.strip().str.lower()
    mapping = {"yes":1, "y":1, "true":1, "1":1, "no":0, "n":0, "false":0, "0":0}
    if s.isin(mapping.keys()).all():
        return s.map(mapping).astype(int)
    codes, uniques = pd.factorize(s)
    if len(uniques) == 2:
        return pd.Series(codes, index=series.index)
    raise ValueError(f"Cannot interpret target column values. Found unique values: {list(uniques)[:10]}")

def convert_last_interaction(df: pd.DataFrame):
    # Keep Last Interaction as is, no conversion needed
    return df

def drop_identifier_columns(df: pd.DataFrame):
    # drop typical identifier columns: CustomerID, id, custid
    id_cols = [c for c in df.columns if c.strip().lower() in ("customerid", "id", "custid")]
    if id_cols:
        print("Dropping identifier columns:", id_cols)
        df = df.drop(columns=id_cols)
    return df

def run_train(train_csv, test_csv=None, out_path=DEFAULT_OUT, target=None, do_gridsearch=False):
    project_root = SRC_DIR.parent
    train_p = Path(train_csv)
    test_p = Path(test_csv) if test_csv else None
    out_p = Path(out_path)

    if not train_p.is_absolute():
        train_p = project_root.joinpath(train_p)
    if test_p and not test_p.is_absolute():
        test_p = project_root.joinpath(test_p)
    if not out_p.is_absolute():
        out_p = project_root.joinpath(out_p)

    print("Train file:", train_p)
    if test_p:
        print("Test file :", test_p)
    print("Model out :", out_p)

    if not train_p.exists():
        raise FileNotFoundError(f"Training file not found: {train_p}")

    train_df = pd.read_csv(train_p)

    target_col = infer_target_column(train_df, target or DEFAULT_TARGET)
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(train_df.columns)}")
    print("Detected target column:", target_col)

    total_rows = len(train_df)
    missing_labels = int(train_df[target_col].isna().sum())
    print(f"Training rows: {total_rows}, missing target ({target_col}) rows: {missing_labels}")
    if missing_labels > 0:
        print("Sample rows with missing target (first 5):")
        print(train_df[train_df[target_col].isna()].head(5))

    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    if train_df.shape[0] == 0:
        raise ValueError("No labeled rows remain after dropping missing targets.")

    y_train_raw = train_df[target_col]
    y_train = prepare_target(y_train_raw)
    X_train = train_df.drop(columns=[target_col])

    # drop identifier columns and convert last interaction
    X_train = drop_identifier_columns(X_train)
    X_train = convert_last_interaction(X_train)

    numeric_cols, categorical_cols = detect_feature_types(pd.concat([X_train, y_train.rename(target_col)], axis=1), target_col)
    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", categorical_cols)

    pipeline = build_pipeline(numeric_cols, categorical_cols, random_state=RANDOM_STATE)

    model = None
    best_params = None
    if do_gridsearch:
        param_grid = {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [None, 10],
        }
        gs = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=1, verbose=1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        print("Fitting pipeline on training data...")
        pipeline.fit(X_train, y_train)
        model = pipeline

    accuracy = None
    if test_p:
        if not test_p.exists():
            raise FileNotFoundError(f"Test file not found: {test_p}")
        test_df = pd.read_csv(test_p)
        # If test has target, evaluate; otherwise generate predictions
        if target_col in test_df.columns:
            n_missing_test = int(test_df[target_col].isna().sum())
            if n_missing_test > 0:
                print(f"Test set has {n_missing_test} missing labels; dropping them for evaluation.")
                test_df = test_df.dropna(subset=[target_col]).reset_index(drop=True)
            y_test = prepare_target(test_df[target_col])
            X_test = test_df.drop(columns=[target_col])
            # apply same cleaning to test features
            X_test = drop_identifier_columns(X_test)
            X_test = convert_last_interaction(X_test)
            preds = model.predict(X_test)
            accuracy = float(accuracy_score(y_test, preds))
            print(f"Test Accuracy: {accuracy:.4f}")
            print("Classification report:")
            print(classification_report(y_test, preds))
            print("Confusion matrix:\n", confusion_matrix(y_test, preds))
            # try ROC AUC if possible and binary
            try:
                if hasattr(model.named_steps["classifier"], "predict_proba") and len(np.unique(y_test)) == 2:
                    probs = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, probs)
                    print(f"ROC AUC: {auc:.4f}")
            except Exception as e:
                print("Could not compute ROC AUC:", e)
        else:
            X_test = test_df
            X_test = drop_identifier_columns(X_test)
            X_test = convert_last_interaction(X_test)
            preds = model.predict(X_test)
            out_dir = out_p.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"prediction": preds}).to_csv(out_dir / "test_predictions.csv", index=False)
            print("Test CSV had no target column. Predictions saved to", out_dir / "test_predictions.csv")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "best_params": best_params,
        "accuracy": accuracy
    }, out_p)
    print("Saved model to:", out_p)
    return {"out_path": str(out_p), "accuracy": accuracy}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=False, default="data/train.csv", help="Path to training CSV (default: data/train.csv)")
    parser.add_argument("--test", required=False, default="data/test.csv", help="Path to test CSV (optional)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output joblib path")
    parser.add_argument("--target", default=None, help="Target column name (optional). Defaults to 'Churn' or last column.")
    parser.add_argument("--grid", action="store_true", help="Run small grid search")
    return parser.parse_args()

def main():
    args = parse_args()
    result = run_train(args.train, args.test, args.out, args.target, do_gridsearch=args.grid)
    print(f"Saved model to {result['out_path']}")
    if result["accuracy"] is not None:
        print(f"Test accuracy: {result['accuracy']:.4f}")

if __name__ == "__main__":
    main()