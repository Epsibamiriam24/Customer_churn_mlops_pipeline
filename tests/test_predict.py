# tests/test_predict.py
import pandas as pd
from pathlib import Path
from src.train import run_train
from src.predict import predict_from_csv
from sklearn.datasets import make_classification

def test_predict_from_csv(tmp_path):
    # Create tiny dataset and train
    X, y = make_classification(n_samples=60, n_features=4, n_informative=2, random_state=0)
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y

    train_df = df.sample(frac=0.8, random_state=1)
    test_df = df.drop(train_df.index).drop(columns=["target"])  # no labels for prediction step

    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test_input.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    model_path = tmp_path / "model.joblib"
    run_train(str(train_csv), test_csv=None, out_path=str(model_path), target="target", do_gridsearch=False)

    # Run prediction (no target in input)
    preds_df = predict_from_csv(str(model_path), str(test_csv))
    assert "prediction" in preds_df.columns
    assert len(preds_df) == len(test_df)
