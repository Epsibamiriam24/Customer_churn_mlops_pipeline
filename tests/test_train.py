# tests/test_train.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.train import run_train
from sklearn.datasets import make_classification

def test_run_train_creates_model(tmp_path):
    # generate small synthetic classification data
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
    cols = [f"f{i}" for i in range(X.shape[1])]
    train_df = pd.DataFrame(X, columns=cols)
    train_df["target"] = y

    # split a small test set
    test_df = train_df.sample(frac=0.2, random_state=1)
    train_df = train_df.drop(test_df.index)

    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    out_model = tmp_path / "model.joblib"
    res = run_train(str(train_csv), str(test_csv), out_path=str(out_model), target="target", do_gridsearch=False)
    assert Path(res["out_path"]).exists()
    # accuracy should be a float between 0 and 1
    assert res["accuracy"] is None or (0.0 <= res["accuracy"] <= 1.0)
