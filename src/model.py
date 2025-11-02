# src/model.py
import inspect
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

def _onehot_encoder_handle_unknown_ignore():
    """
    Return a OneHotEncoder instance that works across sklearn versions.
    Some versions expect `sparse=False`, others expect `sparse_output=False`.
    """
    sig = inspect.signature(OneHotEncoder.__init__)
    kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in sig.parameters:
        kwargs["sparse_output"] = False
    elif "sparse" in sig.parameters:
        kwargs["sparse"] = False
    else:
        return OneHotEncoder(handle_unknown="ignore")
    return OneHotEncoder(**kwargs)

def build_pipeline(numeric_cols, categorical_cols, random_state=42):
    """
    Returns an sklearn Pipeline that applies preprocessing then a RandomForestClassifier.
    numeric_cols: list of numeric column names
    categorical_cols: list of categorical column names
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", _onehot_encoder_handle_unknown_ignore()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    # Use balanced class weight to help with class imbalance
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])
    return pipeline
