"""
data.py

Data loading and preprocessing utilities.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV file and return as DataFrame."""
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Shape: {df.shape}")
    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features (X) and target (y)."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y


def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline that:
    - Scales numeric features with StandardScaler
    - One-hot encodes categorical features
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = (
        X.select_dtypes(include=["object", "category"]) .columns.tolist()
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            )
        )

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return transformer
