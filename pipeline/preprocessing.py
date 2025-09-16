import polars as pl
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
    FunctionTransformer
)


def get_scaler(method:str):
    """
    Returns a Scikit-learn transformer based on the chosen scaling method.
    Supports:
    - standard: StandardScaler
    - log2: log2 transform + StandardScaler
    - log10: log10 transform + StandardScaler
    - minmax: MinMaxScaler
    - robust: RobustScaler
    - maxabs: MaxAbsScaler
    - quantile: QuantileTransformer
    """
    method = method.lower()
    if method == 'standard':
        return StandardScaler()
    elif method == 'log2':
        return Pipeline([
            ('log2', FunctionTransformer(func=np.log2, inverse_func=np.exp2, validate=True)),
            ('scaler', StandardScaler())
        ])
    elif method == 'log10':
        return Pipeline([
            ('log10', FunctionTransformer(func=np.log10, inverse_func=lambda x: 10**x, validate=True)),
            ('scaler', StandardScaler())
        ])
    elif method == 'minmax':
        return MinMaxScaler()
    elif method == 'robust':
        return RobustScaler()
    elif method == 'maxabs':
        return MaxAbsScaler()
    elif method == 'quantile':
        return QuantileTransformer(output_distribution='normal')
    else:
        raise ValueError(f"Unsupported scaling method: {method}")


def preprocess_features(X: pl.DataFrame, scaling_method: str = "standard") -> Tuple[pl.DataFrame, Dict[str, LabelEncoder], object]:
    """
    Preprocesses the feature matrix X:
    - Label encodes categorical features (Utf8 and bool)
    - Scales numeric features using StandardScaler

    Returns:
        - Processed X as a Polars DataFrame
        - Dictionary of LabelEncoders used
        - StandardScaler fitted to the numeric columns
    """
    df = X.clone()
    le_dict = {}

    # Detect and cast Utf8 and Boolean columns to Categorical
    categorical_cols = []
    for col, dtype in df.schema.items():
        if dtype == pl.Utf8 or dtype == pl.Boolean:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
            categorical_cols.append(col)

    # Label encode categoricals using sklearn (via pandas bridge)
    df_pd = df.to_pandas()

    for col in categorical_cols:
        le = LabelEncoder()
        df_pd[col] = le.fit_transform(df_pd[col])
        le_dict[col] = le

    # Scale numeric features
    numeric_cols = df_pd.select_dtypes(include=np.number).columns.tolist()
    scaler = get_scaler(scaling_method)
    df_pd[numeric_cols] = scaler.fit_transform(df_pd[numeric_cols])

    # Return Polars DataFrame + encoders
    X_processed = pl.from_pandas(df_pd)

    return X_processed, le_dict, scaler


def preprocess_target(y: Union[pl.Series, pl.DataFrame]) -> Tuple[np.ndarray, Union[LabelEncoder, None]]:
    """
    Preprocesses the target variable y:
    - Label encodes if not already numeric
    - Returns numpy array and optional LabelEncoder
    """
    if isinstance(y, pl.DataFrame):
        y = y.to_series()

    # If y is numeric (int or float), assume it's already encoded
    if y.dtype in (pl.Int64, pl.Int32, pl.UInt32, pl.UInt64, pl.Float64):
        return y.to_numpy(), None

    # Otherwise, encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.to_numpy())

    return y_encoded, le
