import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder


def preprocess_target(y: pd.Series):
    """
    Preprocesses the target variable y:
    - Label encodes if not already numeric
    - Returns pandas Series and optional LabelEncoder
    """
    #if pd.api.types.is_numeric_dtype(y):
    #    return y, None

    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    return pd.Series(y_encoded, index=y.index), le
