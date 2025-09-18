import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, FunctionTransformer
)
from sklearn.pipeline import Pipeline


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
