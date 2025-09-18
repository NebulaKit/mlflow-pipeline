import pandas as pd
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from .preprocessing_utils import get_scaler


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Handles categorical encoding + numeric feature scaling
    in a train/test safe manner (fit on train, transform on test).
    """
    def __init__(self, scaling_method: str = "standard"):
        self.scaling_method = scaling_method
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = None
        self.categorical_cols: List[str] = []
        self.numeric_cols: List[str] = []
        
    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        
        # Detect categoricals
        self.categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Label encode categoricals
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            
        # Detect numerics
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Fit scaler to numeric features
        self.scaler = get_scaler(self.scaling_method)
        self.scaler.fit(df[self.numeric_cols])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # Encode categoricals with fitted encoders
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))
                
        # Scale numeric features
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        
        return df
