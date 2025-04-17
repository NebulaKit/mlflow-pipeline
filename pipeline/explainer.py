import shap
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Optional, List
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def explain_model(model, X: pl.DataFrame, output_path: str, class_names: List[str]) -> List[str]:
    """
    Generates SHAP summary plot(s) for binary or multiclass classification.
    Supports tree-based, linear, and fallback models.
    Saves plots to disk.
    """
    # Convert Polars to pandas if needed
    if isinstance(X, pl.DataFrame):
        X = X.to_pandas()

    # Determine appropriate SHAP explainer
    if isinstance(model, (XGBClassifier, RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)

    else:
        explainer = shap.KernelExplainer(model.predict_proba, X[:100])  # Sample 100 rows TODO: make dynamic
        shap_values = explainer.shap_values(X)

    # Handle multiclass or binary
    is_multiclass = len(class_names) > 2
    shap_paths = []

    if is_multiclass:
        for i, class_name in enumerate(class_names):
            path = os.path.splitext(output_path)[0] + f"_{class_name}.png"
            shap_paths.append(path)
            plt.figure()
            shap.summary_plot(shap_values[:, :, i], X, feature_names=X.columns, show=False)
            plt.title(f"SHAP Summary Plot: {class_name}", pad=20)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()

    else:
        shap_paths.append(output_path)
        plt.figure()
        shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
        plt.title("SHAP Summary Plot", pad=20)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    return shap_paths
