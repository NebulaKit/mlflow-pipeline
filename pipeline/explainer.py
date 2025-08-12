import shap
import matplotlib.pyplot as plt
import os
from typing import List
import polars as pl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def explain_model(model, X: pl.DataFrame, output_path: str, class_names: List[str], feature_map_path: str) -> List[str]:
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
        # Use KernelExplainer for other models (slower, approximate)
        sample_size = min(100, len(X))  # Use up to 100 rows, or fewer if less data
        background_data = shap.sample(X, sample_size, random_state=42)

        # For newer sklearn versions (>=1.2), wrap predict_proba to avoid errors
        model_output = model.predict_proba if hasattr(model, "predict_proba") else lambda x: model.predict(x)

        print("Using KernelExplainer — this may take a while...")
        explainer = shap.KernelExplainer(model_output, background_data)
        shap_values = explainer.shap_values(X)
        
        

    # Handle multiclass or binary
    is_multiclass = len(class_names) > 2
    shap_paths = []
    
    feature_names = X.columns
    if feature_map_path:
        # Load feature map if provided
        feature_map = pd.read_csv(feature_map_path)
        id_to_name = dict(zip(feature_map['ID'], feature_map['Name']))
        feature_names = [id_to_name.get(col, col) for col in X.columns]

    if is_multiclass:
        for i, class_name in enumerate(class_names):
            path = os.path.splitext(output_path)[0] + f"_{class_name}.png"
            shap_paths.append(path)
            plt.figure()
            shap.summary_plot(shap_values[:, :, i], X, feature_names=feature_names, show=False, max_display=5)
            plt.title(f"SHAP Summary Plot: {class_name}", pad=20)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            
            # Save SHAP values
            df = pd.DataFrame(shap_values[:, :, i], columns=feature_names)
            df.to_csv(os.path.splitext(output_path)[0] + f"_{class_name}.csv", index=False)

    else:
        shap_paths.append(output_path)
        plt.figure()
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=5)
        plt.title("SHAP Summary Plot", pad=20)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        # Save SHAP values
        df = pd.DataFrame(shap_values, columns=feature_names)
        df.to_csv(os.path.splitext(output_path)[0] + ".csv", index=False)
        
    return shap_paths
