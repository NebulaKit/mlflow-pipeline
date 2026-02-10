import shap
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict
import polars as pl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from .utils import build_feature_labels


def explain_model(
    model,
    model_name: str,
    X,
    output_dir: str,
    class_names: List[str],
    feature_map_path: str,
    max_plot_display: int = 20
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Generates SHAP summary plot(s) for binary or multiclass classification.
    Supports tree-based, linear, and fallback models.
    Saves plots to disk.
    
    Returns:
        shap_paths (List[str]): paths to saved SHAP plots
        top_features_per_class (Dict[str, List[str]]): class->top-N ORIGINAL feature IDs
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
        sample_size = min(100, len(X))
        background_data = shap.sample(X, sample_size, random_state=42)
        model_output = model.predict_proba if hasattr(model, "predict_proba") else lambda x: model.predict(x)
        print("Using KernelExplainer — this may take a while...")
        explainer = shap.KernelExplainer(model_output, background_data)
        shap_values = explainer.shap_values(X)
        
    # Handle multiclass or binary
    is_multiclass = len(class_names) > 2
    shap_paths = []
    top_features_per_class: Dict[str, List[str]] = {}

    # Keep ORIGINAL IDs for downstream use; use mapped names only for plotting/CSVs
    original_feature_ids = list(X.columns)
    feature_names_formatted, feature_names = build_feature_labels(
        original_feature_ids, feature_map_path
    )

    if is_multiclass:
        for i, class_name in enumerate(class_names):
            # Support both SHAP shapes: list-of-arrays or (N,F,C)
            class_shap = shap_values[i] if isinstance(shap_values, list) else shap_values[:, :, i]

            # --- top-N ORIGINAL feature IDs for this class ---
            mean_abs = np.abs(class_shap).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:max_plot_display]
            top_features_per_class[class_name] = [original_feature_ids[j] for j in order]

            # Plot + save
            path = os.path.join(output_dir, f"{class_name}_summary_plot.png")
            shap_paths.append(path)
            plt.figure()
            shap.summary_plot(class_shap, X, feature_names=feature_names, show=False, max_display=max_plot_display)
            plt.title(f"{model_name} SHAP Summary Plot: {class_name}", pad=20)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            
            # Save SHAP values
            if feature_names_formatted:
                pd.DataFrame(class_shap, columns=feature_names_formatted).to_csv(
                    os.path.join(output_dir, f"{class_name}_shap_values.csv"), index=False
                )
            pd.DataFrame(class_shap, columns=original_feature_ids).to_csv(
                os.path.join(output_dir, f"{class_name}_shap_values_ids.csv"), index=False
            )

    else:
        print("Binary classification detected for SHAP explanation.")
        # Binary: use positive class
        class_shap = shap_values[1] if isinstance(shap_values, list) and len(shap_values) >= 2 else shap_values

        # --- top-N ORIGINAL feature IDs for positive class ---
        mean_abs = np.abs(class_shap).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:max_plot_display]
        pos_key = class_names[-1] if len(class_names) >= 2 else "positive"
        order = np.asarray(order).ravel().astype(int)
        orig = np.asarray(original_feature_ids)
        top_features_per_class[pos_key] = orig[order].tolist()

        # Plot + save
        path = os.path.join(output_dir, f"positive_class_summary_plot.png")
        shap_paths.append(path)
        plt.figure()
        shap.summary_plot(class_shap, X, feature_names=feature_names, show=False, max_display=max_plot_display)
        plt.title(f"{model_name} SHAP Summary Plot", pad=20)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        
        # Save SHAP values
        if feature_names_formatted:
            pd.DataFrame(class_shap, columns=feature_names_formatted).to_csv(
                os.path.join(output_dir, "positive_class_shap_values.csv"), index=False
            )
        pd.DataFrame(class_shap, columns=original_feature_ids).to_csv(
            os.path.join(output_dir, "positive_class_shap_values_ids.csv"), index=False
        )
        
    return shap_paths, top_features_per_class
