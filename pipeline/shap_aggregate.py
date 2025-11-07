import os
from typing import Dict, List
import pandas as pd


def aggregate_shap_and_get_top(
    shap_root_dir: str,
    class_names: List[str],
    top_n: int = 20,
) -> Dict[str, List[str]]:
    """
    Aggregate SHAP values across classifier subfolders for each class
    using ONLY the canonical IDs CSVs:
        {class}_shap_values_ids.csv

    Produces:
        outputs/.../shap/aggregate/abs_mean_shap_values.csv
            rows = feature IDs, columns = classes
            values = mean over models of mean(|SHAP|) per feature

    Returns:
        dict[class_name -> List[top_n feature IDs]]
    """
    # Find classifier subdirs (exclude 'aggregate')
    if not os.path.isdir(shap_root_dir):
        raise FileNotFoundError(f"SHAP root directory not found: {shap_root_dir}")

    model_dirs = [
        os.path.join(shap_root_dir, d)
        for d in os.listdir(shap_root_dir)
        if os.path.isdir(os.path.join(shap_root_dir, d)) and d.lower() != "aggregate"
    ]
    if not model_dirs:
        raise FileNotFoundError(f"No classifier subfolders found in: {shap_root_dir}")

    # Collect mean(|SHAP|) per feature per model per class
    per_class_tables: Dict[str, List[pd.Series]] = {c: [] for c in class_names}

    for model_dir in model_dirs:
        for c in class_names:
            # IDs-only file
            path_ids = os.path.join(model_dir, f"{c}_shap_values_ids.csv")
            if not os.path.isfile(path_ids):
                # silently skip if that class wasn't produced by this model
                continue

            print(f"Loading SHAP IDs CSV for class '{c}' from model dir: {model_dir}")
            df = pd.read_csv(path_ids)
            # mean(|SHAP|) per feature for this model & class
            s = df.abs().mean(axis=0)
            s.name = os.path.basename(model_dir)
            per_class_tables[c].append(s)

    # Aggregate across models (average the per-model means)
    agg_per_class: Dict[str, pd.Series] = {}
    for c in class_names:
        if per_class_tables[c]:
            class_df = pd.concat(per_class_tables[c], axis=1)  # index=feature IDs, columns=models
            agg_per_class[c] = class_df.mean(axis=1, skipna=True)

    if not agg_per_class:
        raise RuntimeError("No per-class SHAP ID CSVs were found or loaded successfully.")

    # Build a single matrix: rows = features (union), cols = classes
    all_feats = sorted(set().union(*[s.index for s in agg_per_class.values()]))
    mat = pd.DataFrame(index=all_feats, columns=class_names, dtype=float)
    for c in class_names:
        if c in agg_per_class:
            mat[c] = agg_per_class[c].reindex(all_feats)

    # Save aggregate matrix
    agg_dir = os.path.join(shap_root_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    out_csv = os.path.join(agg_dir, "abs_mean_shap_values.csv")
    mat.to_csv(out_csv)

    # Top-N IDs per class
    top_by_class: Dict[str, List[str]] = {}
    for c in class_names:
        if c in mat.columns:
            s = mat[c].dropna().sort_values(ascending=False)
            top_by_class[c] = s.head(top_n).index.tolist()

    return top_by_class
