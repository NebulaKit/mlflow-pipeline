from argparse import Namespace
from typing import Union
from pipeline.config import Config
import os
import joblib
from typing import Dict, Optional, List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd


def build_feature_labels(
    original_feature_ids: List[str],
    feature_map_path: Optional[str] = None
) -> Tuple[Optional[List[str]], List[str]]:
    """
    Build human-readable feature labels if a valid feature map is provided.

    Args:
        original_feature_ids: List of original feature IDs (canonical feature columns).
        feature_map_path: Optional path to feature_map CSV with columns ['ID', 'Name'].

    Returns:
        feature_names_formatted:
            List of 'Name [ID=...]' in the same order as original_feature_ids
            if feature_map exists and is valid, otherwise None.
        feature_names:
            List of plain names if feature map provided,
            or original IDs if feature map missing/invalid.
    """
    ids = [str(x) for x in original_feature_ids]

    # Default: no mapping
    feature_names_formatted = None
    feature_names = ids[:]  # fallback to IDs

    if feature_map_path and os.path.exists(feature_map_path):
        try:
            fmap = pd.read_csv(feature_map_path)
            if {"ID", "Name"}.issubset(fmap.columns):
                id_to_name = dict(zip(fmap["ID"].astype(str), fmap["Name"].astype(str)))
                feature_names = [id_to_name.get(fid, fid) for fid in ids]
                feature_names_formatted = [
                    f"{id_to_name.get(fid, fid)} [ID={fid}]" for fid in ids
                ]
            else:
                print("Feature map missing required columns ['ID','Name']; using IDs only.")
        except Exception as e:
            print(f"Could not read feature map: {e}. Using IDs only.")
    else:
        if feature_map_path:
            print(f"Feature map not found at: {feature_map_path}. Using IDs only.")

    return feature_names_formatted, feature_names


def override_config_from_args(config: Config, args: Union[Namespace, dict]) -> Config:
    """
    Overrides a Config object with CLI arguments or a dict.
    Falls back to config defaults if arguments are missing or None.
    """
    arg_dict = vars(args) if isinstance(args, Namespace) else args
    return Config(
        **{
            k: arg_dict[k] if k in arg_dict and arg_dict[k] is not None else getattr(config, k)
            for k in config.__dataclass_fields__.keys()
        }
    )


def save_preprocessing_artifacts(
    le_dict: Dict[str, LabelEncoder],
    label_encoder: Optional[LabelEncoder],
    scaler: StandardScaler,
    dir_name: str,
    data_file_name: str
) -> None:
    """
    Save preprocessing artifacts (feature encoders, target encoder, scaler) to disk.
    """
    output_path = os.path.join(dir_name, data_file_name, "preprocessing.joblib")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({
        "feature_encoders": le_dict,
        "target_encoder": label_encoder,
        "scaler": scaler
    }, output_path)
    
    
def resolve_scoring(y, user_metric: Optional[str]) -> str:
    """
    Resolve scoring metric for GridSearchCV.
    - Uses user-specified metric if provided.
    - Otherwise defaults to 'roc_auc' (binary) or 'roc_auc_ovr' (multi-class).
    """
    import numpy as np

    is_multiclass = len(np.unique(y)) > 2

    if user_metric is not None:
        return user_metric  # Trust user choice

    if is_multiclass:
        return "roc_auc_ovr"   # One-vs-Rest macro AUC
    else:
        return "roc_auc"
