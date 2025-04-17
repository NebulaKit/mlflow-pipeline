from argparse import Namespace
from typing import Union
from pipeline.config import Config
import os
import joblib
from typing import Dict, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
    output_path: str = "models/preprocessing.joblib"
) -> None:
    """
    Save preprocessing artifacts (feature encoders, target encoder, scaler) to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({
        "feature_encoders": le_dict,
        "target_encoder": label_encoder,
        "scaler": scaler
    }, output_path)
