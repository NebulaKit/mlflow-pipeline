from dataclasses import dataclass, field
from typing import List, Optional
import os

@dataclass
class Config:
    data_path: str = "data/raw/synthetic_lipidomics.csv"
    label_col: str = "Group"
    feature_map_path: str = "data/raw/lipid_map.csv"
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    models_to_run: Optional[List[str]] = field(default_factory=list)
    output_dir: str = "outputs"
    model_dir: str = "models"
    experiment_name: str = "Lipidomics_ML_Models"
    
    @property
    def preprocessing_artifact_path(self) -> str:
        """
        Returns the full path to the preprocessing artifact file.
        """
        return os.path.join(self.model_dir, "preprocessing.joblib")
