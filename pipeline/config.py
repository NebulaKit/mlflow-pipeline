from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Config:
    data_path: str = "data/raw/synthetic_lipidomics.csv"
    label_col: str = "Group"
    control_name: str = "D"
    feature_map_path: str = "data/raw/lipid_map.csv"
    scaling_method: str = "standard" # options: 'standard', 'log2', 'log10', 'minmax', 'robust', 'maxabs', 'quantile'
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    models_to_run: Optional[List[str]] = field(default_factory=list)
    do_grid_search: bool = True
    grid_search_metric: Optional[str] = "f1_macro"
    output_dir: str = "outputs/"
    models_dir: str = "models/"
    experiment_name: str = "Lipidomics_ML_Models"
    shap_max_display: int = 10
    
    @property
    def preprocessing_artifact_path(self) -> str:
        """
        Returns the full path to the preprocessing artifact file.
        """
        return os.path.join(self.models_dir, "preprocessing.joblib")
