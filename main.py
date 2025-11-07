import argparse
from pipeline.config import Config
from pipeline.utils import override_config_from_args
from pipeline.runner import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--label_col", type=str)
    parser.add_argument("--control_name", type=str)
    parser.add_argument("--feature_map_path", type=str)
    parser.add_argument("--scaling_method", type=str, choices=['standard', 'log2', 'log10', 'minmax', 'robust', 'maxabs', 'quantile'], default='standard')
    parser.add_argument("--test_size", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cv_folds", type=int)
    parser.add_argument("--models_to_run", type=str, nargs='*')
    parser.add_argument("--do_grid_search", type=bool)
    parser.add_argument("--grid_search_metric", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--models_dir", type=str)   
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--shap_max_display", type=int)
    args = parser.parse_args()

    # Start with default config
    default_config = Config()

    # Override using CLI args
    config = override_config_from_args(default_config, args)
    
    run_pipeline(config)
