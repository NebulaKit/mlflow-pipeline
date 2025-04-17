import argparse
from pipeline.config import Config
from pipeline.utils import override_config_from_args
from pipeline.runner import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--label_col", type=str)
    parser.add_argument("--test_size", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cv_folds", type=int)
    parser.add_argument("--models_to_run", type=str, nargs='*')
    parser.add_argument("--experiment_name", type=str)
    args = parser.parse_args()

    # Start with default config
    default_config = Config()

    # Override using CLI args
    config = override_config_from_args(default_config, args)
    
    run_pipeline(config)
