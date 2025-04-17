import os
import joblib
import mlflow
import pandas as pd
from pipeline.config import Config
from pipeline.data_loader import load_data
from pipeline.preprocessing import preprocess_features, preprocess_target
from pipeline.utils import save_preprocessing_artifacts
from pipeline.model_registry import get_classifiers
from pipeline.evaluator import evaluate_cv, evaluate_test
from pipeline.explainer import explain_model

def run_pipeline(config: Config):
    mlflow.set_experiment(config.experiment_name)

    # Load and preprocess
    X_raw, y_raw = load_data(config.data_path, config.label_col)
    X, le_dict, scaler = preprocess_features(X_raw)
    y, label_encoder = preprocess_target(y_raw)
    save_preprocessing_artifacts(
        le_dict,
        label_encoder,
        scaler,
        output_path=config.preprocessing_artifact_path
    )

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.seed
    )

    classifiers = get_classifiers()
    to_run = config.models_to_run or classifiers.keys()

    for name in to_run:
        print(f'Running: {name}')
        with mlflow.start_run(run_name=name):
            model = classifiers[name]
            mlflow.log_param("model", name)
            mlflow.log_param("seed", config.seed)
            mlflow.log_param("cv_folds", config.cv_folds)

            # CV
            cv_metrics = evaluate_cv(model, X_train, y_train, folds=config.cv_folds)
            for m, val in cv_metrics.items():
                mlflow.log_metric(f"cv_{m.lower()}", val["mean"])

            # Fit & test
            model.fit(X_train, y_train)
            test_metrics = evaluate_test(model, X_test, y_test)
            for m, val in test_metrics.items():
                mlflow.log_metric(f"test_{m.lower()}", val)

            # Save model
            model_path = os.path.join(config.model_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

            # SHAP
            shap_path = os.path.join(config.output_dir, f"{name}_shap.png")
            class_names = (
                label_encoder.inverse_transform(model.classes_).tolist()
                if label_encoder is not None
                else [str(cls) for cls in model.classes_]
            )
            shap_paths = explain_model(model, X_test, output_path=shap_path, class_names=class_names)
            for p in shap_paths:
                mlflow.log_artifact(p)
