import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pipeline.config import Config
from pipeline.data_loader import load_data
from pipeline.preprocessing.preprocessor import Preprocessor
from pipeline.preprocessing.target_preprocessor import preprocess_target
from pipeline.utils import save_preprocessing_artifacts, resolve_scoring
from pipeline.model_registry import get_classifiers
from pipeline.evaluator import evaluate_cv, evaluate_test
from pipeline.explainer import explain_model
from sklearn.model_selection import train_test_split
from pipeline.differential_expression_analysis import plot_de_boxplots_for_top_features
from pipeline.shap_aggregate import aggregate_shap_and_get_top
from sklearn.base import clone


def run_pipeline(config: Config):
    mlflow.set_experiment(config.experiment_name)
    
    # Enable MLflow autologging for sklearn
    mlflow.sklearn.autolog()
    # Does not support polars dataframes yet

    # Load data
    X_raw, y_raw = load_data(config.data_path, config.label_col)
    
    # Split
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=config.test_size, stratify=y_raw, random_state=config.seed
    )
    
    # Target preprocessing
    y_train, label_encoder = preprocess_target(y_train_raw)
    y_test, _ = preprocess_target(y_test_raw)
    n_classes = len(np.unique(y_train))
    print(f"label_encoder: {label_encoder}")
    if label_encoder is not None:
        class_names = label_encoder.inverse_transform(np.arange(n_classes)).tolist()
    else:
        class_names = np.unique(y_train).tolist()
    print(f"Classes detected: {class_names}")
    
    # Feature preprocessing
    preprocessor = Preprocessor(scaling_method=config.scaling_method)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Save preprocessing artifacts
    data_file_name = os.path.splitext(os.path.basename(config.data_path))[0]
    save_preprocessing_artifacts(
        preprocessor.label_encoders,
        label_encoder,
        preprocessor.scaler,
        dir_name=config.models_dir,
        data_file_name=data_file_name
    )
    mlflow.log_param("scaling_method", config.scaling_method)

    
    classifiers = get_classifiers(config.seed)
    to_run = config.models_to_run or classifiers.keys()
    
    df_cv = pd.DataFrame(columns=['Classifier', 'AUCS'])
    results = []

    for name in to_run:
        mlflow.end_run()  # End previous run if any
        print(f'Running: {name}')
        with mlflow.start_run(run_name=name):
            model, param_grid = classifiers[name]
            
            mlflow.log_param("model", name)
            mlflow.log_param("seed", config.seed)
            mlflow.log_param("cv_folds", config.cv_folds)
            
            
            if config.do_grid_search and param_grid:
                from sklearn.model_selection import GridSearchCV
                scoring = resolve_scoring(y_train, config.grid_search_metric)
                print(f"Performing grid search for {name} optimizing {scoring}...")
                mlflow.log_param("grid_search", True)
                mlflow.log_param("grid_search_metric", scoring)
                grid = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=config.cv_folds,
                    n_jobs=-1,
                    verbose=1
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                
                mlflow.log_params(grid.best_params_)
                print("Best parameters found: ", grid.best_params_)
                mlflow.log_metric("cv_best_score", grid.best_score_) # TODO: not logged properly
                
                # Get an unfitted clone with the same hyperparameters
                model = clone(best_model)
                    

            # CV
            cv_metrics = evaluate_cv(model, X_train, y_train, folds=config.cv_folds)
            for m, val in cv_metrics.items():
                mlflow.log_metric(f"cv_{m.lower()}", val["mean"])
                print(f"cv_{m.lower()}s", val["aucs"])
                print(f"cv_{m.lower()}_avg", val["mean"])
                df_cv = pd.concat([df_cv, pd.DataFrame([{'Classifier': name, 'AUCS': val["aucs"]}])], ignore_index=True)

            # Fit & test
            model.fit(X_train, y_train)
            test_metrics = evaluate_test(model, X_test, y_test)
            for m, val in test_metrics.items():
                mlflow.log_metric(f"test_{m.lower()}", val)
                print(f"test_{m.lower()}", val)
                
            results.append({
                'Classifier': name,
                'AUC': test_metrics['AUC'],
                'Accuracy': test_metrics['Accuracy'],
                'Precision': test_metrics['Precision'],
                'Recall': test_metrics['Recall'],
                'F1': test_metrics['F1']
            })

            # Save model
            model_path = os.path.join(config.models_dir, data_file_name,  f"{name}.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

            # SHAP
            print(f"Explaining model {name} with SHAP...")
            print(f"Class names: {class_names}")
            shap_dir = os.path.join(config.output_dir, data_file_name, "shap", name)
            os.makedirs(shap_dir, exist_ok=True)
            shap_paths, top_feats  = explain_model(model,
                                       name,
                                       X_test,
                                       output_dir=shap_dir,
                                       class_names=class_names,
                                       feature_map_path = config.feature_map_path,
                                       max_plot_display=config.shap_max_display)
            
            
            # DE analysis boxplots for top SHAP features
            print(f"Preparing DE analysis boxplots for top SHAP features: {top_feats}")
            class_plot_paths = plot_de_boxplots_for_top_features(
                top_features_per_class=top_feats,
                X_raw=X_raw,
                y_raw=y_raw,
                group_col_name=config.label_col,
                control_name=config.control_name,
                output_dir=shap_dir,
                model_name=name,
                feature_map_path=config.feature_map_path,
                use_log2_fc=True,
                facet_cols=5
            )
            
            # Log to MLflow
            for p in shap_paths:
                mlflow.log_artifact(p)
            for p in class_plot_paths.values():
                mlflow.log_artifact(p)
    

    # Save CV results
    cv_results_path = os.path.join(config.output_dir, data_file_name, "cv_results.csv")
    df_cv.to_csv(cv_results_path, index=False)
    mlflow.log_artifact(cv_results_path)
    
    # Save final model results
    final_results_path = os.path.join(config.output_dir, data_file_name, "final_results.csv")
    df_final = pd.DataFrame(results)
    df_final.to_csv(final_results_path, index=False)
    mlflow.log_artifact(final_results_path)
    
    # Aggregate SHAP values across classifiers and perform DE analysis
    print("Aggregating SHAP values across classifiers and extracting biomarkers...")
    shap_root = os.path.join(config.output_dir, data_file_name, "shap")
    agg_dir = os.path.join(config.output_dir, data_file_name, "shap", "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    top_feats_agg = aggregate_shap_and_get_top(
        shap_root_dir=shap_root,
        class_names=class_names,
        top_n=config.shap_max_display
    )

    agg_class_plot_paths = plot_de_boxplots_for_top_features(
        top_features_per_class=top_feats_agg,
        X_raw=X_raw,
        y_raw=y_raw,
        group_col_name=config.label_col,
        control_name=config.control_name,
        output_dir=agg_dir,
        model_name="Cross-Model Consensus",
        feature_map_path=config.feature_map_path,
        use_log2_fc=True,
        facet_cols=5
    )
    for p in agg_class_plot_paths.values():
        mlflow.log_artifact(p)
