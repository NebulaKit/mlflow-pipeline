from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import numpy as np


def get_classifiers(seed: int):
    return {
        'XGBoost': (
           XGBClassifier(eval_metric='logloss', random_state=seed),
           {
               'n_estimators': [30, 50, 100],
               'max_depth': [2, 3, 5],
               'learning_rate': [0.2, 0.3, 0.35]
           }
        ),
        'LASSO': (
            LogisticRegression(penalty="l1", solver="saga", max_iter=20000, random_state=seed),
            {
                "C": np.logspace(-2, 2, 5),
                "class_weight": [None, "balanced"]
            }     
        ),
        # 'ElasticNet': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000),
        # 'KNN': KNeighborsClassifier(),
        # 'SVM': SVC(probability=True),
        'DecisionTree': (
            DecisionTreeClassifier(random_state=seed),
            {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=seed), 
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        ),
        'ExtraTrees': (
            ExtraTreesClassifier(random_state=seed),
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        ),
        "MLP": ( 
            MLPClassifier(max_iter=1000, random_state=seed, early_stopping=True, n_iter_no_change=15, validation_fraction=0.15), 
            { 
             "hidden_layer_sizes": [ (64,), (128,), (64, 32), (128, 64), ],
             "activation": ["relu", "tanh"],
             "alpha": [1e-5, 1e-4, 1e-3],
             "learning_rate_init": [1e-4, 3e-4, 1e-3],
             "solver": ["adam"], 
             "batch_size": ["auto", 32, 64, 128],
             "beta_1": [0.9, 0.95]
            }
        )
    }
