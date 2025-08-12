from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import numpy as np


def evaluate_cv(model, X, y, folds):
    """
    Evaluate a model using stratified cross-validation.
    Calculates average AUC, standard error, and 95% CI.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs = []

    is_multiclass = len(np.unique(y)) > 2

    for train_idx, val_idx in skf.split(X, y):
        X_train_cv, X_val = X[train_idx], X[val_idx]
        y_train_cv, y_val = y[train_idx], y[val_idx]

        model.fit(X_train_cv, y_train_cv)
        y_val_proba = model.predict_proba(X_val)

        if is_multiclass:
            auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(y_val, y_val_proba[:, 1])

        aucs.append(auc)

    aucs = np.array(aucs)
    avg_auc = np.mean(aucs)
    stderr = np.std(aucs, ddof=1) / np.sqrt(len(aucs))
    ci_low, ci_high = stats.t.interval(0.95, len(aucs) - 1, loc=avg_auc, scale=stderr)

    return {
        "AUC": {
            "aucs": aucs,
            "mean": avg_auc,
            "sem": stderr,
            "ci_lower": ci_low,
            "ci_upper": ci_high
        }
    }


def evaluate_test(model, X_test, y_test):
    """
    Evaluate a trained model on the held-out test set.
    Returns classification metrics including AUC, accuracy, precision, recall, and F1.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    is_multiclass = len(np.unique(y_test)) > 2

    if is_multiclass:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    else:
        auc = roc_auc_score(y_test, y_proba[:, 1])

    return {
        "AUC": auc,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro" if is_multiclass else "binary"),
        "Recall": recall_score(y_test, y_pred, average="macro" if is_multiclass else "binary"),
        "F1": f1_score(y_test, y_pred, average="macro" if is_multiclass else "binary")
    }
