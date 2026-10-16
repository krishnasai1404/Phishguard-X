import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from .experiments import MODELS, GB_NAME, psi


def _metrics(y_true, prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "fpr": fp / max(fp + tn, 1),
        "auc": roc_auc_score(y_true, prob) if np.unique(y_true).size == 2 else np.nan,
        "n": int(len(y_true)),
        "n_phishing": int(np.sum(y_true == 1)),
        "n_legitimate": int(np.sum(y_true == 0)),
    }


def exp_retrain_v3(
    X, y, month, model_name=GB_NAME, init_months=3,
    psi_threshold=2.0, prediction_threshold=0.5, min_reference_size=30
):
    """Compare static, monthly, and PSI-triggered retraining chronologically.

    Protocol: train on the first init_months; evaluate each later month before
    updating; assume that month's labels become available after evaluation.
    Monthly retrains after each evaluated month. Drift-triggered retrains only
    when PSI against the fixed baseline exceeds psi_threshold. The threshold
    must be selected using separate validation data, not these evaluation months.
    """
    y = np.asarray(y, dtype=int)
    month = np.asarray(month).astype(str)
    if len(X) != len(y) or len(y) != len(month):
        raise ValueError("X, y, and month must have equal lengths.")
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    months = sorted(np.unique(month))
    if len(months) <= init_months:
        raise ValueError("Need more months than init_months.")
    initial_mask = np.isin(month, months[:init_months])
    if np.unique(y[initial_mask]).size < 2:
        raise ValueError("Initial training window must contain both classes.")

    model_factory = MODELS[model_name]
    initial_model = model_factory().fit(X.loc[initial_mask], y[initial_mask])
    reference_month = months[init_months - 1]
    reference_mask = month == reference_month
    reference_scores = initial_model.predict_proba(X.loc[reference_mask])[:, 1]
    if len(reference_scores) < min_reference_size:
        raise ValueError(
            f"Reference month {reference_month} has only {len(reference_scores)} "
            f"rows; need at least {min_reference_size}."
        )

    rows = []
    for strategy in ("static", "monthly", "drift_triggered"):
        current_model = initial_model
        seen_months = list(months[:init_months])

        for idx in range(init_months, len(months)):
            current_month = months[idx]
            current_mask = month == current_month
            X_current = X.loc[current_mask]
            y_current = y[current_mask]
            probabilities = current_model.predict_proba(X_current)[:, 1]

            drift_stat = psi(reference_scores, probabilities)
            drift_flag = bool(np.isfinite(drift_stat) and drift_stat > psi_threshold)
            has_future_month = idx < len(months) - 1
            if strategy == "monthly":
                retrain_after = has_future_month
            elif strategy == "drift_triggered":
                retrain_after = has_future_month and drift_flag
            else:
                retrain_after = False

            rows.append({
                "strategy": strategy,
                "drift_method": "psi_fixed_baseline",
                "psi_threshold": psi_threshold,
                "reference_month": reference_month,
                "month": current_month,
                "drift_stat": drift_stat,
                "drift_flag": drift_flag,
                "retrained_after_month": retrain_after,
                **_metrics(y_current, probabilities, prediction_threshold),
            })

            if retrain_after:
                seen_months.append(current_month)
                train_mask = np.isin(month, seen_months)
                current_model = model_factory().fit(X.loc[train_mask], y[train_mask])

    return pd.DataFrame(rows)


def summarize_v3(results):
    """Summarize means and retraining counts by strategy."""
    return results.groupby("strategy").agg(
        months=("month", "nunique"),
        mean_f1=("f1", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_fpr=("fpr", "mean"),
        mean_auc=("auc", "mean"),
        drift_flags=("drift_flag", "sum"),
        retraining_events=("retrained_after_month", "sum"),
    ).reset_index()