"""Run all experiments:  python -m src.experiments --synthetic   (pipeline test only)
                        python -m src.experiments --real        (your actual results)
Outputs CSVs/plots into results/."""
import argparse, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .features import extract, FEATURE_GROUPS
from .data import load_real, make_synthetic

SEED = 42
try:
    from xgboost import XGBClassifier
    def make_gb(): return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=SEED)
    GB_NAME = "XGBoost"
except ImportError:                                   # fallback so the code still runs without xgboost
    def make_gb(): return HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
    GB_NAME = "HistGradBoost"

MODELS = {
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
    GB_NAME: make_gb,
}


def metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "fpr": fp / max(fp + tn, 1),
            "auc": roc_auc_score(y, prob) if len(set(y)) > 1 else np.nan}


def featurize(df):
    X = pd.DataFrame([extract(u) for u in df["url"]])
    return X, df["label"].to_numpy(), df["ts"].dt.to_period("M").astype(str).to_numpy()


def psi(ref, cur, bins=10):
    """Population Stability Index between two score distributions (label-free drift signal)."""
    edges = np.linspace(0, 1, bins + 1)
    r = np.histogram(ref, edges)[0] / max(len(ref), 1) + 1e-4
    c = np.histogram(cur, edges)[0] / max(len(cur), 1) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


# ---------- Exp 1: random vs temporal split ----------
def exp_split(X, y, ts_order):
    rows = []
    n = len(X); cut = int(n * 0.7)
    for name, mk in MODELS.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
        m = mk().fit(Xtr, ytr); rows.append({"model": name, "split": "random", **metrics(yte, m.predict_proba(Xte)[:, 1])})
        idx = np.argsort(ts_order, kind="stable")           # data is already time-sorted; kept explicit
        tr, te = idx[:cut], idx[cut:]
        m = mk().fit(X.iloc[tr], y[tr]); rows.append({"model": name, "split": "temporal", **metrics(y[te], m.predict_proba(X.iloc[te])[:, 1])})
    return pd.DataFrame(rows)


# ---------- Exp 2: retraining strategies ----------
def drift_signal(ref, cur, method="psi", psi_thr=0.2, ks_alpha=0.01):
    """Label-free drift check on model score distributions. Returns (statistic, triggered)."""
    if method == "ks":
        from scipy.stats import ks_2samp
        r = ks_2samp(ref, cur)
        return float(r.statistic), bool(r.pvalue < ks_alpha)
    d = psi(ref, cur)
    return d, d > psi_thr


def exp_retrain(X, y, month, model_name=GB_NAME, init_months=3, drift="psi", psi_thr=0.2):
    months = sorted(set(month))
    mk = MODELS[model_name]
    def fit_at(t):                                           # train on all data before month index t
        past = np.isin(month, months[:t]); hold = month == months[t - 1]
        fit_mask = past & ~hold
        ref = mk().fit(X[fit_mask], y[fit_mask]).predict_proba(X[hold])[:, 1]   # hold out last month for reference scores
        return mk().fit(X[past], y[past]), ref
    out = []
    for strat in ["static", "monthly", "drift_triggered"]:
        model, ref = fit_at(init_months)
        for t in range(init_months, len(months)):
            cur = month == months[t]
            prob = model.predict_proba(X[cur])[:, 1]
            stat, hit = drift_signal(ref, prob, drift, psi_thr)
            retrain = t + 1 < len(months) and (strat == "monthly" or (strat == "drift_triggered" and hit))
            out.append({"strategy": strat, "drift_method": drift, "month": months[t], "drift_stat": stat,
                        "drift_flag": hit, "retrained": retrain, **metrics(y[cur], prob)})
            if retrain:
                model, ref = fit_at(t + 1)
    return pd.DataFrame(out)


# ---------- Exp 3: feature-group ablation (temporal split) ----------
def exp_ablation(X, y, model_name=GB_NAME):
    cut = int(len(X) * 0.7); rows = []
    for dropped in [None] + list(FEATURE_GROUPS):
        cols = [c for c in X.columns if dropped is None or c not in FEATURE_GROUPS[dropped]]
        m = MODELS[model_name]().fit(X.iloc[:cut][cols], y[:cut])
        rows.append({"dropped_group": dropped or "none", **metrics(y[cut:], m.predict_proba(X.iloc[cut:][cols])[:, 1])})
    return pd.DataFrame(rows)


# ---------- Exp 4: latency ----------
def exp_latency(df, X, y):
    cut = int(len(X) * 0.7); rows = []
    sample = df["url"].iloc[cut:cut + 500].tolist()
    for name, mk in MODELS.items():
        m = mk().fit(X.iloc[:cut], y[:cut])
        t0 = time.perf_counter()
        for u in sample: m.predict_proba(pd.DataFrame([extract(u)]))
        rows.append({"model": name, "ms_per_url": (time.perf_counter() - t0) / len(sample) * 1000})
    return pd.DataFrame(rows)


def plot_retrain(r, path):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(7, 4))
    for s, g in r.groupby("strategy"): plt.plot(g["month"], g["f1"], marker="o", label=s)
    plt.xticks(rotation=45); plt.ylabel("F1"); plt.title("F1 over time by retraining strategy")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", action="store_true"); g.add_argument("--real", action="store_true")
    a = ap.parse_args()
    df = make_synthetic() if a.synthetic else load_real()
    tag = "SYNTHETIC_" if a.synthetic else ""
    print(f"{len(df)} rows, {df.ts.min().date()} -> {df.ts.max().date()}, phishing share {df.label.mean():.2f}")
    X, y, month = featurize(df)
    for name, fn in [("split", lambda: exp_split(X, y, df["ts"].to_numpy())),
                     ("retrain", lambda: exp_retrain(X, y, month)),
                     ("ablation", lambda: exp_ablation(X, y)),
                     ("latency", lambda: exp_latency(df, X, y))]:
        r = fn(); r.to_csv(f"results/{tag}{name}.csv", index=False)
        print(f"\n== {name} ==\n{r.round(3).to_string(index=False) if name != 'retrain' else r.groupby('strategy')[['f1','recall','fpr']].mean().round(3)}")
        if name == "retrain": plot_retrain(r, f"results/{tag}retrain_f1.png")
