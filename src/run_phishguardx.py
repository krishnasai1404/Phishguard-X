"""PhishGuard-X experiments (URL + HTML).
    python -m src.run_phishguardx --synthetic     # pipeline smoke test only
    python -m src.run_phishguardx --real          # reads data/pages.jsonl (from src.crawler)
Writes results/<tag>mm_*.csv"""
import argparse
import numpy as np, pandas as pd
from . import features, html_features
from .experiments import MODELS, GB_NAME, metrics, exp_retrain, plot_retrain
from .data import load_pages, make_synthetic_multimodal
from .fusion import FusionModel


def build(df):
    Xu = pd.DataFrame([features.extract(u) for u in df["url"]])
    Xh = pd.DataFrame([html_features.extract(h, u) for h, u in zip(df["html"], df["url"])])
    return Xu, Xh, df["label"].to_numpy(), df["ts"].dt.to_period("M").astype(str).to_numpy()


def exp_modality(Xu, Xh, y, split=0.7):
    """RQ1/RQ5: URL only vs HTML only vs URL+HTML (early fusion) vs URL+HTML (late fusion), temporal split."""
    cut = int(len(y) * split); mk = MODELS[GB_NAME]; rows = []
    Xe = pd.concat([Xu, Xh], axis=1)
    for name, X in [("URL only", Xu), ("HTML only", Xh), ("URL+HTML (early fusion)", Xe)]:
        m = mk().fit(X.iloc[:cut], y[:cut])
        rows.append({"modality": name, **metrics(y[cut:], m.predict_proba(X.iloc[cut:])[:, 1])})
    f = FusionModel(mk).fit(Xu.iloc[:cut], Xh.iloc[:cut], y[:cut])
    rows.append({"modality": "URL+HTML (late fusion)", **metrics(y[cut:], f.predict_proba(Xu.iloc[cut:], Xh.iloc[cut:]))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", action="store_true"); g.add_argument("--real", action="store_true")
    a = ap.parse_args()
    df = make_synthetic_multimodal() if a.synthetic else load_pages()
    tag = "SYNTHETIC_" if a.synthetic else ""
    print(f"{len(df)} pages, {df.ts.min().date()} -> {df.ts.max().date()}, phishing share {df.label.mean():.2f}")
    Xu, Xh, y, month = build(df)

    r = exp_modality(Xu, Xh, y); r.to_csv(f"results/{tag}mm_modality.csv", index=False)
    print("\n== modality ablation (temporal split) ==\n", r.round(3).to_string(index=False))

    Xe = pd.concat([Xu, Xh], axis=1)
    rs = [exp_retrain(Xe, y, month, drift=d) for d in ("psi", "ks")]
    rr = pd.concat(rs); rr.to_csv(f"results/{tag}mm_retrain.csv", index=False)
    summ = (rr.groupby(["strategy", "drift_method"]).agg(f1=("f1", "mean"), recall=("recall", "mean"),
            fpr=("fpr", "mean"), retrains=("retrained", "sum")).round(3))
    print("\n== retraining strategies (early-fusion model) ==\n", summ)
    plot_retrain(rs[0], f"results/{tag}mm_retrain_f1.png")
