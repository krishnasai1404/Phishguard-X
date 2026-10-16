"""Dataset builder.
Real data (put in data/):  phishtank.csv (columns: url, submission_time)
                           tranco.csv    (no header: rank,domain)
Output: DataFrame[url, label, ts]  (label 1 = phishing)

LIMITATIONS to state in the paper:
 * Tranco has no timestamps, so benign rows get random dates over the phishing time range.
 * Tranco entries are bare domains, so by default phishing URLs are reduced to scheme+host
   (host_only=True). Otherwise a model can cheat by learning "has a path => phishing".
"""
import numpy as np, pandas as pd
from urllib.parse import urlparse


def _host_only(u: str) -> str:
    p = urlparse(u if "://" in u else "http://" + u)
    return f"{p.scheme}://{p.netloc}"


def load_real(phish_csv="data/phishtank.csv", tranco_csv="data/tranco.csv",
              benign_ratio=1.0, host_only=True, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ph = pd.read_csv(phish_csv)[["url", "submission_time"]].dropna()
    ph["ts"] = pd.to_datetime(ph["submission_time"], utc=True, errors="coerce").dt.tz_localize(None)
    ph = ph.dropna(subset=["ts"]).drop_duplicates("url")
    ph["label"] = 1
    if host_only:
        ph["url"] = ph["url"].map(_host_only)
    tr = pd.read_csv(tranco_csv, header=None, names=["rank", "domain"])
    be = tr.sample(n=min(int(len(ph) * benign_ratio), len(tr)), random_state=seed).copy()
    span = (ph["ts"].max() - ph["ts"].min()).total_seconds()
    be["ts"] = ph["ts"].min() + pd.to_timedelta(rng.uniform(0, span, len(be)), unit="s")
    be["url"] = "https://" + be["domain"]
    be["label"] = 0
    df = pd.concat([ph[["url", "label", "ts"]], be[["url", "label", "ts"]]], ignore_index=True)
    return df.sort_values("ts").reset_index(drop=True)


def make_synthetic(n_per_month=400, months=12, seed=0) -> pd.DataFrame:
    """Fake data with built-in drift, ONLY to test that the pipeline runs. Never report results from it."""
    rng = np.random.default_rng(seed)
    words = ["login", "secure", "verify", "account", "update", "bank", "paypal", "wallet"]
    benign = ["google", "wikipedia", "github", "nytimes", "amazon", "youtube", "reddit", "bbc"]
    old, new = [".com", ".net", ".tk", ".ml"], [".com", ".xyz", ".top", ".click"]
    chars, rows, start = list("abcdefghijklmnopqrstuvwxyz0123456789"), [], pd.Timestamp("2025-01-01")
    for m in range(months):
        drift = m / max(months - 1, 1)
        for _ in range(n_per_month):
            ts = start + pd.Timedelta(days=30 * m + int(rng.integers(0, 30)))
            tld = rng.choice(new if rng.random() < drift else old)
            w = "-".join(rng.choice(words, size=int(rng.integers(1, 3 + int(drift * 2)))))
            rnd = "".join(rng.choice(chars, int(rng.integers(3, 6 + int(drift * 8)))))
            rows.append((f"http://{w}-{rnd}{tld}", 1, ts))
            rows.append((f"https://www.{rng.choice(benign)}{rng.choice(['.com', '.org'])}", 0, ts))
    return pd.DataFrame(rows, columns=["url", "label", "ts"]).sort_values("ts").reset_index(drop=True)


def load_pages(jsonl="data/pages.jsonl") -> pd.DataFrame:
    """Load crawled pages (url,label,ts,html). Rows with empty html (dead pages) are dropped."""
    df = pd.read_json(jsonl, lines=True)
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df = df[df["html"].fillna("").str.len() > 0]
    return df.sort_values("ts").reset_index(drop=True)


def make_synthetic_multimodal(n_per_month=400, months=12, seed=0, noise=0.2) -> pd.DataFrame:
    """Fake URL+HTML data with drift and URL-appearance noise. PIPELINE TESTING ONLY - never report it."""
    rng = np.random.default_rng(seed)
    df = make_synthetic(n_per_month, months, seed)
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789")
    t0, span = df["ts"].min(), (df["ts"].max() - df["ts"].min()).total_seconds()
    urls, htmls = [], []
    for u, y, ts in zip(df["url"], df["label"], df["ts"]):
        drift = (ts - t0).total_seconds() / span
        if rng.random() < noise:                                   # URL looks like the opposite class
            rnd = "".join(rng.choice(chars, 5))
            u = f"https://www.shop-{rnd}.com" if y == 1 else f"http://login-{rnd}.xyz"
        host = u.split("//")[1]
        p_pw, p_ext = (0.8 - 0.3 * drift, 0.7) if y == 1 else (0.2, 0.1)     # attackers adapt over time
        form = ""
        if rng.random() < p_pw:
            act = f"http://collect-{rng.integers(999)}.top/p.php" if rng.random() < p_ext else "/login"
            form = f'<form action="{act}"><input type="text"><input type="password"></form>'
        n_ext = int(rng.integers(0, 6)) if y == 1 else int(rng.integers(0, 15))
        links = "".join(f'<a href="https://other{i}.com">x</a>' for i in range(n_ext)) + '<a href="#">x</a>' * int(y * rng.integers(0, 4))
        title = rng.choice(["Sign in", "Welcome", "Account", "Home"]) + (" - PayPal" if y == 1 and rng.random() < 0.3 else "")
        urls.append(u); htmls.append(f"<html><head><title>{title}</title></head><body>{form}{links}</body></html>")
    df["url"], df["html"] = urls, htmls
    return df
