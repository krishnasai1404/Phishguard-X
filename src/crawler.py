"""Collect raw HTML for URLs.  python -m src.crawler data/urls.csv data/pages.jsonl
Input CSV columns: url,label   (timestamp is taken at crawl time, which gives REAL timestamps for both classes)

SAFETY: fetches static HTML text only (no JS execution, no rendering, size-capped).
Run it inside a VM/container, never open the saved pages in a browser, and never click anything.
TIP: phishing pages die within days. Crawl fresh PhishTank/OpenPhish entries daily, and crawl benign
Tranco sites on the same days. Run this for 4-6 weeks to build your own temporal dataset."""
import csv, json, sys, urllib.request, datetime as dt
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0 (research crawler; academic use)"}


def fetch(row):
    url = row["url"] if "://" in row["url"] else "http://" + row["url"]
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=8) as r:
            html = r.read(1_000_000).decode("utf-8", errors="ignore")
    except Exception:
        html = ""
    return {"url": row["url"], "label": int(row["label"]), "ts": dt.datetime.utcnow().isoformat(), "html": html}


if __name__ == "__main__":
    rows = list(csv.DictReader(open(sys.argv[1])))
    with ThreadPoolExecutor(16) as ex, open(sys.argv[2], "a") as out:
        for rec in ex.map(fetch, rows):
            out.write(json.dumps(rec) + "\n")
