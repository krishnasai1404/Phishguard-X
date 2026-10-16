"""Lexical URL features (no network calls, so they are fast and reproducible)."""
import math, re
from collections import Counter
from urllib.parse import urlparse

KEYWORDS = ["login", "secure", "account", "verify", "update", "bank", "signin",
            "confirm", "paypal", "password", "wallet", "support", "webscr"]
SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly")
SUSPICIOUS_TLDS = (".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".gq", ".click", ".work", ".zip")
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def extract(url: str) -> dict:
    u = url if "://" in url else "http://" + url
    p = urlparse(u)
    host, path, query = (p.hostname or ""), p.path, p.query
    parts = host.split(".")
    nd = sum(c.isdigit() for c in url)
    return {
        # length group
        "url_len": len(url), "host_len": len(host), "path_len": len(path), "query_len": len(query),
        # character-count group
        "n_dots": url.count("."), "n_hyphens": url.count("-"), "n_underscores": url.count("_"),
        "n_slashes": url.count("/"), "n_at": url.count("@"), "n_qmarks": url.count("?"),
        "n_equals": url.count("="), "n_amp": url.count("&"), "n_percent": url.count("%"),
        "n_digits": nd, "digit_ratio": nd / max(len(url), 1),
        # host structure group
        "n_subdomains": max(len(parts) - 2, 0), "has_ip": int(bool(IP_RE.match(host))),
        "host_hyphens": host.count("-"), "host_digits": sum(c.isdigit() for c in host),
        "tld_len": len(parts[-1]) if parts else 0,
        "suspicious_tld": int(host.endswith(SUSPICIOUS_TLDS)),
        "is_shortener": int(host in SHORTENERS), "has_port": int(p.port is not None),
        # statistical group
        "url_entropy": entropy(url), "host_entropy": entropy(host), "path_entropy": entropy(path),
        # semantic group
        "n_keywords": sum(k in url.lower() for k in KEYWORDS),
        "https": int(p.scheme == "https"), "path_depth": len([x for x in path.split("/") if x]),
    }


FEATURE_GROUPS = {
    "length": ["url_len", "host_len", "path_len", "query_len"],
    "char_counts": ["n_dots", "n_hyphens", "n_underscores", "n_slashes", "n_at", "n_qmarks",
                    "n_equals", "n_amp", "n_percent", "n_digits", "digit_ratio"],
    "host_structure": ["n_subdomains", "has_ip", "host_hyphens", "host_digits", "tld_len",
                       "suspicious_tld", "is_shortener", "has_port"],
    "entropy": ["url_entropy", "host_entropy", "path_entropy"],
    "semantic": ["n_keywords", "https", "path_depth"],
}
