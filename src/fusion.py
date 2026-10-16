"""Late-fusion model: URL model + HTML model -> logistic meta-learner on out-of-fold probabilities.
If the page is unavailable (dead site / no HTML), it falls back to the URL model alone."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class FusionModel:
    def __init__(self, make_model, folds=5, seed=0):
        self.mk, self.folds, self.seed = make_model, folds, seed

    def fit(self, Xu, Xh, y):
        oof = np.zeros((len(y), 2))
        for tr, va in StratifiedKFold(self.folds, shuffle=True, random_state=self.seed).split(Xu, y):
            oof[va, 0] = self.mk().fit(Xu.iloc[tr], y[tr]).predict_proba(Xu.iloc[va])[:, 1]
            oof[va, 1] = self.mk().fit(Xh.iloc[tr], y[tr]).predict_proba(Xh.iloc[va])[:, 1]
        self.url_m, self.html_m = self.mk().fit(Xu, y), self.mk().fit(Xh, y)
        self.meta = LogisticRegression().fit(_logit(oof), y)
        return self

    def predict_parts(self, Xu, Xh=None):
        pu = self.url_m.predict_proba(Xu)[:, 1]
        if Xh is None:
            return {"url": pu, "html": None, "fused": pu}
        ph = self.html_m.predict_proba(Xh)[:, 1]
        return {"url": pu, "html": ph, "fused": self.meta.predict_proba(_logit(np.c_[pu, ph]))[:, 1]}

    def predict_proba(self, Xu, Xh=None):
        return self.predict_parts(Xu, Xh)["fused"]



def reasons(u: dict, h: dict | None, probability: float | None = None) -> list[str]:
    r = []

    if u["has_ip"]:
        r.append("URL uses a raw IP address")
    if u["suspicious_tld"]:
        r.append("Domain uses a TLD common in abuse")
    if u["n_keywords"] >= 2:
        r.append("URL contains several sensitive keywords")
    if u["n_subdomains"] >= 3:
        r.append("Unusually many subdomains")
    if u["n_at"]:
        r.append("URL contains '@'")
    if u["host_entropy"] > 3.8:
        r.append("Domain name has high character entropy")

    if h:
        if h["has_password"]:
            r.append("Page contains a password field")
        if h["form_action_external"]:
            r.append("Form submits data to another domain")
        if h["brand_title_mismatch"]:
            r.append("Page title and domain show a brand mismatch")
        if h["js_obfuscation"] > 2:
            r.append("Obfuscated JavaScript indicators detected")
        if h["right_click_disabled"]:
            r.append("Right-click is disabled")
        if h["null_link_ratio"] > 0.5:
            r.append("Most links have empty or unusable destinations")

    if r:
        return r

    if probability is not None and probability >= 0.8:
        return ["Model assigns high phishing probability, but the rule-based checks found no matching indicators. Treat this result cautiously."]
    if probability is not None and probability >= 0.5:
        return ["Model assigns medium phishing probability; no matching rule-based indicators were found."]
    return ["No suspicious indicators matched the available rules; this does not guarantee the page is safe."]
