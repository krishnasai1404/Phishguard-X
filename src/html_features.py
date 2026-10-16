"""Webpage (HTML/DOM) features. Static parsing only: no JavaScript is ever executed."""
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

BRANDS = ["paypal", "google", "microsoft", "apple", "amazon", "facebook", "netflix", "dhl",
          "fedex", "instagram", "whatsapp", "sbi", "hdfc", "icici", "outlook", "dropbox"]


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.forms, self.inputs, self.links, self.scripts, self.imgs, self.icons = [], [], [], [], [], []
        self.script_text, self.title = [], ""
        self.iframes = self.mouseover = self.text_len = 0
        self.meta_refresh = self.ctx_menu = self._in_title = self._in_script = False

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if "oncontextmenu" in a: self.ctx_menu = True
        if "onmouseover" in a: self.mouseover += 1
        if tag == "form": self.forms.append(a.get("action", ""))
        elif tag == "input": self.inputs.append(a.get("type", "text").lower())
        elif tag == "a": self.links.append(a.get("href", ""))
        elif tag == "script":
            self._in_script = True
            if a.get("src"): self.scripts.append(a["src"])
        elif tag == "iframe": self.iframes += 1
        elif tag == "img": self.imgs.append(a.get("src", ""))
        elif tag == "link" and "icon" in a.get("rel", "").lower(): self.icons.append(a.get("href", ""))
        elif tag == "meta" and a.get("http-equiv", "").lower() == "refresh": self.meta_refresh = True
        elif tag == "title": self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title": self._in_title = False
        if tag == "script": self._in_script = False

    def handle_data(self, d):
        if self._in_title: self.title += d
        elif self._in_script: self.script_text.append(d)
        else: self.text_len += len(d.strip())


def extract(html: str, page_url: str) -> dict:
    c = _Collector()
    try:
        c.feed(html or "")
    except Exception:
        pass
    host = (urlparse(page_url if "://" in page_url else "http://" + page_url).hostname or "").removeprefix("www.")

    def ext(u):
        if not u.startswith(("http://", "https://", "//")): return 0
        h = (urlparse("http:" + u if u.startswith("//") else u).hostname or "").removeprefix("www.")
        return int(h != "" and h != host)

    js = " ".join(c.script_text).lower()
    nl, ns, ni = len(c.links), len(c.scripts), len(c.imgs)
    title = c.title.lower()
    return {
        "n_forms": len(c.forms),
        "has_password": int("password" in c.inputs),
        "n_inputs": len(c.inputs), "n_hidden": c.inputs.count("hidden"),
        "form_action_external": int(any(ext(f) for f in c.forms)),
        "form_action_empty": int(any(f in ("", "#", "about:blank") for f in c.forms)),
        "n_links": nl, "ext_link_ratio": sum(ext(l) for l in c.links) / max(nl, 1),
        "null_link_ratio": sum(l in ("", "#", "javascript:void(0)") for l in c.links) / max(nl, 1),
        "n_scripts": ns, "ext_script_ratio": sum(ext(s) for s in c.scripts) / max(ns, 1),
        "n_iframes": c.iframes, "meta_refresh": int(c.meta_refresh),
        "right_click_disabled": int(c.ctx_menu), "n_mouseover": c.mouseover,
        "n_images": ni, "ext_img_ratio": sum(ext(i) for i in c.imgs) / max(ni, 1),
        "favicon_external": int(any(ext(i) for i in c.icons)),
        "title_len": len(title), "text_len": c.text_len,
        "js_obfuscation": js.count("eval(") + js.count("atob(") + js.count("fromcharcode") + js.count("unescape("),
        "js_doc_write": js.count("document.write"),
        "brand_title_mismatch": int(any(b in title and b not in host for b in BRANDS)),
    }
