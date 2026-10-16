"""Train the final fusion model on all data and save it for the demo.
    python -m src.export_model --synthetic | --real"""
import sys, joblib
from .experiments import MODELS, GB_NAME
from .data import load_pages, make_synthetic_multimodal
from .fusion import FusionModel
from .run_phishguardx import build

df = make_synthetic_multimodal() if "--synthetic" in sys.argv else load_pages()
Xu, Xh, y, _ = build(df)
fm = FusionModel(MODELS[GB_NAME]).fit(Xu, Xh, y)
joblib.dump({"fusion": fm, "url_cols": list(Xu.columns), "html_cols": list(Xh.columns)}, "results/model.joblib")
print("saved results/model.joblib")
