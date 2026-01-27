import os
import joblib
from sentence_transformers import SentenceTransformer

_EMBEDDER = None
_MODEL = None

def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER

def _get_model():
    global _MODEL
    if _MODEL is None:
        path = os.path.join("models", "lr_coercion.joblib")
        _MODEL = joblib.load(path)
    return _MODEL

def predict_proba(text: str) -> float:
    embedder = _get_embedder()
    model = _get_model()
    X = embedder.encode([str(text)], convert_to_numpy=True, show_progress_bar=False)
    p = model.predict_proba(X)[0, 1]
    return float(p)