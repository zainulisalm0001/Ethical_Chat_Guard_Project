import json
from pathlib import Path

DEFAULT_THRESHOLD = 0.55
MIN_THRESHOLD = 0.45
MAX_THRESHOLD = 0.95

def load_threshold(path: str = "models/threshold.json") -> float:
    p = Path(path)
    if not p.exists():
        return DEFAULT_THRESHOLD
    try:
        obj = json.loads(p.read_text())
        th = float(obj.get("threshold", DEFAULT_THRESHOLD))
    except Exception:
        return DEFAULT_THRESHOLD
    th = max(MIN_THRESHOLD, min(MAX_THRESHOLD, th))
    return th