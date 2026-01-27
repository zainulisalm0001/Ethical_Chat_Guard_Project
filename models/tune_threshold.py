import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from services.sbert_lr import predict_proba

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text_col", default="assistant_reply")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--out", default="models/coercion_threshold.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[[args.text_col, args.label_col]].dropna()
    df[args.label_col] = df[args.label_col].astype(int)

    _, val = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[args.label_col],
    )

    y = val[args.label_col].values
    texts = val[args.text_col].astype(str).tolist()
    probs = [float(predict_proba(t)) for t in texts]

    best = {"threshold": 0.5, "f1": -1.0}
    for th in [i / 100 for i in range(10, 91)]:
        preds = [1 if p >= th else 0 for p in probs]
        f1 = f1_score(y, preds)
        if f1 > best["f1"]:
            best = {"threshold": th, "f1": float(f1)}

    payload = {
        "threshold": best["threshold"],
        "val_f1": best["f1"],
        "n_val": int(len(val)),
        "text_col": args.text_col,
        "label_col": args.label_col,
        "csv": args.csv,
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()