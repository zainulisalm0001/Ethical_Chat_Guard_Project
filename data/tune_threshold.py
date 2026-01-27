import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from services.sbert_lr import predict_proba

def find_best_threshold(texts, labels):
    probs = np.array([predict_proba(t) for t in texts])
    labels = np.array(labels)

    best_th = 0.5
    best_f1 = 0.0

    for th in np.linspace(0.05, 0.95, 37):
        preds = (probs >= th).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return best_th, best_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--text_col", required=True)
    parser.add_argument("--label_col", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text_col, args.label_col])

    texts = df[args.text_col].astype(str).tolist()
    labels = df[args.label_col].astype(int).tolist()

    best_th, best_f1 = find_best_threshold(texts, labels)

    result = {
        "threshold": float(best_th),
        "best_f1": float(best_f1),
        "rows_used": len(texts)
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved threshold to:", args.out)
    print(result)

if __name__ == "__main__":
    main()