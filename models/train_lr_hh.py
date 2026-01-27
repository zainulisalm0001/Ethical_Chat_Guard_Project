import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

SEED = 42

df = pd.read_csv("data/hh_coercion_weak_labels.csv")
texts = df["assistant_reply"].astype(str).tolist()
y = df["label"].astype(int).values

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

model = LogisticRegression(max_iter=4000, C=1.0)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred, digits=4))

joblib.dump(model, "models/lr_coercion.joblib")
print("Saved: models/lr_coercion.joblib")