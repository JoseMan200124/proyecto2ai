"""
Entrena un RandomForest con el CSV generado por prepare_lsm_dataset.py.
Uso:
    python -m app.models.train_gesture_model  lsm_landmarks.csv
"""

import sys, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def main(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    print(classification_report(y_te, y_pred))

    cm = confusion_matrix(y_te, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
    disp.plot(figsize=(10, 10), cmap="Blues")
    plt.tight_layout()
    plt.savefig("metrics.png")

    # Guarda modelo en app/models/model.pkl
    model_path = Path(__file__).parent / "model.pkl"
    joblib.dump(clf, model_path)
    print(f"Modelo guardado en {model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python -m app.models.train_gesture_model lsm_landmarks.csv")
        sys.exit(1)
    main(Path(sys.argv[1]))
