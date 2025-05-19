"""
Entrena un RandomForest con el CSV generado por prepare_lsm_dataset.py.

Uso:
    python -m app.models.train_gesture_model lsm_landmarks.csv
"""

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ----------------------------------------------------------------------
def main(csv_path: Path):
    # ---------- carga datos ----------
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # ---------- entrena modelo ----------
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_tr, y_tr)

    # ---------- métricas ----------
    y_pred = clf.predict(X_te)
    print(classification_report(y_te, y_pred))

    cm = confusion_matrix(y_te, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)

    # crea figura con tamaño deseado
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusión")
    plt.tight_layout()
    plt.savefig("metrics.png")
    print("Matriz de confusión guardada en metrics.png")

    # ---------- guarda modelo ----------
    model_path = Path(__file__).parent / "model.pkl"
    joblib.dump(clf, model_path)
    print(f"Modelo guardado en {model_path}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Uso: python -m app.models.train_gesture_model lsm_landmarks.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    csv_file = Path(sys.argv[1])
    if not csv_file.exists():
        print(f"❌ No se encontró {csv_file}", file=sys.stderr)
        sys.exit(1)

    main(csv_file)
