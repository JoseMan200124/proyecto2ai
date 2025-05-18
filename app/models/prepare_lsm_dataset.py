"""
Genera un CSV con 63 features + label a partir del dataset LSM-249.
Requisitos:
    1. Descarga y descomprime el dataset en ./data/LSM/
       cada signo = carpeta, dentro subcarpetas por persona.
    2. Ejecuta:
        python -m app.models.prepare_lsm_dataset
"""

from pathlib import Path
import csv, cv2, numpy as np
from tqdm import tqdm
from app.utils.preprocessing import extract_hand_landmarks

DATA_DIR   = Path(__file__).resolve().parents[2] / "data" / "LSM"
OUTPUT_CSV = Path(__file__).resolve().parents[2] / "lsm_landmarks.csv"

def main():
    if not DATA_DIR.exists():
        raise RuntimeError(f"No se encontró {DATA_DIR}. Descarga primero el dataset.")

    features_written = 0
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"f{i}" for i in range(63)] + ["label"])

        # Recorre carpetas SIGN/PERSON/*.jpg
        for sign_dir in tqdm(sorted(DATA_DIR.iterdir()), desc="Signos"):
            if not sign_dir.is_dir(): continue
            label = sign_dir.name
            images = list(sign_dir.rglob("*.jpg")) + list(sign_dir.rglob("*.png"))
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None: continue
                vec = extract_hand_landmarks(img)
                if vec is not None:
                    writer.writerow(np.append(vec, label))
                    features_written += 1

    print(f"Vectores escritos: {features_written}")
    print(f"CSV generado en: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
