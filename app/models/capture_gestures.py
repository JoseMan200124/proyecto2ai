"""
Script para capturar tus propias muestras y etiquetarlas.
Uso:
    python -m app.models.capture_gestures  salida.csv
Pulsa números/teclas para etiquetar cada gesto y ESC para salir.
"""

import sys
import csv
import cv2
import numpy as np
from pathlib import Path
from app.utils.preprocessing import extract_hand_landmarks

LABEL_KEYS = {
    ord('1'): 'palm_open',
    ord('2'): 'fist',
    ord('3'): 'thumbs_up',
    ord('4'): 'thumbs_down',
    ord('5'): 'victory',
}

def main(output_csv: Path):
    cap = cv2.VideoCapture(0)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        header = [f"f{i}" for i in range(63)] + ["label"]
        writer.writerow(header)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = extract_hand_landmarks(frame)
            cv2.imshow("Capture – press 1-5 to label, ESC to exit", frame)

            key = cv2.waitKey(1) & 0xff
            if key == 27:  # ESC
                break
            if key in LABEL_KEYS and landmarks is not None:
                row = np.append(landmarks, LABEL_KEYS[key])
                writer.writerow(row)
                print(f"Saved {LABEL_KEYS[key]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python -m app.models.capture_gestures salida.csv")
        sys.exit(1)
    main(Path(sys.argv[1]))
