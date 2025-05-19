# app/main.py
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import joblib
from pathlib import Path

from app.utils.preprocessing import extract_hand_landmarks

app = FastAPI(title="Reconocimiento de Gestos – LSM-249")

# ----------------------------------------------------------------------
# CORS (por si sirves el frontend desde file:// o puerto distinto)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

model = None  # se cargará en startup


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "No se encontró model.pkl. Ejecuta prepare_lsm_dataset y train_gesture_model."
        )
    # mmap_mode="r" evita cargar todo el RandomForest en RAM de golpe
    model = joblib.load(MODEL_PATH, mmap_mode="r")
    print("✅ Modelo cargado:", MODEL_PATH)


# ----------------------------------------------------------------------
# Static & templates
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ----------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ─── Validar tipo de archivo ───────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Solo imágenes JPG/PNG")

    # ─── Leer bytes → ndarray BGR ──────────────────────────────────────
    img_bytes = await file.read()
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "No se pudo decodificar la imagen")

    # ─── Extraer features (landmarks) ──────────────────────────────────
    feat = extract_hand_landmarks(frame)
    if feat is None:
        return {"success": False, "message": "No se detectó mano"}

    # ─── Predicción ────────────────────────────────────────────────────
    pred = model.predict([feat])[0]

    # Asegurar que sea JSON-serializable (convierte int64 → int)
    if isinstance(pred, (np.generic, np.number)):
        pred = pred.item()

    return {"success": True, "prediction": pred}
