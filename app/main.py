from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2, numpy as np, joblib
from pathlib import Path
from app.utils.preprocessing import extract_hand_landmarks

app = FastAPI(title="Reconocimiento de Gestos – LSM 249")

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
if not MODEL_PATH.exists():
    raise RuntimeError("No se encontró model.pkl. Ejecuta prepare_lsm_dataset y train_gesture_model.")
model = joblib.load(MODEL_PATH)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "Solo imágenes JPG/PNG")
    img_bytes = await file.read()
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    feat = extract_hand_landmarks(frame)
    if feat is None:
        return JSONResponse({"success": False, "message": "No se detectó mano"})
    pred = model.predict([feat])[0]
    return JSONResponse({"success": True, "prediction": pred})
