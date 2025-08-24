import io
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel


# ================================
# Settings / Paths
# ================================
BASE_DIR: Path = Path(__file__).parent
MODEL_PATH: Path = BASE_DIR / "plant_disease_model.onnx"
CLASSES_PATH: Path = BASE_DIR / "class_names.json"
LOG_DIR: Path = BASE_DIR / "logs"


# ================================
# Logging
# ================================
def setup_logging(log_dir: Path = LOG_DIR) -> logging.Logger:
    """
    Configure a module-local logger that logs to console and a dated file.
    File pattern: logs/api-YYYY-MM-DD.log
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"api-{datetime.now().strftime('%Y-%m-%d')}.log"

    logger = logging.getLogger("plant_api")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False

    logger.info(f"Logging to {logfile}")
    return logger


LOGGER = setup_logging()


# ================================
# Image preprocessing
# ================================
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Convert input image to NCHW float32 tensor [1, 3, 224, 224].
    Uses ImageNet mean/std normalization (must match training).
    """
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"cannot open image: {e}")

    image = image.resize((224, 224), Image.Resampling.LANCZOS)

    x = np.asarray(image, dtype=np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std

    # HWC -> CHW -> NCHW
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return x  # shape: (1, 3, 224, 224)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    s = np.sum(e, axis=axis, keepdims=True)
    s = np.where(s == 0, 1.0, s)  # avoid division by zero
    return e / s


# ================================
# Lifespan: load/release resources
# ================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load artifacts on startup
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not CLASSES_PATH.exists():
        raise RuntimeError(f"class_names.json not found: {CLASSES_PATH}")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    if not isinstance(class_names, list) or not all(isinstance(c, str) for c in class_names):
        raise RuntimeError("class_names.json must be a JSON array of strings")

    app.state.class_names = class_names

    try:
        app.state.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]  # adjust providers if GPU/DirectML is available
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")

    LOGGER.info(f"Model loaded. Classes: {len(app.state.class_names)}")

    try:
        yield
    finally:
        # Cleanup on shutdown (optional)
        app.state.session = None
        LOGGER.info("Resources released.")


# ================================
# FastAPI app
# ================================
app = FastAPI(title="Plant Disease Detector API", lifespan=lifespan)

# CORS (in production, restrict to your frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # e.g., ["https://your-frontend.app"]
    allow_credentials=False,        # must be False if allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# Healthcheck
# ================================
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classes": len(getattr(app.state, "class_names", []))}


# ================================
# Response models
# ================================
class DiseasePrediction(BaseModel):
    disease_class: str
    confidence: float


class PredictResponse(BaseModel):
    disease_class: str
    confidence: float
    top_predicts: List[DiseasePrediction]


# ================================
# Prediction endpoint
# ================================
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    session: ort.InferenceSession | None = getattr(app.state, "session", None)
    class_names: List[str] | None = getattr(app.state, "class_names", None)

    if session is None or not class_names:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content: bytes = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        inp = preprocess_image(content)  # (1, 3, 224, 224)
    except Exception as e:
        LOGGER.warning(f"Preprocess failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: inp})[0]  # (1, num_classes)
    except Exception as e:
        LOGGER.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    if output.ndim != 2 or output.shape[0] != 1:
        msg = f"Unexpected model output shape: {output.shape}"
        LOGGER.error(msg)
        raise HTTPException(status_code=500, detail=msg)

    logits = output[0]  # shape: (num_classes,)
    probs = softmax(logits, axis=-1).astype(float)

    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])

    k = min(5, len(class_names))
    top_predicts_idx = np.argsort(probs)[::-1][:k]
    top_predicts = [
        DiseasePrediction(disease_class=class_names[i], confidence=float(probs[i]))
        for i in top_predicts_idx
    ]

    LOGGER.info(f"Predicted: {class_names[top_idx]} ({top_conf:.4f})")

    return PredictResponse(
        disease_class=class_names[top_idx],
        confidence=top_conf,
        top_predicts=top_predicts,
    )


# ================================
# Local runner
# ================================
if __name__ == "__main__":
    # For production, prefer CLI: `uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2`
    uvicorn.run(app, host="0.0.0.0", port=8000)
