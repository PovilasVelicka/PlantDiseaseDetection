from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
from PIL import Image
import numpy as np
import json
import io
from pathlib import Path
import uvicorn
from contextlib import asynccontextmanager

# ================================
# ⚙️ Настройки
# ================================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "plant_disease_model.onnx"
CLASSES_PATH = BASE_DIR / "class_names.json"

# ================================
# 🔄 Предобработка изображения
# ================================
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    RU: Приводим изображение к [1, 3, 224, 224], нормализуем как при обучении
    EN: Convert image to [1, 3, 224, 224], normalize like in training
    """
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((224, 224))
    x = np.asarray(image).astype("float32") / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = np.expand_dims(x, axis=0)   # [C,H,W] -> [1,C,H,W]
    return x

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ================================
# 🌱 Lifespan: загрузка/освобождение ресурсов
# 🌱 Lifespan: load/release resources
# ================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # RU: Загрузка артефактов при старте
    # EN: Load artifacts on startup
    if not MODEL_PATH.exists():
        raise RuntimeError(f"❌ Model file not found: {MODEL_PATH}")
    if not CLASSES_PATH.exists():
        raise RuntimeError(f"❌ class_names.json not found: {CLASSES_PATH}")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        app.state.class_names = json.load(f)

    app.state.session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    print(f"✅ Model loaded with {len(app.state.class_names)} classes.")

    try:
        yield
    finally:
        # RU: Освобождение ресурсов при завершении (опционально)
        # EN: Cleanup on shutdown (optional)
        app.state.session = None

# ================================
# 🌐 Инициализация FastAPI с lifespan
# ================================
app = FastAPI(title="Plant Disease Detector API", lifespan=lifespan)

# CORS (на проде лучше указать домен фронта)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["https://your-frontend.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# 📌 Healthcheck
# ================================
@app.get("/health")
def health():
    return {"status": "ok", "classes": len(getattr(app.state, "class_names", []))}

# ================================
# 📌 Модель ответа
# ================================
class PredictResponse(BaseModel):
    klass: str
    confidence: float
    topk: list[dict]

# ================================
# 📌 Эндпоинт предсказания
# ================================
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    session = getattr(app.state, "session", None)
    class_names = getattr(app.state, "class_names", None)

    if session is None or not class_names:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content = await file.read()
    try:
        inp = preprocess_image(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: inp})[0]  # [1, num_classes]

    probs = softmax(output[0])
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])

    # Top-5
    topk_idx = np.argsort(probs)[::-1][:5]
    topk = [{"klass": class_names[i], "confidence": float(probs[i])} for i in topk_idx]

    return PredictResponse(klass=class_names[top_idx], confidence=top_conf, topk=topk)

# ================================
# 🔹 Локальный запуск
# ================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


