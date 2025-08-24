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
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"cannot open image: {e}")

    # RU: LANCZOS даёт качественный ресайз; при обучении используйте тот же метод
    # EN: LANCZOS gives high-quality resize; match training pipeline if possible
    image = image.resize((224, 224), Image.LANCZOS)

    x = np.asarray(image, dtype=np.float32) / 255.0
    # RU: Нормализация, как у вас было; при необходимости замените на ImageNet mean/std
    # EN: Keep your (x-0.5)/0.5 normalization; swap to ImageNet mean/std if you trained that way
    x = (x - 0.5) / 0.5

    # RU: HWC -> CHW -> NCHW
    # EN: HWC -> CHW -> NCHW
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return x  # (1, 3, 224, 224)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    RU: Численно стабильный softmax
    EN: Numerically stable softmax
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    s = np.sum(e, axis=axis, keepdims=True)
    # RU: Предотвращаем деление на ноль
    # EN: Avoid division by zero
    s = np.where(s == 0, 1.0, s)
    return e / s

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
        class_names = json.load(f)
        if not isinstance(class_names, list) or not all(isinstance(c, str) for c in class_names):
            raise RuntimeError("class_names.json must be a JSON array of strings")
        app.state.class_names = class_names

    # RU: Явно выбираем CPU, но можно проверить доступные провайдеры
    # EN: Use CPU provider explicitly; you may probe available providers if needed
    try:
        app.state.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")

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
    allow_credentials=False,  # RU: Нельзя True с allow_origins=["*"]
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
class DiseasePrediction(BaseModel):
    disease_class: str
    confidence: float

class PredictResponse(BaseModel):
    disease_class: str
    confidence: float
    top_predicts: list[DiseasePrediction]

# ================================
# 📌 Эндпоинт предсказания
# ================================
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    session = getattr(app.state, "session", None)
    class_names = getattr(app.state, "class_names", None)

    if session is None or not class_names:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # RU: Ограничения по типу можно добавить через file.content_type
    # EN: You can validate MIME type via file.content_type
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        inp = preprocess_image(content)  # (1, 3, 224, 224)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: inp})[0]  # (1, num_classes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    if output.ndim != 2 or output.shape[0] != 1:
        raise HTTPException(status_code=500, detail=f"Unexpected model output shape: {output.shape}")

    logits = output[0]  # (num_classes,)
    probs = softmax(logits, axis=-1).astype(float)  # ensure JSON‑serializable

    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])

    # Top‑5 (или меньше, если классов <5)
    k = min(5, len(class_names))
    top_predicts_idx = np.argsort(probs)[::-1][:k]
    top_predicts = [
        {"disease_class": class_names[i], "confidence": float(probs[i])}
        for i in top_predicts_idx
    ]

    return PredictResponse(
        disease_class=class_names[top_idx],
        confidence=top_conf,
        top_predicts=top_predicts,  # ✅ возвращаем правильную структуру
    )

# ================================
# 🔹 Локальный запуск
# ================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
