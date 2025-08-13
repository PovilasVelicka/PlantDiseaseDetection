import onnxruntime as ort
from PIL import Image
import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# ================================
# ⚙️ Настройки
# ================================
# RU: Путь к модели и списку классов
# EN: Paths to model and class list
BASE_DIR = Path(__file__).parent.parent  # корень проекта
API_DIR = BASE_DIR / "plant_api"
MODEL_PATH = API_DIR / "plant_disease_model.onnx"
CLASSES_PATH = API_DIR / "class_names.json"

# RU: Проверим, что все файлы есть
# EN: Check all files exist
if not MODEL_PATH.exists():
    print(f"❌ Модель не найдена: {MODEL_PATH}")
    sys.exit(1)
if not CLASSES_PATH.exists():
    print(f"❌ class_names.json не найден: {CLASSES_PATH}")
    sys.exit(1)

# RU: Загружаем список классов
# EN: Load class names
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# RU: Загружаем модель ONNX
# EN: Load ONNX model
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

# ================================
# 🔄 Предобработка изображения
# ================================
def preprocess_image(image_path: Path) -> np.ndarray:
    """
    RU: Приводим изображение к [1, 3, 224, 224] и нормализуем как при обучении
    EN: Convert image to [1, 3, 224, 224] and normalize like during training
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    x = np.asarray(image).astype("float32") / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = np.expand_dims(x, axis=0)   # [C,H,W] -> [1,C,H,W]
    return x, image

# ================================
# 📊 Предсказание
# ================================
def predict(image_path: Path):
    input_tensor, image = preprocess_image(image_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]

    probs = softmax(output[0])
    top_idx = int(np.argmax(probs))
    top_class = class_names[top_idx]
    confidence = float(probs[top_idx])

    print(f"📌 Файл: {image_path}")
    print(f"✅ Предсказанный класс: {top_class}")
    print(f"🎯 Уверенность: {confidence:.2f}")

    # RU: Топ-5 классов
    # EN: Top-5 classes
    top5_idx = np.argsort(probs)[::-1][:5]
    print("\n🏆 Top-5:")
    for i in top5_idx:
        print(f" - {class_names[i]}: {probs[i]:.2f}")

    # RU: Показ изображения
    # EN: Show image
    plt.imshow(image)
    plt.title(f"Pred: {top_class} ({confidence:.2f})")
    plt.axis("off")
    plt.show()

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ================================
# 🚀 Запуск
# ================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Укажите путь к изображению, например:\n   python test_model.py /path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"❌ Файл не найден: {image_path}")
        sys.exit(1)

    predict(image_path)
