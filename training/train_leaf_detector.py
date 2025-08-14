"""
RU: Обучение бинарного классификатора leaf / not_leaf
EN: Training binary classifier leaf / not_leaf
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json
from pathlib import Path

# ================================
# ⚙️ Настройки
# ================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "leaf_detector_dataset"
API_DIR = BASE_DIR / "plant_api"
MODEL_PATH = API_DIR / "leaf_detector.onnx"
CLASSES_PATH = API_DIR / "leaf_classes.json"

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
IMG_SIZE = 224

# ================================
# 📦 Датасет и трансформации
# ================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class_names = dataset.classes
print(f"📊 Классы: {class_names}")

# ================================
# 🧠 Модель (MobileNetV2)
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ================================
# 🚀 Обучение
# ================================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    acc = correct / total
    print(f"📅 Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.4f}")

# ================================
# 💾 Сохранение в ONNX
# ================================
API_DIR.mkdir(parents=True, exist_ok=True)

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
torch.onnx.export(
    model, dummy_input, MODEL_PATH,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

with open(CLASSES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print(f"✅ Модель сохранена в {MODEL_PATH}")
print(f"✅ Классы сохранены в {CLASSES_PATH}")
