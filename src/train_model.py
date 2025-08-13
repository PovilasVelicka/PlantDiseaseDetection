# 🧠 Обучение модели MobileNetV2 (Transfer Learning)
# 🧠 Training MobileNetV2 model (Transfer Learning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from torchvision import datasets, transforms
import os
BASE_DIR = "../content/plant_disease_split"
# ================================
# 2️⃣ Аугментации
# 2️⃣ Augmentations
# ================================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),                # RU: приводим размер / EN: resize
    transforms.RandomHorizontalFlip(),            # RU: случайное отражение / EN: random flip
    transforms.RandomRotation(10),                # RU: случайный поворот / EN: rotation
    transforms.ToTensor(),                        # RU: в тензор / EN: to tensor
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])        # RU: нормализация / EN: normalize
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])
])

# ================================
# 3️⃣ DataLoader
# ================================
train_dataset = datasets.ImageFolder(os.path.join(BASE_DIR, "train"), transform=train_transforms)
val_dataset   = datasets.ImageFolder(os.path.join(BASE_DIR, "val"), transform=val_test_transforms)
test_dataset  = datasets.ImageFolder(os.path.join(BASE_DIR, "test"), transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"📊 Классы / Classes: {train_dataset.classes}")
print(f"📈 Размер train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")


# RU: Определяем устройство (GPU если доступно)
# EN: Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Используем устройство / Using device: {device}")

# RU: Загружаем предобученную MobileNetV2
# EN: Load pretrained MobileNetV2
model = models.mobilenet_v2(pretrained=True)

# RU: Замораживаем все параметры кроме классификатора
# EN: Freeze all layers except classifier
for param in model.features.parameters():
    param.requires_grad = False

# RU: Заменяем последний слой под наше число классов
# EN: Replace last layer for our number of classes
num_classes = len(train_dataset.classes)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

model = model.to(device)

# ================================
# Loss и оптимизатор / Loss & optimizer
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ================================
# Функция обучения
# Training function
# ================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        # ---- Training phase ----
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # ---- Validation phase ----
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"📅 Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    return model

# Запуск обучения
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

# 💾 Сохранение обученной модели
# 💾 Saving trained model

# RU: Сохраняем веса в формате PyTorch
# EN: Save weights in PyTorch format
torch.save(model.state_dict(), "plant_disease_model.pth")
print("✅ Модель сохранена в plant_disease_model.pth")

# RU: Экспорт в ONNX для деплоя в Azure или браузер
# EN: Export to ONNX for Azure or browser deployment
dummy_input = torch.randn(1, 3, 224, 224, device=device)
torch.onnx.export(
    model,
    dummy_input,
    "plant_disease_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("✅ Модель экспортирована в plant_disease_model.onnx")