# 📂 Подготовка данных для обучения
# 📂 Preparing data for training

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import os
import shutil
from pathlib import Path

# ================================
# 1️⃣ Разделение на train/val/test
# 1️⃣ Splitting into train/val/test
# ================================

# RU: Путь к отфильтрованным данным (Tomato + Apple)
# EN: Path to filtered data (Tomato + Apple)
FILTERED_DIR = "../content/plant_disease/Filtered"
BASE_DIR = "../content/plant_disease_split"

# RU: Если папка уже есть — удалим, чтобы не было конфликта
# EN: Remove existing folder to avoid conflicts
if os.path.exists(BASE_DIR):
    shutil.rmtree(BASE_DIR)

# RU: Создаём папки train, val, test
# EN: Create train, val, test folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(BASE_DIR, split), exist_ok=True)

# RU: Для каждого класса делаем разделение
# EN: Split each class separately
for cls in os.listdir(FILTERED_DIR):
    cls_path = Path(FILTERED_DIR) / cls
    # RU: Собираем все файлы и фильтруем по расширению в нижнем регистре
    # EN: Collect all files and filter by lowercase extension
    images = [f for f in cls_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    # RU: Разделение 70% train, 20% val, 10% test
    # EN: Split 70% train, 20% val, 10% test
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)

    os.makedirs(f"{BASE_DIR}/train/{cls}", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/val/{cls}", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/test/{cls}", exist_ok=True)

    for img in train_imgs:
        shutil.copy(img, f"{BASE_DIR}/train/{cls}")

    for img in val_imgs:
        shutil.copy(img, f"{BASE_DIR}/val/{cls}")

    for img in test_imgs:
        shutil.copy(img, f"{BASE_DIR}/test/{cls}")

print("✅ Данные разделены на train / val / test")
