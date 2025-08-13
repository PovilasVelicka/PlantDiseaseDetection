# 📦 Подключение к Kaggle и загрузка датасета PlantVillage
# 📦 Connect to Kaggle and download PlantVillage dataset

# RU: Устанавливаем Kaggle API и создаём папки
# EN: Install Kaggle API and create folders

import os

# RU: Папка для датасета
# EN: Dataset folder
DATA_DIR = "../content/plant_disease"
os.makedirs(DATA_DIR, exist_ok=True)

# ⚠️ ВАЖНО: Перед запуском нужно загрузить kaggle.json в Colab
# RU: Файл kaggle.json можно получить в профиле Kaggle -> Account -> API -> Create New Token
# EN: Get kaggle.json from your Kaggle account (Profile -> Account -> API -> Create New Token)

import os
import subprocess

# DATA_DIR = "plant_disease"
# os.makedirs(DATA_DIR, exist_ok=True)

# 1️⃣ Создаём папку ~/.kaggle
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

# 2️⃣ Копируем kaggle.json (замени путь на свой)
source_kaggle_json = r"kaggle.json"  # путь к kaggle.json на твоём ПК
dest_kaggle_json = os.path.expanduser("~\\.kaggle\\kaggle.json")
os.replace(source_kaggle_json, dest_kaggle_json)

# 3️⃣ Даем права на файл (Linux/MacOS; на Windows можно пропустить)
try:
    os.chmod(dest_kaggle_json, 0o600)
except PermissionError:
    print("⚠️ chmod пропущен — Windows не требует этой команды")

# 4️⃣ Скачиваем датасет через Kaggle CLI
subprocess.run(
    ["kaggle", "datasets", "download", "-d", "emmarex/plantdisease", "-p", DATA_DIR],
    check=True
)

# 5️⃣ Распаковываем архив
import zipfile
zip_path = os.path.join(DATA_DIR, "plantdisease.zip")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

print("✅ Датасет скачан и распакован")


# ------------------------
# 📂 Фильтрация на Tomato и Apple
# ------------------------
import shutil
from pathlib import Path

# RU: Папка с изображениями
# EN: Folder with images
RAW_DIR = Path(DATA_DIR) / "PlantVillage"

# RU: Новая папка для отфильтрованных данных
# EN: New folder for filtered dataset
FILTERED_DIR = Path(DATA_DIR) / "Filtered"
os.makedirs(FILTERED_DIR, exist_ok=True)

# RU: Классы, которые оставляем
# EN: Classes we want to keep
KEEP_CLASSES = [c for c in os.listdir(RAW_DIR) if c.startswith("Tomato")]

# RU: Копируем только нужные папки
# EN: Copy only the required classes
for cls in KEEP_CLASSES:
    src = RAW_DIR / cls
    dst = FILTERED_DIR / cls
    shutil.copytree(src, dst, dirs_exist_ok=True)

print(f"✅ Оставлено классов / Classes kept: {KEEP_CLASSES}")