import os
import zipfile
import shutil
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

class PlantDiseaseTrainer:
    def __init__(self, kaggle_dataset="emmarex/plantdisease", batch_size=32, num_epochs=5, lr=0.001):
        self.kaggle_dataset = kaggle_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        self.dataset_dir = Path(__file__).parent / "dataset"
        self.split_dir = self.dataset_dir / "split"
        self.filtered_dir = self.dataset_dir / "PlantVillage"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []

    def _check_kaggle_json(self):
        kaggle_path_user = Path.home() / ".kaggle" / "kaggle.json"
        kaggle_path_local = Path(__file__).parent.parent / "kaggle.json"

        if kaggle_path_user.exists():
            print(f"✅ Найден kaggle.json в {kaggle_path_user}")
            return kaggle_path_user
        elif kaggle_path_local.exists():
            print(f"✅ Найден kaggle.json в {kaggle_path_local}, копируем в ~/.kaggle")
            os.makedirs(Path.home() / ".kaggle", exist_ok=True)
            shutil.copy(kaggle_path_local, kaggle_path_user)
            os.chmod(kaggle_path_user, 0o600)
            return kaggle_path_user
        else:
            raise FileNotFoundError("❌ kaggle.json не найден ни в ~/.kaggle, ни в корне проекта.")

    def download_dataset(self):
        if self.filtered_dir.exists():
            print("📦 Датасет уже скачан и распакован, пропускаем загрузку.")
            return

        self._check_kaggle_json()

        import subprocess
        subprocess.run([
            "kaggle", "datasets", "download", "-d", self.kaggle_dataset,
            "-p", str(self.dataset_dir)
        ], check=True)

        zip_path = self.dataset_dir / f"{self.kaggle_dataset.split('/')[-1]}.zip"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_dir)

        print("✅ Датасет загружен и распакован.")

    def prepare_data(self):
        """Split dataset into train/val/test with proper folder structure"""
        if self.split_dir.exists():
            shutil.rmtree(self.split_dir)

        for cls in os.listdir(self.filtered_dir):
            cls_path = self.filtered_dir / cls
            images = [f for f in cls_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

            if len(images) < 2:
                print(f"⚠️ Класс {cls} пропущен — мало изображений")
                continue

            train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)

            for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
                dst_dir = self.split_dir / split_name / cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                for img in split_imgs:
                    shutil.copy(img, dst_dir)

        print("✅ Данные разделены на train/val/test")

    def create_dataloaders(self):
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        val_test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        train_dataset = datasets.ImageFolder(self.split_dir / "train", transform=train_transforms)
        val_dataset = datasets.ImageFolder(self.split_dir / "val", transform=val_test_transforms)
        test_dataset = datasets.ImageFolder(self.split_dir / "test", transform=val_test_transforms)

        self.class_names = train_dataset.classes

        return (
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        )

    def build_model(self):
        self.model = models.mobilenet_v2(pretrained=True)
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier[1] = nn.Linear(self.model.last_channel, len(self.class_names))
        self.model = self.model.to(self.device)

    def train(self, train_loader, val_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss, val_corrects = 0.0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)

            val_loss /= len(val_loader.dataset)
            val_acc = val_corrects.double() / len(val_loader.dataset)

            print(f"📅 Epoch {epoch+1}/{self.num_epochs} | "
                  f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    def save_model(self):
        api_dir = Path(__file__).parent.parent / "plant_api"
        api_dir.mkdir(exist_ok=True)

        onnx_path = api_dir / "plant_disease_model.onnx"
        json_path = api_dir / "class_names.json"

        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        torch.onnx.export(
            self.model, dummy_input, onnx_path,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.class_names, f, ensure_ascii=False, indent=2)

        print(f"✅ Модель сохранена в {onnx_path}")
        print(f"✅ Классы сохранены в {json_path}")


if __name__ == "__main__":
    trainer = PlantDiseaseTrainer(num_epochs=5, batch_size=32)
    trainer.download_dataset()
    trainer.prepare_data()
    train_loader, val_loader, test_loader = trainer.create_dataloaders()
    trainer.build_model()
    trainer.train(train_loader, val_loader)
    trainer.save_model()
