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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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
        self.optimizer = None          # <— добавим
        self.criterion = None          # <— добавим
        self.current_epoch = 0         # <— добавим (для продолжения)
        self.class_names = []


    def _create_optimizer(self, lr=None):
        """Создаём оптимизатор по trainable-параметрам."""
        if lr is not None:
            self.lr = lr
        # важный момент: только те параметры, у которых requires_grad=True
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

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
        # train_transforms = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(10),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                          std=[0.5, 0.5, 0.5])
        # ])
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([
            # Геометрия и масштаб / Geometry & scale
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.5, 1.0),  # больше «зумов»/кропов из разных масштабов
                ratio=(0.7, 1.4)  # разное соотношение сторон / aspect ratios
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180, fill=tuple(int(255 * m) for m in IMAGENET_MEAN)),

            # Перспектива и небольшие деформации / Perspective & warps
            transforms.RandomPerspective(distortion_scale=0.3, p=0.2),

            # Фотометрика / Photometric jitter
            transforms.ColorJitter(  # яркость/контраст/насыщенность/гамма
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02
            ),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),

            # В тензор и нормализация / Tensor + normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

            # Стирание фрагмента (закрытые/шумные блоки) / RandomErasing
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        ])

        # val_test_transforms = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                          std=[0.5, 0.5, 0.5])
        # ])

        # Для val/test оставляем «чистую» предобработку (без случайностей)
        val_test_transforms = transforms.Compose([
            transforms.Resize(256),  # стандарт для инференса / inference standard
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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

    # def train(self, train_loader, val_loader):
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    #
    #     for epoch in range(self.num_epochs):
    #         self.model.train()
    #         running_loss, running_corrects = 0.0, 0
    #
    #         for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]"):
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             optimizer.zero_grad()
    #             outputs = self.model(inputs)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()
    #
    #             _, preds = torch.max(outputs, 1)
    #             running_loss += loss.item() * inputs.size(0)
    #             running_corrects += torch.sum(preds == labels.data)
    #
    #         epoch_loss = running_loss / len(train_loader.dataset)
    #         epoch_acc = running_corrects.double() / len(train_loader.dataset)
    #
    #         # Validation
    #         self.model.eval()
    #         val_loss, val_corrects = 0.0, 0
    #         with torch.no_grad():
    #             for inputs, labels in val_loader:
    #                 inputs, labels = inputs.to(self.device), labels.to(self.device)
    #                 outputs = self.model(inputs)
    #                 loss = criterion(outputs, labels)
    #                 _, preds = torch.max(outputs, 1)
    #                 val_loss += loss.item() * inputs.size(0)
    #                 val_corrects += torch.sum(preds == labels.data)
    #
    #         val_loss /= len(val_loader.dataset)
    #         val_acc = val_corrects.double() / len(val_loader.dataset)
    #
    #         print(f"📅 Epoch {epoch+1}/{self.num_epochs} | "
    #               f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
    #               f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    def train(self, train_loader, val_loader, epochs=None, start_epoch=None):
        """Обучение с возможностью продолжения с нужного номера эпохи."""
        if epochs is None:
            epochs = self.num_epochs
        if start_epoch is None:
            start_epoch = self.current_epoch

        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        if self.optimizer is None:
            self._create_optimizer()  # создадим, если ещё не создан

        total_epochs = start_epoch + epochs
        for epoch in range(start_epoch, total_epochs):
            self.model.train()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Train]"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

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
                    loss = self.criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)

            val_loss /= len(val_loader.dataset)
            val_acc = val_corrects.double() / len(val_loader.dataset)

            print(f"📅 Epoch {epoch + 1}/{total_epochs} | "
                  f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # запомним, до какой эпохи дошли
            self.current_epoch = epoch + 1
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

    # --- Новое: сохранение/загрузка чекпоинта PyTorch для продолжения ---
    def save_checkpoint(self, path="plant_checkpoint.pt"):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "epoch": self.current_epoch,
            "class_names": self.class_names,
            "lr": self.lr,
        }, path)
        print(f"💾 Checkpoint сохранён: {path}")

    def load_checkpoint(self, path="plant_checkpoint.pt"):
        ckpt = torch.load(path, map_location=self.device)
        # модель должна быть уже построена под нужное число классов
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(ckpt["model_state"])
        self.class_names = ckpt.get("class_names", self.class_names)
        self.current_epoch = ckpt.get("epoch", 0)

        # пересоздаём оптимизатор (на случай изменения requires_grad)
        self._create_optimizer(lr=ckpt.get("lr", self.lr))
        if ckpt.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])

        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        print(f"✅ Checkpoint загружен. Текущая эпоха: {self.current_epoch}")

    # --- Новое: дообучение (ещё N эпох), с опциональным размораживанием и новым LR ---
    def fine_tune(self, train_loader, val_loader, extra_epochs=2, lr=None, unfreeze=False):
        """
        extra_epochs: сколько эпох добавить
        lr: можно задать новый learning rate для дообучения
        unfreeze: если True — разморозим backbone (features) перед дообучением
        """
        if unfreeze:
            for p in self.model.features.parameters():
                p.requires_grad = True

        # при смене набора trainable-параметров — пересоздаём оптимизатор
        self._create_optimizer(lr=lr if lr is not None else self.lr)

        # продолжаем обучение с текущей эпохи
        self.train(train_loader, val_loader, epochs=extra_epochs, start_epoch=self.current_epoch)
    def evaluate_test(self, test_loader):
        """Оценка модели на test с графиками и отчётом"""
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                logits = self.model(x)
                preds = logits.argmax(1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(y.numpy())

        y_true, y_pred = np.array(y_true), np.array(y_pred)

        # Общая точность
        acc = accuracy_score(y_true, y_pred)
        print(f"✅ Test accuracy: {acc:.4f}\n")

        # Classification report
        print("📊 Classification report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion(cm, self.class_names, normalize=True)

        # Per-class accuracy
        self._plot_per_class_accuracy(y_true, y_pred, self.class_names)

    # --- вспомогательные функции для графиков ---
    def _plot_confusion(self, cm, class_names, normalize=True, title="Confusion Matrix"):
        if normalize:
            cm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.show()

    def _plot_per_class_accuracy(self, y_true, y_pred, class_names):
        n = len(class_names)
        per_cls = []
        for i in range(n):
            idx = (y_true == i)
            acc_i = (y_pred[idx] == i).mean() if idx.any() else 0.0
            per_cls.append(acc_i)
        fig = plt.figure(figsize=(10, 4))
        plt.bar(range(n), per_cls)
        plt.xticks(range(n), class_names, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Per-class accuracy")
        plt.tight_layout()
        plt.show()
        return per_cls

# if __name__ == "__main__":
#     trainer = PlantDiseaseTrainer(num_epochs=5, batch_size=32)
#     trainer.download_dataset()
#     trainer.prepare_data()
#     train_loader, val_loader, test_loader = trainer.create_dataloaders()
#     trainer.build_model()
#     trainer.train(train_loader, val_loader)
#     trainer.save_model()
if __name__ == "__main__":
    # trainer = PlantDiseaseTrainer(num_epochs=50, batch_size=32)
    # trainer.download_dataset()
    # trainer.prepare_data()
    # train_loader, val_loader, test_loader = trainer.create_dataloaders()
    # trainer.build_model()
    #
    # # Первый прогон (5 эпох)
    # trainer.train(train_loader, val_loader)
    # trainer.save_checkpoint("plant_ckpt.pt")   # можно сохранить точку

    # # # ... Спустя время решили «дотренировать» ещё 2 эпохи:
    # # # 1) Вариант без перезапуска процесса:
    # # trainer.fine_tune(train_loader, val_loader, extra_epochs=2, lr=5e-4, unfreeze=True)
    # # trainer.save_checkpoint("plant_ckpt.pt")   # можно сохранить точку
    # 2) Вариант из нового процесса:
    # trainer = PlantDiseaseTrainer()
    # train_loader, val_loader, test_loader = trainer.create_dataloaders()
    # trainer.build_model()
    # trainer.load_checkpoint("plant_ckpt.pt")
    # trainer.fine_tune(train_loader, val_loader, extra_epochs=5, lr=1e-4, unfreeze=True)
    # trainer.save_checkpoint("plant_ckpt.pt")  # можно сохранить точку

    # # 3)
    # trainer = PlantDiseaseTrainer()
    # train_loader, val_loader, test_loader = trainer.create_dataloaders()
    #
    # trainer.build_model()
    # trainer.load_checkpoint("plant_ckpt.pt")  # подтянет current_epoch и веса
    #
    # # Разморозить backbone и дообучать малым LR
    # for p in trainer.model.features.parameters():
    #     p.requires_grad = True
    #
    # # Новый оптимизатор под размороженные параметры
    # trainer.optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, trainer.model.parameters()),
    #     lr=1e-4, weight_decay=1e-4
    # )
    #
    # trainer.fine_tune(train_loader, val_loader, extra_epochs=15, lr=1e-4, unfreeze=False)
    #
    # trainer.save_checkpoint("plant_ckpt.pt")
    # trainer.save_model()
# >>>>>>>>>>>>>>>>>TEST MODEL <<<<<<<<<<<<<<<<<<<
    trainer = PlantDiseaseTrainer()

    # создаём DataLoader'ы
    train_loader, val_loader, test_loader = trainer.create_dataloaders()

    # строим модель
    trainer.build_model()

    # загружаем чекпоинт с весами (PyTorch, НЕ onnx!)
    ckpt_path = Path("plant_ckpt.pt")
    if ckpt_path.exists():
        trainer.load_checkpoint(str(ckpt_path))
    else:
        print("⚠️ Чекпоинт не найден, модель будет пустая.")

    # теперь запускаем тестовую оценку с графиками
    trainer.evaluate_test(test_loader)