import os
import zipfile
import shutil
import json
import logging
from typing import Optional, Tuple, List, Self

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
import random
import subprocess
import yaml


# =======================
# Logging configuration
# =======================
def setup_logging(log_dir: Path = Path("logs")) -> logging.Logger:
    """
    Configure root logger to write both to console and to a dated file.
    File name format: logs/YYYY-MM-DD.log
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

    logger = logging.getLogger("plant_trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(ch_fmt)

    # File handler
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fh_fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    logger.info(f"Logging to {logfile}")
    return logger


# =======================
# Utils: reproducibility
# =======================
def set_seed(seed: int = 42):
    """
    Fix all relevant seeds and flags for best-effort reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PlantDiseaseTrainer:
    def __init__(self,
                 kaggle_dataset: str = "emmarex/plantdisease",
                 batch_size: int = 32,
                 num_epochs: int = 5,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 label_smoothing: float = 0.05,
                 use_weighted_sampler: bool = True,
                 amp: bool = True,
                 log_dir: str | Path = "logs") -> Self:
        # Config
        self.kaggle_dataset = kaggle_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.use_weighted_sampler = use_weighted_sampler
        self.amp = amp

        # Paths
        here = Path(__file__).parent
        self.dataset_dir = here / "dataset"
        self.split_dir = self.dataset_dir / "split"
        self.filtered_dir = self.dataset_dir / "PlantVillage"
        self.api_dir = here.parent / "plant_api"
        self.ckpt_dir = here / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.logger = setup_logging(Path(log_dir))

        # State
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None
        self.criterion: nn.Module | None = None
        self.current_epoch: int = 0
        self.class_names: list[str] = []
        self.scaler = GradScaler(enabled=self.amp)  # mixed precision scaler

        self.best_val_acc: float = 0.0


    # ----------
    # Config IO
    # ----------
    @staticmethod
    def from_yaml(path: str | Path) -> "PlantDiseaseTrainer":
        """
        Construct trainer from YAML config file.
        """
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        trainer = PlantDiseaseTrainer(
            kaggle_dataset=cfg.get("kaggle_dataset", "emmarex/plantdisease"),
            batch_size=cfg.get("batch_size", 32),
            num_epochs=cfg.get("num_epochs", 5),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
            label_smoothing=cfg.get("label_smoothing", 0.05),
            use_weighted_sampler=cfg.get("use_weighted_sampler", True),
            amp=cfg.get("amp", True),
            log_dir=cfg.get("log_dir", "logs"),
        )
        return trainer


    # -------------
    # Kaggle fetch
    # -------------
    def _check_kaggle_json(self) -> Path:
        """
        Ensure kaggle.json exists either in ~/.kaggle or project root; copy if needed.
        """
        kaggle_path_user = Path.home() / ".kaggle" / "kaggle.json"
        kaggle_path_local = Path(__file__).parent.parent / "kaggle.json"

        if kaggle_path_user.exists():
            self.logger.info(f"Found kaggle.json at {kaggle_path_user}")
            return kaggle_path_user
        elif kaggle_path_local.exists():
            self.logger.info(f"Found kaggle.json at {kaggle_path_local}, copying to ~/.kaggle")
            (Path.home() / ".kaggle").mkdir(exist_ok=True)
            shutil.copy(kaggle_path_local, kaggle_path_user)
            os.chmod(kaggle_path_user, 0o600)
            return kaggle_path_user
        else:
            raise FileNotFoundError("kaggle.json not found in ~/.kaggle or project root.")


    def download_dataset(self, drop_if_exists: bool = False):
        """
        Download the dataset from Kaggle if not present (optionally re-download).
        """
        if self.filtered_dir.exists() and not drop_if_exists:
            self.logger.info("Dataset already downloaded and extracted — skipping download.")
            return

        if self.filtered_dir.exists():
            shutil.rmtree(self.filtered_dir)

        self._check_kaggle_json()
        self.dataset_dir.mkdir(exist_ok=True)

        self.logger.info(f"Downloading Kaggle dataset: {self.kaggle_dataset}")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", self.kaggle_dataset, "-p", str(self.dataset_dir)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Kaggle CLI download failed: {e}")
            raise

        zip_path = self.dataset_dir / f"{self.kaggle_dataset.split('/')[-1]}.zip"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_dir)

        self.logger.info("Dataset downloaded and extracted.")


    # -------------
    # Data split
    # -------------
    def prepare_data(self, drop_if_exists: bool = False):
        """
        Split images into train/val/test folder structure under dataset/split.
        """
        if self.split_dir.exists() and not drop_if_exists:
            self.logger.info("Split folders already exist — skipping split.")
            return

        if self.split_dir.exists():
            shutil.rmtree(self.split_dir)

        classes = [d for d in os.listdir(self.filtered_dir) if (self.filtered_dir / d).is_dir()]
        if not classes:
            self.logger.warning(f"No class folders found under {self.filtered_dir}.")

        for cls in classes:
            cls_path = self.filtered_dir / cls
            images = [f for f in cls_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

            if len(images) < 2:
                self.logger.warning(f"Skipping class '{cls}' — not enough images.")
                continue

            # 70% train, 20% val, 10% test (approx.)
            train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)

            for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
                dst_dir = self.split_dir / split_name / cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                for img in split_imgs:
                    shutil.copy(img, dst_dir)

        self.logger.info("Data split into train/val/test.")


    # -----------------
    # Dataloaders + AU
    # -----------------
    def create_dataloaders(self):
        """
        Build DataLoaders and transforms for train/val/test.
        """
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        # Soft but diverse augmentations to improve generalization.
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30, fill=tuple(int(255 * m) for m in imagenet_mean)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.5, 2.0)),
        ])

        val_test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

        train_dataset = datasets.ImageFolder(self.split_dir / "train", transform=train_transforms)
        val_dataset = datasets.ImageFolder(self.split_dir / "val", transform=val_test_transforms)
        test_dataset = datasets.ImageFolder(self.split_dir / "test", transform=val_test_transforms)

        self.class_names = train_dataset.classes

        if self.use_weighted_sampler:
            label_indices = [lbl for _, lbl in train_dataset.samples]
            cnt = Counter(label_indices)
            total = sum(cnt.values())
            class_weights = torch.tensor([total / cnt[i] for i in range(len(self.class_names))], dtype=torch.float)
            sample_weights = [class_weights[y].item() for y in label_indices]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=os.cpu_count() or 0,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=os.cpu_count() or 0,
                pin_memory=torch.cuda.is_available(),
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() or 0,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() or 0,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, val_loader, test_loader


    # -------------
    # Model & Opt
    # -------------
    def build_model(self):
        """
        Load MobileNetV2, freeze the feature extractor, and replace the classifier head.
        """
        if not self.class_names:
            raise RuntimeError("class_names is empty. Call create_dataloaders() before build_model().")

        # New weights API (torchvision >= 0.13)
        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        for p in self.model.features.parameters():
            p.requires_grad = False

        self.model.classifier[1] = nn.Linear(self.model.last_channel, len(self.class_names))
        self.model = self.model.to(self.device)

    def _create_optimizer(self, lr: float | None = None):
        """
        (Re)create optimizer and scheduler for current trainable parameters.
        """
        if lr is not None:
            self.lr = lr
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)


    # -------------
    # Train / Eval
    # -------------
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: Optional[int] = None,
              start_epoch: Optional[int] = None) -> Self:
        """
        Standard supervised training loop with validation and optional resume.
        Uses mixed precision if enabled (self.amp).
        Automatically saves best validation checkpoint to checkpoints/best_val.pt.
        """
        if epochs is None:
            epochs = self.num_epochs
        if start_epoch is None:
            start_epoch = self.current_epoch

        # Loss, optimizer, scheduler bootstrap
        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        if self.optimizer is None:
            self._create_optimizer()

        total_epochs = start_epoch + epochs
        for epoch in range(start_epoch, total_epochs):
            self.model.train()
            running_loss: float = 0.0
            running_corrects: int = 0

            # --- Training loop ---
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Train]"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                # Forward pass with autocast (mixed precision)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                # Backward with GradScaler to avoid underflow
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Metrics accumulation
                _, preds = torch.max(outputs, 1)
                running_loss += float(loss.item()) * inputs.size(0)
                running_corrects += int(torch.sum(preds == labels.data))

            # LR schedule step (per epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_loss: float = running_loss / len(train_loader.dataset)
            epoch_acc: float = running_corrects / len(train_loader.dataset)

            # --- Validation loop ---
            self.model.eval()
            val_loss_sum: float = 0.0
            val_corrects: int = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    with torch.cuda.amp.autocast(enabled=self.amp):
                        outputs = self.model(inputs)
                        vloss = self.criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    val_loss_sum += float(vloss.item()) * inputs.size(0)
                    val_corrects += int(torch.sum(preds == labels.data))

            val_loss: float = val_loss_sum / len(val_loader.dataset)
            val_acc: float = val_corrects / len(val_loader.dataset)
            cur_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            self.logger.info(
                f"Epoch {epoch + 1}/{total_epochs} | "
                f"LR: {cur_lr:.6f} | "
                f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            # Save best-by-validation checkpoint
            if not hasattr(self, "best_val_acc"):
                self.best_val_acc = 0.0
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_path = self.ckpt_dir / "best_val.pt"
                self.save_checkpoint(str(best_path), save_optimizer=False)
                self.logger.info(f"💾 New best model saved (val_acc={val_acc:.4f}) → {best_path}")

            # Update internal epoch counter
            self.current_epoch = epoch + 1

        return self


    # ----------------
    # Save / Load
    # ----------------
    def save_checkpoint(self, path: str = "plant_ckpt.pt", save_optimizer: bool = False) -> Self:
        """
        Save a checkpoint. By default we skip optimizer state (safer across code changes).
        """
        data = {
            "model_state": self.model.state_dict(),
            "epoch": self.current_epoch,
            "class_names": self.class_names,
            "lr": self.lr,
        }
        if save_optimizer and self.optimizer is not None:
            data["optimizer_state"] = self.optimizer.state_dict()
        torch.save(data, path)
        self.logger.info(f"Checkpoint saved to: {path}")

        return self


    def load_checkpoint(self, path: str = "plant_ckpt.pt", load_optimizer: bool = False) -> Self:
        """
        Load a checkpoint. Optimizer state is not loaded unless explicitly requested.
        """
        ckpt = torch.load(path, map_location=self.device)
        if self.model is None:
            self.build_model()

        # IMPORTANT: If number/order of classes changed, state_dict loading may fail or misalign.
        self.model.load_state_dict(ckpt["model_state"])
        self.class_names = ckpt.get("class_names", self.class_names)
        self.current_epoch = ckpt.get("epoch", 0)
        self.lr = ckpt.get("lr", self.lr)

        self.optimizer = None
        self.scheduler = None
        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        if load_optimizer and ckpt.get("optimizer_state") is not None:
            self._create_optimizer(lr=self.lr)
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except ValueError as e:
                self.logger.warning(f"Skipping optimizer_state load due to mismatch: {e}")
                self._create_optimizer(lr=self.lr)

        self.logger.info(f"Checkpoint loaded. Current epoch: {self.current_epoch}")

        return self


    # -------------
    # Fine-tuning
    # -------------
    def fine_tune(self,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  unfreeze: bool = True) -> Self:
        """
        Fine-tune the model; by default unfreezes the backbone and continues training.
        Before starting, save a safety backup checkpoint to allow rollback.
        """
        # Safety backup before any changes
        backup_name = self.ckpt_dir / f"backup_pre_ft_e{self.current_epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        self.save_checkpoint(str(backup_name), save_optimizer=False)
        self.logger.info(f"Created rollback checkpoint: {backup_name}")

        if unfreeze:
            for p in self.model.features.parameters():
                p.requires_grad = True
        self._create_optimizer(lr = self.lr)
        self.train(train_loader, val_loader, epochs=self.num_epochs, start_epoch=self.current_epoch)

        return self


    # -------------
    # Evaluation
    # -------------
    def evaluate_test(self, test_loader: DataLoader) -> Self:
        """
        Evaluate on the test set and display metrics and plots.
        """
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                with autocast(enabled=self.amp):
                    logits = self.model(x)
                preds = logits.argmax(1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(y.numpy())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        acc = accuracy_score(y_true, y_pred)
        self.logger.info(f"Test accuracy: {acc:.4f}")

        self.logger.info("Classification report:\n" +
                         classification_report(y_true, y_pred, target_names=self.class_names))

        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion(cm, self.class_names, normalize=True)
        self._plot_per_class_accuracy(y_true, y_pred, self.class_names)

        return self

    # ---- plots ----
    @staticmethod
    def _plot_confusion(cm: np.ndarray,
                        class_names: List[str],
                        normalize: bool = True,
                        title: str = "Confusion Matrix"):
        """
        Plot confusion matrix (optionally normalized per true class).
        """
        if normalize:
            cm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig = plt.figure(figsize=(16, 12))
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


    @staticmethod
    def _plot_per_class_accuracy(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 class_names: List[str]) -> List[float]:
        """
        Compute and plot per-class accuracy.
        """
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


    # ---------------
    # ONNX export
    # ---------------
    def save_model(self) -> Self:
        """
        Export the model to ONNX and save class names for production use.
        """
        self.api_dir.mkdir(exist_ok=True)
        onnx_path = self.api_dir / "plant_disease_model.onnx"
        json_path = self.api_dir / "class_names.json"

        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        torch.onnx.export(
            self.model, dummy_input, onnx_path,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.class_names, f, ensure_ascii=False, indent=2)

        self.logger.info(f"ONNX model saved to {onnx_path}")
        self.logger.info(f"Class names saved to {json_path}")

        return self


# ==========================
# Minimal runnable examples
# ==========================
if __name__ == "__main__":
    set_seed(42)

    # Load config if present; otherwise use defaults
    cfg_path = Path(__file__).parent / "config.yaml"
    if cfg_path.exists():
        trainer = PlantDiseaseTrainer.from_yaml(cfg_path)
    else:
        trainer = PlantDiseaseTrainer()

    # 1) Prepare data
    trainer.download_dataset()
    trainer.prepare_data()

    # 2) Build dataloaders
    train_loader, val_loader, test_loader = trainer.create_dataloaders()

    # 3) Build model
    trainer.build_model()

    ckpt_path = Path("plant_ckpt.pt")
    if ckpt_path.exists():
        # If checkpoint exists → load & fine-tune
        trainer.logger.info("Checkpoint found → loading and fine-tuning...")
        trainer.load_checkpoint(str(ckpt_path), load_optimizer=False)
        trainer.fine_tune(train_loader, val_loader, unfreeze=True)
    else:
        # If no checkpoint → train from scratch
        trainer.logger.info("No checkpoint found → training from scratch...")
        trainer.train(train_loader, val_loader)

    # Save latest state and ONNX
    trainer.save_checkpoint("plant_ckpt.pt")
    trainer.save_model()

    # Final evaluation
    trainer.evaluate_test(test_loader)
