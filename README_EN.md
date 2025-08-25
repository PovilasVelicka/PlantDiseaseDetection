# 🌱 Plant Disease Detection Project

This project consists of three parts:

1. **training/** – training and fine-tuning the deep learning model  
2. **plant_api/** – FastAPI backend with the trained ONNX model  
3. **plant_frontend/** – simple HTML/JS frontend (`index.html`) to upload a leaf image and get predictions  

---

## 📂 Project Structure

```
training/         # Training code, dataset preparation, model saving
plant_api/        # FastAPI server that loads the ONNX model
plant_frontend/   # Frontend to interact with the API (static HTML)
```

---

## 📥 Dataset

- The dataset is automatically downloaded from **Kaggle** using the ID in `config.yaml`.
- It is then split into:
  - **train** (70%) – model training
  - **val** (20%) – validation (model selection)
  - **test** (10%) – final accuracy reporting

All splits are **stratified** by class (so all classes are represented in each split).

> 💡 You can optionally add additional images (e.g., Unknown or real photos) manually to improve generalization.

---

## 🧠 Training & Fine-tuning

The model is based on **MobileNetV2** (pretrained on ImageNet).  
By default, only the classifier is trained — the feature extractor is frozen.  
You can later unfreeze it and fine-tune.

Training features:
- CrossEntropy loss with **label smoothing**
- Cosine annealing LR scheduler
- Optional **weighted sampling** for class imbalance
- Logging to `logs/` with timestamps
- Backup created before fine-tuning

Run training directly from:

```bash
python training/train_module.py
```

---

## 💾 Model Versions

| File | Description |
|------|-------------|
| `plant_ckpt.pt` | Last trained model checkpoint |
| `best_val.pt` | Best model by validation accuracy |
| `plant_disease_model.onnx` | Exported ONNX model (used by API) |
| `class_names.json` | List of class names |

---

## ⚙️ Configuration

Edit `config.yaml` to change training parameters:

```yaml
batch_size: 32
num_epochs: 10
lr: 0.001
weight_decay: 0.0001
label_smoothing: 0.05
use_weighted_sampler: true
amp: false  # Mixed precision (recommended if using GPU)
```

---

## 🌐 Running the API

1. Copy the trained model to the API folder:
   - `plant_disease_model.onnx`
   - `class_names.json`

2. Start the FastAPI server:

```bash
cd plant_api
uvicorn app:app --host 0.0.0.0 --port 8000
```

3. API Endpoints:

| Method | URL        | Description           |
|--------|------------|-----------------------|
| GET    | `/health`  | Health check          |
| POST   | `/predict` | Upload image & get result |

---

## 🖼️ Frontend (index.html)

- Open `plant_frontend/index.html` in a browser
- Upload a leaf image
- The page sends it to the backend API and shows the top prediction

⚠️ Make sure the API is running before opening the frontend.  
If your server is not running on `localhost:8000`, you may need to update the URL in `index.html`.

