# 🌿 Plant Disease Detection

This project allows you to **run a ready-to-use plant disease classifier** via a local API and frontend, or train your own model from scratch or by fine-tuning.

---

## 🚀 1. Quick Start (NO Python installed)

> If you don’t have Python:  
➡️ Download and install [Python 3.10+](https://www.python.org/downloads/)  
✅ During installation, check **"Add Python to PATH"**

---

## 🧩 2. Install dependencies

In terminal or PowerShell, go to the project folder and run:

```bash
pip install -r requirements.txt
```

---

## ▶️ 3. Run the ready-to-use model

```bash
python app.py
```

This will:
- ✅ Start FastAPI server at `http://localhost:8000`
- 🌐 Open the frontend from `plant_frontend/index.html` in your browser

---

## 🧪 4. Test the API

Use sample images in `test_images/` folder to test predictions.

- You can upload:
  - 📷 Images of leaves (from test_images)
  - 📎 Or random non-leaf images to test robustness

---

## 🧠 5. Train or Fine-tune your own model (optional)

All training is managed inside `training/train_module.py`.

### 📦 Dataset

Automatically downloaded from Kaggle (`emmarex/plantdisease`) and split into:

- `train`: 70%
- `val`: 20%
- `test`: 10%

Classes with less than 2 images are skipped.

---

### 🔁 Fine-tuning existing model

If a checkpoint `plant_ckpt.pt` is present, the model will **resume training** from that point.

It will use parameters from `training/config.yaml`:

```yaml
extra_epochs: 5
lr: 0.0001
```

You can adjust learning rate and number of extra epochs there.

---

### 🧱 Train from scratch

If no checkpoint is found — model will be trained from scratch using:

- MobileNetV2 backbone (initially frozen)
- Cosine LR scheduler
- Label smoothing

---

### 📂 Optional: Add `Unknown/` class

To improve model generalization:

- Add a directory `Unknown/` in the dataset
- Place **non-leaf images** in it (e.g., buildings, people, animals)
- Images should be `256x256` size or similar

This helps model learn what *not* to classify as a disease.

---

## 📁 Project Structure

```
project_root/
├── app.py                  # Launcher: runs API and frontend
├── requirements.txt        # Dependencies (API + training)
├── test_images/            # Images to test model via UI
├── training/
│   ├── train_module.py     # Model training and evaluation
│   └── config.yaml         # Parameters for fine-tuning
├── plant_api/
│   ├── main.py             # FastAPI backend
│   ├── plant_disease_model.onnx   # Trained model (exported)
│   └── class_names.json    # Class labels
├── plant_frontend/
│   └── index.html          # User interface (HTML)
```

---

## 🛑 Stop the server

When `app.py` is running, press **ENTER** in terminal to stop the server.

---
