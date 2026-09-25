<div align="center">

# 🚦 Traffic Sign Detection — 61-Class Big Dataset Model

**A YOLOv8-based object detection model for real-time traffic sign & traffic light recognition**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

</div>

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Requirements](#-requirements)
3. [Project Structure](#-project-structure)
4. [Data](#-data)
5. [Installation](#-installation)
6. [Project Methodology](#-project-methodology)
7. [Visualization](#-visualization)
8. [Technology Used](#-technology-used)
9. [Future Improvements](#-future-improvements)

---

## 🔎 Overview

This project fine-tunes a pretrained **YOLOv8s** model to detect and classify **61 types** of traffic signs and traffic lights. It builds on an earlier 34-class version by moving to a larger, more diverse Roboflow dataset, translating French class labels to English, and injecting **harsh-condition augmentation** (fog, rain, motion blur, low light, occlusion) into training data so the model generalizes better to real-world driving conditions.

| Metric | Score |
|---|---|
| 🎯 Precision | **0.879** |
| 🔁 Recall | **0.850** |
| 📦 mAP50 | **0.892** |
| 📊 mAP50-95 | **0.754** |

> Improves on both the earlier 34-class model and the harsh-condition-only "Albumented" model across all four metrics.

---

## ⚙️ Requirements

- Python 3.10+
- NVIDIA GPU (trained/tested on a Tesla T4, via Google Colab)
- Google Drive (used for persistent storage of dataset/checkpoints)
- Roboflow API key ([get one free here](https://app.roboflow.com/settings/api))

**Python packages:**
```
ultralytics
roboflow
albumentations
opencv-python
matplotlib
PyYAML
tqdm
```

---

## 🗂️ Project Structure

```
Diploma Major Project(DCSE691)/
└── updated Model with 61 classes/
    ├── dataset/
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   ├── valid/
    │   ├── test/
    │   └── data.yaml
    ├── TrafficSign_TrainedModel_BigData/        # first training run
    ├── TrafficSign_TrainedModel_BigData_continued/
    │   └── weights/
    │       ├── best.pt
    │       └── last.pt
    └── Test_Prediction_Results/                 # saved inference outputs
```

---

## 📊 Data

- **Source:** Roboflow — *"Traffic Signs and Traffic Lights"* (workspace: `raouf-xjwsn`, project: `raouf`)
- **Classes:** 61 (originally French-labeled, translated to English — e.g. `Vitesse limitee` → `Speed Limit`, `Cedez le passage` → `Give Way`)
- **Validation set:** 861 images, 1,673 labeled instances
- **Format:** YOLOv8 (`data.yaml`)
- **Augmentation:** Synthetic fog, rain, motion blur, low brightness, and occlusion patches applied to the **training split only** — validation/test are kept clean for a fair, unbiased evaluation.

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Install dependencies
pip install -q ultralytics roboflow albumentations

# 3. Set your Roboflow API key
export ROBOFLOW_API_KEY="your-api-key-here"

# 4. Run the notebook / training script
```

---

## 🧭 Project Methodology

1. **Environment setup** — verify GPU availability, install core libraries
2. **Mount storage** — connect Google Drive for persistent dataset/model storage
3. **Download dataset** — pull the labeled dataset from Roboflow (cached after first run)
4. **Fix `data.yaml`** — convert relative paths to absolute so training doesn't break
5. **Translate class names** — map French label names to English equivalents
6. **Harsh-condition augmentation** — apply fog/rain/blur/brightness/occlusion transforms to the training set only
7. **Train** — fine-tune `yolov8s.pt` for 100 epochs, with checkpoint-resume support
8. **Evaluate** — run `model.val()` on the held-out validation split
9. **Predict** — run inference on the test set and on manually uploaded images
10. **Export** — export the final model to **ONNX** for deployment

**Training configuration:**

| Parameter | Value |
|---|---|
| Epochs | 100 |
| Image size | 640 |
| Batch size | 16 |
| Patience (early stopping) | 20 |
| Seed | 42 |
| Horizontal / vertical flip | 0 / 0 |
| Rotation (degrees) | 5 |
| Translate | 0.10 |
| Scale | 0.30 |
| Shear | 2.0 |
| Perspective | 0.0005 |
| HSV (h / s / v) | 0.015 / 0.70 / 0.40 |

---

## 📈 Visualization

The training run automatically generates (via Ultralytics `plots=True`):
- 📉 Loss curves (box loss, classification loss, DFL loss) across epochs
- 📊 Precision-Recall and F1-confidence curves
- 🧩 Confusion matrix across all 61 classes
- 🖼️ Sample batches with predicted vs. ground-truth bounding boxes

Additional visual checks included in the pipeline:
- Preview grid of harsh-condition augmented training images (sanity check before training)
- Annotated test-set predictions with bounding boxes and class labels
- Manually uploaded image inference, plotted with `results[0].plot()`

*(Add your actual result images/GIFs here once exported from the `runs/` or `Test_Prediction_Results/` folder, e.g. `![Confusion Matrix](assets/confusion_matrix.png)`)*

---

## 🧰 Technology Used

| Category | Tools |
|---|---|
| Model | ![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-00FFFF?logo=yolo&logoColor=black) |
| Framework | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) |
| Data augmentation | ![Albumentations](https://img.shields.io/badge/Albumentations-FF6F00) |
| Dataset hosting | ![Roboflow](https://img.shields.io/badge/Roboflow-6706CE?logo=roboflow&logoColor=white) |
| Compute | ![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?logo=googlecolab&logoColor=white) |
| Image processing | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white) |
| Export format | ![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white) |

---

## 🚀 Future Improvements

- ⚖️ Address class imbalance — several classes have very few labeled examples, hurting per-class accuracy
- 🇮🇳 Fine-tune or extend with India-specific traffic sign data (current dataset is not India-specific)
- 📱 Deploy the ONNX model to a lightweight edge/mobile app for real-time detection
- 🎥 Add video-stream inference support, not just static images
- 🧪 Run systematic hyperparameter tuning (learning rate, augmentation strength) instead of fixed values
- 📦 Package as a reusable pip module / API endpoint for easy integration into other projects

---

<div align="center">
Made with YOLOv8 · Trained on Google Colab
</div>
