Traffic Sign Detection — 61-Class Big Dataset Model

A YOLOv8-based object detection model that identifies and localizes traffic signs and traffic lights in images. This version extends the earlier 34-class model to 61 classes using a larger, more diverse dataset, and includes harsh-condition (fog/rain/blur) data augmentation for real-world robustness.

Overview

This project fine-tunes a pretrained YOLOv8s model on a labeled traffic sign dataset sourced from Roboflow, with class names translated from French to English. The pipeline covers dataset preparation, training, checkpoint-resume training, evaluation, inference, and export to ONNX for deployment.

Dataset
Source: Roboflow — "Traffic Signs and Traffic Lights" (workspace: raouf-xjwsn, project: raouf)
Classes: 61 (originally labeled in French, translated to English — e.g. Vitesse limitee → Speed Limit, Cedez le passage → Give Way)
Validation set: 861 images, 1,673 labeled instances
Format: YOLOv8 (data.yaml)
Tech Stack
Model: YOLOv8s (yolov8s.pt, Ultralytics)
Libraries: ultralytics, roboflow, albumentations, opencv-python, matplotlib, PyYAML
Training hardware: Google Colab, Tesla T4 GPU
Ultralytics version: 8.4.145
Pipeline
Environment setup — verify GPU, install ultralytics, roboflow, albumentations
Mount Google Drive — for persistent dataset and model storage
Download dataset — pulled from Roboflow, only downloaded once
Fix data.yaml — convert relative paths to absolute so training doesn't fail
Translate class names — French label names mapped to English equivalents
Harsh-condition augmentation — synthetic fog, rain, motion blur, low brightness, and occlusion patches applied to the training set only (validation/test kept clean for fair evaluation)
Train — fine-tune yolov8s.pt for 100 epochs (with checkpoint-resume support if training is interrupted)
Evaluate — run model.val() on the validation split
Predict — run inference on the test image set, and optionally on a manually uploaded image
Export — export the trained model to ONNX format for deployment
Training Configuration
Parameter	Value
Epochs	100
Image size	640
Batch size	16
Patience (early stopping)	20
Seed	42
Horizontal/vertical flip	0 / 0
Rotation (degrees)	5
Translate	0.10
Scale	0.30
Shear	2.0
Perspective	0.0005
HSV (h / s / v)	0.015 / 0.70 / 0.40
Results
Metric	Score
Precision	0.879
Recall	0.850
mAP50	0.892
mAP50-95	0.754

This is an improvement over both the earlier 34-class model and the harsh-condition-only "Albumented" model on all four metrics.

Usage

Run inference on a folder of images:

python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model.predict(source="path/to/images", imgsz=640, conf=0.25, save=True)

Export to ONNX:

python
model.export(format="onnx")
Notes / Known Limitations
Class distribution is imbalanced — some classes have very few labeled examples, which can affect per-class accuracy.
Training was resumed from a checkpoint partway through (last.pt), so epoch numbering in the logs restarts rather than continuing sequentially.
