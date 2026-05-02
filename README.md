# Driver Drowsiness Detection System

AI-powered drowsiness classifier using MobileNetV2 transfer learning, OpenCV/Dlib face detection, and TensorFlow Lite export for edge deployment.

## Dataset

The provided `dataset.zip` contains 10,000 PNG images:

| Split | Drowsy | Non-Drowsy |
| --- | ---: | ---: |
| Train | 3,750 | 3,750 |
| Test | 1,250 | 1,250 |

Expected extracted layout:

```text
dataset/
  train/
    drowsy/
    non_drowsy/
  test/
    drowsy/
    non_drowsy/
```

## Setup

Use a virtual environment for this project. The global Python environment can easily pick up incompatible scientific packages, especially with TensorFlow.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dlib is optional because it can be awkward to install on Windows. If available, the webcam script will use it automatically:

```powershell
pip install dlib
```

## Prepare Data

```powershell
python src\prepare_dataset.py --zip dataset.zip --output .
```

## Preprocess Dataset with Face Detection (Recommended)

For better consistency between training and inference, extract face regions before training:

```powershell
python src\preprocess_dataset_with_face_detection.py --input dataset --output dataset_processed
```

This step:
- Detects faces in all images using OpenCV (or dlib if installed)
- Extracts face regions with padding (~18% of face size)
- Reduces dataset size by ~24× for faster training
- **Ensures train-inference consistency**: both pipelines now use face regions only

Use `--use-dlib` flag for more accurate face detection (slower):

```powershell
python src\preprocess_dataset_with_face_detection.py --input dataset --output dataset_processed --use-dlib
```

## Train and Export

Train on raw images (current approach):

```powershell
python src\train.py --data-dir dataset --epochs 12 --fine-tune-epochs 5
```

Or train on face-extracted images (recommended for production):

```powershell
python src\train.py --data-dir dataset_processed --epochs 12 --fine-tune-epochs 5
```

Outputs are written to `models/`:

- `drowsiness_mobilenetv2.keras`
- `drowsiness_mobilenetv2.tflite`
- `labels.json`
- `metrics.json`

## Predict One Image

```powershell
python src\predict_image.py --image dataset\test\drowsy\A0001.png
```

Use the TFLite model:

```powershell
python src\predict_image.py --image dataset\test\drowsy\A0001.png --model models\drowsiness_mobilenetv2.tflite
```

## Real-Time Webcam Demo

```powershell
python src\realtime_inference.py --model models\drowsiness_mobilenetv2.tflite
```

Press `q` to quit. The script detects the face, classifies the cropped face, and raises an alert after several consecutive drowsy frames.

## Project Approach

1. **Face Detection Preprocessing** (optional but recommended): Extract face regions before training to match the inference pipeline.
2. Load images with TensorFlow directory datasets and MobileNetV2 preprocessing.
3. Train a light classification head while the MobileNetV2 base is frozen.
4. Fine-tune the upper layers of the base network at a lower learning rate.
5. Evaluate on the held-out test split.
6. Convert the trained model to TensorFlow Lite for low-latency edge inference.
7. Run webcam inference with face detection and temporal smoothing to reduce noisy alerts.

## Why Train-Inference Consistency Matters

The inference pipeline uses face detection to extract facial regions before classification. Training on full images creates a **train-inference mismatch**:

| Stage | Input | Issue |
| --- | --- | --- |
| **Training** (without preprocessing) | Full cabin image with background | Model learns background context |
| **Inference** | Face region only (extracted by face detector) | Model receives different input distribution |

This mismatch can hurt real-world performance. **Recommended workflow**:
1. Run face detection preprocessing once: `preprocess_dataset_with_face_detection.py`
2. Train on face-extracted images: `train.py --data-dir dataset_processed`
3. Inference uses the same face regions → consistent performance
