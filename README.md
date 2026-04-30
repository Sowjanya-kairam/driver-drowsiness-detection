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

## Train and Export

```powershell
python src\train.py --data-dir dataset --epochs 12 --fine-tune-epochs 5
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

1. Load images with TensorFlow directory datasets and MobileNetV2 preprocessing.
2. Train a light classification head while the MobileNetV2 base is frozen.
3. Fine-tune the upper layers of the base network at a lower learning rate.
4. Evaluate on the held-out test split.
5. Convert the trained model to TensorFlow Lite for low-latency edge inference.
6. Run webcam inference with face detection and temporal smoothing to reduce noisy alerts.
