import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import CLASS_NAMES, DEFAULT_MODEL_DIR, IMAGE_SIZE


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    return np.expand_dims(image.astype(np.float32), axis=0)


def predict_keras(model_path: Path, image: np.ndarray) -> float:
    model = tf.keras.models.load_model(model_path)
    return float(model.predict(image, verbose=0)[0][0])


def predict_tflite(model_path: Path, image: np.ndarray) -> float:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    image = image.astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], image)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details["index"])[0][0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict drowsiness for one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR / "drowsiness_mobilenetv2.keras"))
    parser.add_argument("--labels", default=str(DEFAULT_MODEL_DIR / "labels.json"))
    args = parser.parse_args()

    model_path = Path(args.model)
    labels_path = Path(args.labels)
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else CLASS_NAMES
    image = load_image(Path(args.image))

    probability_non_drowsy = (
        predict_tflite(model_path, image)
        if model_path.suffix.lower() == ".tflite"
        else predict_keras(model_path, image)
    )
    class_index = 1 if probability_non_drowsy >= 0.5 else 0
    confidence = probability_non_drowsy if class_index == 1 else 1.0 - probability_non_drowsy

    print(f"Prediction: {labels[class_index]}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Non-drowsy probability: {probability_non_drowsy:.4f}")


if __name__ == "__main__":
    main()
