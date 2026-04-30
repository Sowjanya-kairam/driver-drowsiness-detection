import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import DEFAULT_MODEL_DIR, IMAGE_SIZE


def create_face_detector():
    try:
        import dlib

        return "dlib", dlib.get_frontal_face_detector()
    except ImportError:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        return "opencv", cv2.CascadeClassifier(cascade_path)


def detect_faces(detector_type: str, detector, gray_frame):
    if detector_type == "dlib":
        rects = detector(gray_frame, 0)
        return [(r.left(), r.top(), r.width(), r.height()) for r in rects]

    faces = detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return list(faces)


def preprocess_face(frame, box):
    x, y, w, h = box
    height, width = frame.shape[:2]
    pad = int(0.18 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)

    face = frame[y1:y2, x1:x2]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, IMAGE_SIZE)
    return np.expand_dims(face.astype(np.float32), axis=0), (x1, y1, x2 - x1, y2 - y1)


def load_tflite(model_path: Path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details()[0], interpreter.get_output_details()[0]


def predict(interpreter, input_details, output_details, image: np.ndarray) -> float:
    image = image.astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], image)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details["index"])[0][0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time driver drowsiness detection.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR / "drowsiness_mobilenetv2.tflite"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alert-frames", type=int, default=12)
    args = parser.parse_args()

    detector_type, detector = create_face_detector()
    interpreter, input_details, output_details = load_tflite(Path(args.model))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}")

    consecutive_drowsy = 0
    previous_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector_type, detector, gray)
        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)

        status = "No face"
        color = (180, 180, 180)

        if faces:
            face_image, draw_box = preprocess_face(frame, faces[0])
            non_drowsy_prob = predict(interpreter, input_details, output_details, face_image)
            is_drowsy = non_drowsy_prob < args.threshold
            consecutive_drowsy = consecutive_drowsy + 1 if is_drowsy else 0

            status = "Drowsy" if is_drowsy else "Non-Drowsy"
            color = (0, 0, 255) if is_drowsy else (0, 180, 0)
            x, y, w, h = draw_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{status} ({non_drowsy_prob:.2f})",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
        else:
            consecutive_drowsy = 0

        if consecutive_drowsy >= args.alert_frames:
            cv2.putText(frame, "DROWSINESS ALERT", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        current_time = time.time()
        fps = 1.0 / max(current_time - previous_time, 1e-6)
        previous_time = current_time
        cv2.putText(frame, f"{detector_type.upper()} | FPS: {fps:.1f}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
