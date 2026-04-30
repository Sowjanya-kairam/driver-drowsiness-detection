import argparse
import json
from pathlib import Path

import tensorflow as tf

from config import BATCH_SIZE, CLASS_NAMES, DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, IMAGE_SIZE, SEED


def build_datasets(data_dir: Path, batch_size: int):
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected {data_dir}/train and {data_dir}/test. Run prepare_dataset.py first."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
    )


def build_model(weights: str | None = "imagenet") -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights=weights,
        name="mobilenetv2_base",
    )
    base.trainable = False

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="drowsiness_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def fine_tune_model(model: tf.keras.Model, trainable_from: int = 100) -> None:
    base_model = model.get_layer("mobilenetv2_base")
    base_model.trainable = True
    for layer in base_model.layers[:trainable_from]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def evaluate_model(model: tf.keras.Model, test_ds) -> dict:
    y_true = []
    y_prob = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy().astype(int).ravel().tolist())

    for images, _ in test_ds:
        y_prob.extend(model.predict(images, verbose=0).ravel().tolist())

    y_pred = [1 if prob >= 0.5 else 0 for prob in y_prob]
    matrix = [[0, 0], [0, 0]]
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1

    per_class = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = matrix[class_index][class_index]
        false_positive = sum(matrix[row][class_index] for row in range(2)) - true_positive
        false_negative = sum(matrix[class_index]) - true_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": sum(matrix[class_index]),
        }

    accuracy = sum(matrix[i][i] for i in range(2)) / max(len(y_true), 1)
    return {
        "class_names": CLASS_NAMES,
        "accuracy": accuracy,
        "classification_report": per_class,
        "confusion_matrix": matrix,
    }


def export_tflite(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 drowsiness detector.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--weights",
        default="imagenet",
        choices=["imagenet", "none"],
        help="Use ImageNet transfer learning weights or random initialization.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_datasets(data_dir, args.batch_size)
    model = build_model(weights=None if args.weights == "none" else args.weights)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_auc", mode="max"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_dir / "drowsiness_mobilenetv2.keras",
            save_best_only=True,
            monitor="val_auc",
            mode="max",
        ),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    if args.fine_tune_epochs > 0:
        fine_tune_model(model)
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )
        for key, values in fine_history.history.items():
            history.history.setdefault(key, []).extend(values)

    keras_path = model_dir / "drowsiness_mobilenetv2.keras"
    model.save(keras_path)
    export_tflite(model, model_dir / "drowsiness_mobilenetv2.tflite")

    metrics = evaluate_model(model, test_ds)
    metrics["history"] = history.history
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (model_dir / "labels.json").write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")

    print(f"Saved Keras model: {keras_path.resolve()}")
    print(f"Saved TFLite model: {(model_dir / 'drowsiness_mobilenetv2.tflite').resolve()}")
    print(f"Saved metrics: {(model_dir / 'metrics.json').resolve()}")


if __name__ == "__main__":
    main()
