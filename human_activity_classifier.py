"""
Radar HAR Classifier
Human Activity Recognition from Radar Spectrograms
Presented at ICCIS 2025 — BITS Pilani, Goa Campus

Authors: Shri Manasa S, Sarvani Sruthi Chundi
"""

import os
import io
import cv2
import numpy as np
import psycopg2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from scipy.fftpack import fft2, fftshift
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras import layers, models

# ── Configuration ──────────────────────────────────────────────────────────────

DB_CONFIG = {
    "dbname":   os.environ.get("DB_NAME", "postgres"),
    "user":     os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD"),          # set in environment
    "host":     os.environ.get("DB_HOST", "localhost"),
    "port":     os.environ.get("DB_PORT", "5432"),
}

IMAGE_SIZE   = (64, 64)
TEST_SPLIT   = 0.2
RANDOM_SEED  = 42
CNN_EPOCHS   = 10
CNN_BATCH    = 32
SAMPLE_DIR   = "sample_images"
OUTPUT_DIR   = "outputs"

# ── Helpers ────────────────────────────────────────────────────────────────────

def to_magnitude_spectrum(img_array: np.ndarray) -> np.ndarray:
    """Apply 2-D FFT and return log-magnitude spectrum."""
    shifted = fftshift(fft2(img_array))
    spectrum = np.log(np.abs(shifted) + 1)
    if spectrum.shape != IMAGE_SIZE:
        spectrum = cv2.resize(spectrum, IMAGE_SIZE)
    return spectrum


def decode_bytea(raw: bytes, idx: int) -> np.ndarray | None:
    """
    Try two strategies to decode a raw bytea blob into a float64
    grayscale array of shape IMAGE_SIZE.
    Returns None if both strategies fail.
    """
    # Strategy 1 — treat as encoded image file (PNG / JPEG / …)
    try:
        img = Image.open(io.BytesIO(raw)).convert("L").resize(IMAGE_SIZE)
        return np.array(img) / 255.0
    except Exception:
        pass

    # Strategy 2 — treat as raw pixel bytes
    try:
        arr = np.frombuffer(raw, dtype=np.uint8)
        side = int(np.sqrt(len(arr)))
        for dim in [28, 32, 64, 96, 128, 224, 256, side]:
            if len(arr) >= dim * dim:
                arr = arr[: dim * dim].reshape(dim, dim)
                resized = cv2.resize(arr.astype(np.float32), IMAGE_SIZE)
                return resized / 255.0
    except Exception:
        pass

    print(f"  [warn] Could not decode image {idx} — skipping.")
    return None


def save_debug_samples(raw_img: np.ndarray, spectrum: np.ndarray, idx: int) -> None:
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw_img, cmap="gray");   axes[0].set_title(f"Original {idx}")
    axes[1].imshow(spectrum, cmap="viridis"); axes[1].set_title("FFT Spectrum")
    fig.savefig(os.path.join(SAMPLE_DIR, f"sample_{idx}.png"))
    plt.close(fig)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray]:
    print("Connecting to PostgreSQL …")
    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT image_data, classification_result FROM radar_images")
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"Fetched {len(records)} records from database.")

    images, labels = [], []

    for idx, (raw, label) in enumerate(records):
        img_array = decode_bytea(bytes(raw), idx)
        if img_array is None:
            continue

        spectrum = to_magnitude_spectrum(img_array)

        if idx < 5:
            save_debug_samples(img_array, spectrum, idx)

        images.append(spectrum)
        labels.append(label)

    print(f"Successfully processed {len(images)} / {len(records)} images.")

    if not images:
        raise RuntimeError("No images could be processed. Aborting.")

    return np.array(images), np.array(labels)

# ── Model training & evaluation ────────────────────────────────────────────────

def train_svm(X_train, X_test, y_train, y_test) -> float:
    print("\nTraining SVM …")
    clf = SVC(kernel="linear", random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  SVM Accuracy : {acc:.4f}")
    return acc


def train_knn(X_train, X_test, y_train, y_test) -> float:
    print("Training KNN …")
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  KNN Accuracy : {acc:.4f}")
    return acc


def build_cnn(num_classes: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu",
                      input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(X_train, X_test, y_train, y_test, num_classes: int) -> tuple[float, tf.keras.callbacks.History]:
    print("Training CNN …")
    X_tr = X_train[..., np.newaxis]
    X_te = X_test[..., np.newaxis]
    model   = build_cnn(num_classes)
    history = model.fit(
        X_tr, y_train,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH,
        validation_split=0.2,
        verbose=1,
    )
    acc = accuracy_score(y_test, np.argmax(model.predict(X_te), axis=1))
    print(f"  CNN Accuracy : {acc:.4f}")
    return acc, history

# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_comparison(accuracies: dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = list(accuracies.keys())
    vals  = list(accuracies.values())
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, vals, color=colors, width=0.5)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 v + 0.01, f"{v:.2%}", ha="center", fontsize=11)
    plt.title("Model Accuracy Comparison — Radar HAR", fontsize=14)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.show()
    print("  Saved → outputs/model_comparison.png")


def plot_cnn_history(history: tf.keras.callbacks.History) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("CNN Accuracy");  axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Epoch");        axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("CNN Loss");  axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch");    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cnn_training_history.png"), dpi=150)
    plt.show()
    print("  Saved → outputs/cnn_training_history.png")

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load & preprocess
    images, labels = load_data()

    # 2. Encode string labels
    encoder       = LabelEncoder()
    numeric_labels = encoder.fit_transform(labels)
    num_classes   = len(encoder.classes_)
    print(f"\nClasses ({num_classes}): {list(encoder.classes_)}")

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        images, numeric_labels,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=numeric_labels,
    )
    X_flat_train = X_train.reshape(len(X_train), -1)
    X_flat_test  = X_test.reshape(len(X_test),  -1)

    # 4. Train models
    acc_svm           = train_svm(X_flat_train, X_flat_test, y_train, y_test)
    acc_knn           = train_knn(X_flat_train, X_flat_test, y_train, y_test)
    acc_cnn, history  = train_cnn(X_train, X_test, y_train, y_test, num_classes)

    # 5. Plot results
    plot_comparison({"SVM": acc_svm, "KNN": acc_knn, "CNN": acc_cnn})
    plot_cnn_history(history)

    print("\n✅ Analysis complete. All outputs saved to /outputs/")


if __name__ == "__main__":
    main()
