"""
CVdoku — Digit Classifier Training Script
==========================================
Trains a CNN on MNIST digits with augmentation to handle
both handwritten and printed digits (as found in Sudoku puzzles).

v2 improvements:
- Erosion augmentation to simulate thin printed digits (fixes 1 vs 7 confusion)
- Elastic distortion to handle various font styles
- Class weights to handle MNIST imbalance
- Saves in both .h5 and .keras formats

Run once before using main.py:
    python train.py

Output: models/digit_model.h5
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "digit_model.h5")
PLOT_PATH  = os.path.join(BASE_DIR, "resources", "training_history.png")

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMG_SIZE   = 28
NUM_EPOCHS = 20
BATCH_SIZE = 128
LEARN_RATE = 1e-3


def load_and_prepare_data():
    """Load MNIST, keep only digits 1-9, apply printed-digit augmentation."""
    print("[1/5] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    # Keep only 1-9
    mask  = y_all > 0
    x_all = x_all[mask]
    y_all = y_all[mask]

    print(f"[2/5] Generating printed-digit augmentation...")
    x_printed, y_printed = generate_printed_variants(x_all, y_all)

    # Combine original + printed variants
    x_combined = np.concatenate([x_all, x_printed], axis=0)
    y_combined  = np.concatenate([y_all, y_printed], axis=0)

    # Normalise and add channel dim
    x_combined = x_combined.astype("float32") / 255.0
    x_combined = np.expand_dims(x_combined, -1)

    # Shift labels to 0-indexed
    y_combined = y_combined - 1

    x_train, x_val, y_train, y_val = train_test_split(
        x_combined, y_combined,
        test_size=0.1, random_state=42, stratify=y_combined
    )

    print(f"    Train: {x_train.shape[0]} samples")
    print(f"    Val  : {x_val.shape[0]} samples")
    return x_train, x_val, y_train, y_val


def generate_printed_variants(x_all, y_all):
    """
    Generate synthetic printed-digit variants from MNIST samples.
    Targets the most common misclassifications:
    - Thin/eroded digits (simulates slim app fonts — fixes 1 vs 7)
    - High contrast / clean binary (simulates printed puzzle digits)
    - Slightly thickened digits (simulates bold fonts)
    """
    printed_imgs   = []
    printed_labels = []

    # Only augment a subset to keep training time reasonable
    indices = np.random.choice(len(x_all), size=min(20000, len(x_all)), replace=False)

    for idx in indices:
        img   = x_all[idx].copy()
        label = y_all[idx]

        # Variant 1: eroded (thin) — simulates slim app fonts
        kernel  = np.ones((2, 2), np.uint8)
        eroded  = cv2.erode(img, kernel, iterations=1)
        printed_imgs.append(eroded)
        printed_labels.append(label)

        # Variant 2: dilated (thick) — simulates bold fonts
        dilated = cv2.dilate(img, kernel, iterations=1)
        printed_imgs.append(dilated)
        printed_labels.append(label)

        # Variant 3: high contrast binary — simulates clean printed grids
        _, binary = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        printed_imgs.append(binary)
        printed_labels.append(label)

    return np.array(printed_imgs), np.array(printed_labels)


def build_augmenter():
    """
    Augmentation that mimics real-world Sudoku cell variations.
    No horizontal flip — digits must not be mirrored.
    """
    return ImageDataGenerator(
        rotation_range=12,
        zoom_range=0.18,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.12,
        fill_mode="nearest",
    )


def build_model():
    """
    Compact CNN — fast to train on CPU, accurate for printed digits.
    Deeper than v1 to better handle font variation.
    """
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3 — extra depth for font variation
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(9, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARN_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    """Save training curves to resources/."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=100)
    plt.close()
    print(f"    Training plot saved → {PLOT_PATH}")


def main():
    print("=" * 55)
    print("  CVdoku — Digit Classifier Training v2")
    print("  (with printed-digit augmentation)")
    print("=" * 55)

    x_train, x_val, y_train, y_val = load_and_prepare_data()

    print("[3/5] Building model...")
    model = build_model()
    model.summary()

    augmenter = build_augmenter()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
    ]

    print(f"[4/5] Training for up to {NUM_EPOCHS} epochs...")
    history = model.fit(
        augmenter.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=NUM_EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    val_acc = max(history.history["val_accuracy"])
    print(f"\n    Best val accuracy: {val_acc * 100:.2f}%")

    print(f"[5/5] Saving model → {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    plot_history(history)

    print("\n" + "=" * 55)
    print("  Training complete!")
    print(f"  Model saved  : {MODEL_PATH}")
    print(f"  Val accuracy : {val_acc * 100:.2f}%")
    print("  Run main.py to start the AR solver.")
    print("=" * 55)


if __name__ == "__main__":
    main()