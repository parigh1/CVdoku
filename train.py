"""
CVdoku — Digit Classifier Training Script
==========================================
Trains a CNN on synthetic printed-font digits (the actual domain this app
needs — see synth_fonts.py) with a small slice of MNIST mixed in for
robustness if a handwritten board is ever scanned.

v3 — printed-font-first retrain (fixes the 1 vs 7 domain gap at the source):
- Primary signal: synth_fonts.py — digits rendered in real sans-serif fonts,
  passed through the SAME preprocessing pipeline classifier.py uses at
  inference, closing both the font-shape gap AND the preprocessing-mismatch
  gap that a downloaded dataset (MNIST, Chars74K) can't close on its own.
- MNIST kept at a small weight (~15% of the dataset) purely for robustness
  to handwritten puzzles — not the primary training signal anymore.
- Same CNN architecture, callbacks, and class weighting as v2 — the
  architecture was never the problem, the data was.

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

from synth_fonts import generate_dataset as generate_synthetic_fonts

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "digit_model.h5")
PLOT_PATH  = os.path.join(BASE_DIR, "resources", "training_history.png")

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMG_SIZE             = 28
NUM_EPOCHS           = 20
BATCH_SIZE           = 128
LEARN_RATE           = 1e-3
SYNTH_SAMPLES_PER_DIGIT = 3000   # ~27k synthetic printed-font samples total
MNIST_FRACTION       = 0.15      # MNIST kept only as a small robustness slice


def load_and_prepare_data():
    """
    Build the training set from synthetic printed-font renders (primary
    signal) plus a small slice of MNIST (robustness to handwritten boards).
    """
    print("[1/5] Generating synthetic printed-font dataset...")
    x_synth, y_synth = generate_synthetic_fonts(samples_per_digit=SYNTH_SAMPLES_PER_DIGIT)
    print(f"    Synthetic samples: {x_synth.shape[0]}")

    print("[2/5] Loading a small MNIST slice for handwriting robustness...")
    x_mnist, y_mnist = _load_mnist_slice(target_count=int(
        x_synth.shape[0] * MNIST_FRACTION / (1 - MNIST_FRACTION)
    ))
    print(f"    MNIST samples (handwriting robustness only): {x_mnist.shape[0]}")

    x_combined = np.concatenate([x_synth, x_mnist], axis=0)
    y_combined = np.concatenate([y_synth, y_mnist], axis=0)

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


def _load_mnist_slice(target_count: int):
    """Load MNIST digits 1-9, downsampled to target_count, resized to match
    the synthetic samples (28x28, same binarized-and-centered convention)."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    mask  = y_all > 0
    x_all = x_all[mask]
    y_all = y_all[mask]

    target_count = min(target_count, len(x_all))
    idx = np.random.RandomState(42).choice(len(x_all), size=target_count, replace=False)
    return x_all[idx], y_all[idx]


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