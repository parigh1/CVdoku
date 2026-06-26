import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "digit_model.h5")
PLOT_PATH  = os.path.join(BASE_DIR, "resources", "training_history.png")

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMG_SIZE   = 28       # MNIST native size — no resizing needed
NUM_EPOCHS = 15
BATCH_SIZE = 128
LEARN_RATE = 1e-3


def load_and_prepare_data():
    """Load MNIST, keep only digits 1-9 (0 is treated as blank cell)."""
    print("[1/4] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Combine train + test so we can re-split ourselves
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    # Keep only 1-9 — digit 0 represents an empty cell, not a puzzle value
    mask = y_all > 0
    x_all = x_all[mask]
    y_all = y_all[mask]

    # Normalise to [0, 1] and add channel dim → (N, 28, 28, 1)
    x_all = x_all.astype("float32") / 255.0
    x_all = np.expand_dims(x_all, -1)

    # Shift labels: 1→0, 2→1, ..., 9→8  (keras needs 0-indexed classes)
    y_all = y_all - 1

    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=0.1, random_state=42, stratify=y_all
    )

    print(f"    Train: {x_train.shape[0]} samples")
    print(f"    Val  : {x_val.shape[0]} samples")
    return x_train, x_val, y_train, y_val


def build_augmenter():
    """
    Light augmentation that mimics real-world Sudoku cell variations:
    - Small rotations (puzzle held slightly crooked)
    - Zoom (digit size varies per puzzle)
    - Width/height shift (digit not perfectly centred in cell)
    No horizontal flip — digits must not be mirrored.
    """
    return ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode="nearest",
    )


def build_model():
    """
    Compact CNN — fast enough to train on CPU in a few minutes,
    accurate enough for clean Sudoku digit cells (>99% on MNIST val).

    Architecture:
        Conv(32) → Conv(64) → MaxPool → Dropout
        Conv(64) → MaxPool → Dropout
        Flatten → Dense(128) → Dropout → Dense(9, softmax)
    """
    model = keras.Sequential([
        # Block 1
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
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

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(9, activation="softmax"),   # 9 classes: digits 1-9
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARN_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    """Save a training curve to resources/ for the README."""
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
    print("  CVdoku — Digit Classifier Training")
    print("=" * 55)

    x_train, x_val, y_train, y_val = load_and_prepare_data()

    print("[2/4] Building model...")
    model = build_model()
    model.summary()

    augmenter = build_augmenter()

    callbacks = [
        # Stop early if val_loss stops improving
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        # Reduce LR when val_loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    print(f"[3/4] Training for up to {NUM_EPOCHS} epochs...")
    history = model.fit(
        augmenter.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=NUM_EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    val_acc = max(history.history["val_accuracy"])
    print(f"\n    Best val accuracy: {val_acc * 100:.2f}%")

    print(f"[4/4] Saving model → {MODEL_PATH}")
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