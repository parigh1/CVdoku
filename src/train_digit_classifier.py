import numpy as np
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 10
VALIDATION_SPLIT = 0.2
input_shape = (32, 32, 1)  # We use 1 channel (Grayscale) for efficiency
model_save_path = '../models/digit_model.h5'


def pre_process(img):
    """
    Resize MNIST (28x28) to our target size (32x32) and normalize.
    """
    img = cv2.resize(img, (32, 32))
    img = img / 255.0  # Normalize to 0-1
    img = img.reshape(32, 32, 1)  # Add channel dimension
    return img


def create_model():
    """
    Builds a CNN architecture.
    """
    model = Sequential()

    # Layer 1: Extract features (edges, curves)
    model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2: Extract deeper features (shapes)
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Regularization to prevent overfitting
    model.add(Dropout(0.5))
    model.add(Flatten())

    # Classification
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))  # Output 0-9

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # 1. Load Data
    print("Downloading MNIST Data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 2. Preprocessing
    print("Preprocessing images (Resizing to 32x32)...")
    # Map the pre_process function to all images
    # We use a list comprehension for clarity
    x_train = np.array([pre_process(img) for img in x_train])
    x_test = np.array([pre_process(img) for img in x_test])

    # One-Hot Encode labels (e.g., '5' -> [0,0,0,0,0,1,0,0,0,0])
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # 3. Train
    print("Starting Training...")
    model = create_model()
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        verbose=1)

    # 4. Save
    if not os.path.exists('../models'):
        os.makedirs('../models')
    model.save(model_save_path)
    print(f"Success! Model saved to {model_save_path}")