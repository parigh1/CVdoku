import numpy as np
import cv2
from tensorflow import keras


class DigitClassifier:
    """
    Wraps the trained Keras model with confidence-based blank detection.

    Args:
        model_path  : Path to the saved .h5 model file.
        threshold   : Minimum softmax confidence to accept a prediction.
                      Below this → cell treated as blank (returns 0).
                      0.85 works well for clean printed Sudoku grids.
    """

    IMG_SIZE = 28   # Must match the size the model was trained on

    def __init__(self, model_path: str, threshold: float = 0.85):
        print(f"[Classifier] Loading model from {model_path}")
        self.model     = keras.models.load_model(model_path)
        self.threshold = threshold
        print(f"[Classifier] Ready — confidence threshold: {threshold}")

    def preprocess_cell(self, cell_img: np.ndarray) -> np.ndarray:
        """
        Prepare a raw cell crop for the CNN.

        Steps:
        1. Resize to 28×28
        2. Normalise to [0, 1]
        3. Add batch + channel dims → (1, 28, 28, 1)
        """
        cell = cv2.resize(cell_img, (self.IMG_SIZE, self.IMG_SIZE))
        cell = cell.astype("float32") / 255.0
        cell = np.expand_dims(cell, axis=-1)   # channel dim
        cell = np.expand_dims(cell, axis=0)    # batch dim
        return cell

    def predict_cell(self, cell_img: np.ndarray) -> int:
        """
        Predict the digit in a single cell image.

        Returns:
            int: 1-9 if a digit is detected with sufficient confidence,
                 0 if the cell is blank or confidence is too low.
        """
        processed = self.preprocess_cell(cell_img)
        probs     = self.model.predict(processed, verbose=0)[0]  # shape (9,)

        confidence = float(np.max(probs))
        predicted  = int(np.argmax(probs)) + 1  # shift back: 0→1, ..., 8→9

        if confidence < self.threshold:
            return 0   # blank cell

        return predicted

    def predict(self, cells: list) -> list:
        """
        Predict digits for a list of 81 cell images (the full board).

        Args:
            cells: List of 81 grayscale numpy arrays (cell crops).

        Returns:
            List of 81 ints, each 0 (blank) or 1-9.
        """
        if len(cells) != 81:
            raise ValueError(f"Expected 81 cells, got {len(cells)}")

        # Batch all 81 cells into one forward pass — much faster than one at a time
        batch = np.stack([
            cv2.resize(c, (self.IMG_SIZE, self.IMG_SIZE)).astype("float32") / 255.0
            for c in cells
        ])
        batch = np.expand_dims(batch, axis=-1)  # → (81, 28, 28, 1)

        all_probs = self.model.predict(batch, verbose=0)  # → (81, 9)

        results = []
        for probs in all_probs:
            confidence = float(np.max(probs))
            digit      = int(np.argmax(probs)) + 1

            results.append(digit if confidence >= self.threshold else 0)

        return results

    def set_threshold(self, threshold: float):
        """Adjust the confidence threshold at runtime if needed."""
        self.threshold = threshold
        print(f"[Classifier] Threshold updated to {threshold}")