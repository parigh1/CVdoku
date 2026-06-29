"""
CVdoku — Digit Classifier v4
=============================
Key change: 1 vs 7 is now solved by detecting the horizontal bar at the
top of a 7. A 1 has NO horizontal bar. This is far more reliable than
aspect ratio on preprocessed binary cells.
"""

import numpy as np
import cv2
from tensorflow import keras


class DigitClassifier:

    IMG_SIZE = 28

    def __init__(self, model_path: str, threshold: float = 0.85):
        print(f"[Classifier] Loading model from {model_path}")
        self.model     = keras.models.load_model(model_path)
        self.threshold = threshold
        self.debug     = False
        print(f"[Classifier] Ready — confidence threshold: {threshold}")

    # ── Core preprocessing ─────────────────────────────────────────────────────

    def _binarize(self, cell_img: np.ndarray) -> np.ndarray:
        """Convert raw grayscale cell to clean binary at 96×96."""
        cell = cv2.resize(cell_img, (96, 96))
        cell = cv2.GaussianBlur(cell, (5, 5), 0)
        cell = cv2.adaptiveThreshold(
            cell, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=5
        )
        return cell

    def _is_blank(self, binary: np.ndarray) -> bool:
        """True if cell has too few white pixels to contain a digit."""
        density = np.count_nonzero(binary) / binary.size
        if density < 0.012:
            return True
        # Also check: if the only contour is a thin horizontal line (grid artifact)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return True
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 60:
            return True
        x, y, w, h = cv2.boundingRect(largest)
        if w / max(h, 1) > 5.0:   # Pure horizontal line = grid artifact
            return True

        # Solidity check — catches noise/glare streaks that AREN'T flat
        # enough to trip the w/h>5 check above (e.g. a diagonal screen
        # reflection), but are still much sparser within their own
        # bounding box than any real digit stroke. Measured empirically
        # across all 9 digits in 6 sans-serif fonts: the thinnest real
        # digit (a printed '7') never drops below ~0.20 solidity, while
        # simulated noise/glare streaks measured ~0.08-0.10. 0.15 sits in
        # the gap between them.
        solidity = cv2.contourArea(largest) / max(w * h, 1)
        if solidity < 0.15:
            return True

        return False

    def _center_digit(self, binary: np.ndarray):
        """Center main digit contour on a 28×28 canvas (MNIST-style)."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 60:
            return None
        x, y, w, h    = cv2.boundingRect(largest)
        crop          = binary[y:y+h, x:x+w]
        resized       = cv2.resize(crop, (20, 20))
        canvas        = np.zeros((28, 28), dtype=np.uint8)
        canvas[4:24, 4:24] = resized
        return canvas

    def preprocess_cell(self, cell_img: np.ndarray):
        """Full preprocessing pipeline. Returns (1,28,28,1) array or None if blank."""
        binary = self._binarize(cell_img)
        if self._is_blank(binary):
            return None
        centered = self._center_digit(binary)
        if centered is None:
            return None
        out = centered.astype("float32") / 255.0
        out = np.expand_dims(out, axis=-1)
        out = np.expand_dims(out, axis=0)
        return out

    # ── 1 vs 7 disambiguation ──────────────────────────────────────────────────

    def _has_top_horizontal_bar(self, cell_img: np.ndarray) -> bool:
        """
        Detect whether the digit has a strong horizontal bar in its top ~25%.

        A '7' has a clear horizontal stroke at the top.
        A '1' does NOT — it's just a vertical stroke (with optional serifs).

        Method:
        - Binarize the cell
        - Find the digit bounding box
        - Look at the top 25% of the bounding box
        - Count horizontal runs of white pixels
        - If there's a continuous run spanning >50% of the width → it's a 7
        """
        binary  = self._binarize(cell_img)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        if w < 4 or h < 8:
            return False

        # Crop to digit bounding box
        digit_roi = binary[y:y+h, x:x+w]

        # Look at top 30% of the digit
        top_region = digit_roi[:max(1, h//3), :]

        # For each row in the top region, find the longest horizontal run
        max_run_ratio = 0.0
        for row_pixels in top_region:
            # Count consecutive white pixels
            in_run     = False
            run_len    = 0
            max_run    = 0
            for px in row_pixels:
                if px > 0:
                    in_run  = True
                    run_len += 1
                    max_run  = max(max_run, run_len)
                else:
                    in_run  = False
                    run_len = 0
            ratio = max_run / max(w, 1)
            max_run_ratio = max(max_run_ratio, ratio)

        # A horizontal bar spanning >45% of width = definite 7
        return max_run_ratio > 0.45

    # ── Prediction ─────────────────────────────────────────────────────────────

    def _post_process(self, digit: int, cell_img: np.ndarray) -> int:
        """Apply geometric rules after CNN prediction."""
        if digit == 7:
            if not self._has_top_horizontal_bar(cell_img):
                if self.debug:
                    print("    1v7: no horizontal bar → corrected 7 to 1")
                return 1
        return digit

    def predict_cell(self, cell_img: np.ndarray) -> int:
        """Predict digit in a single cell. Returns 0 if blank."""
        processed = self.preprocess_cell(cell_img)
        if processed is None:
            return 0
        probs      = self.model.predict(processed, verbose=0)[0]
        confidence = float(np.max(probs))
        digit      = int(np.argmax(probs)) + 1
        if confidence < self.threshold:
            return 0
        return self._post_process(digit, cell_img)

    def predict(self, cells: list) -> list:
        """Predict all 81 cells. Returns list of 81 ints (0=blank, 1-9)."""
        if len(cells) != 81:
            raise ValueError(f"Expected 81 cells, got {len(cells)}")

        results    = [None] * 81
        valid_imgs = []
        valid_idx  = []
        raw_cells  = []

        # Pass 1: blank detection
        for i, cell in enumerate(cells):
            processed = self.preprocess_cell(cell)
            if processed is None:
                results[i] = 0
            else:
                valid_imgs.append(processed)
                valid_idx.append(i)
                raw_cells.append(cell)

        # Pass 2: batch CNN
        if valid_imgs:
            batch     = np.concatenate(valid_imgs, axis=0)
            all_probs = self.model.predict(batch, verbose=0)

            for j, idx in enumerate(valid_idx):
                probs      = all_probs[j]
                confidence = float(np.max(probs))
                digit      = int(np.argmax(probs)) + 1
                if confidence < self.threshold:
                    results[idx] = 0
                else:
                    results[idx] = self._post_process(digit, raw_cells[j])

        return results

    def predict_with_solidity(self, cells: list) -> tuple:
        """
        Like predict(), but also returns a parallel list of "solidity"
        scores (contour_area / bounding_box_area) for filled cells —
        None for cells predicted blank.

        Why this exists: CNN confidence can't tell a genuine thin digit
        apart from a thin noise/glare streak that narrowly cleared the
        blank-detection gate — both can score 1.00. Solidity is a
        different, independent signal: among cells the model says are
        filled, the ones with the LOWEST solidity are the ones that most
        narrowly cleared _is_blank()'s gate and are the best candidates
        for "this is probably actually a misread blank" — exactly what
        the solver's recovery logic needs when a phantom digit (not a
        digit-confusion, an extra digit where there should be none)
        makes the board unsolvable.

        Returns:
            (digits, solidities) — both length-81 lists.
        """
        if len(cells) != 81:
            raise ValueError(f"Expected 81 cells, got {len(cells)}")

        digits     = [0] * 81
        solidities = [None] * 81
        valid_imgs = []
        valid_idx  = []
        raw_cells  = []

        for i, cell in enumerate(cells):
            binary = self._binarize(cell)
            if self._is_blank(binary):
                continue
            centered = self._center_digit(binary)
            if centered is None:
                continue

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            solidities[i] = cv2.contourArea(largest) / max(w * h, 1)

            out = centered.astype("float32") / 255.0
            out = np.expand_dims(out, axis=-1)
            out = np.expand_dims(out, axis=0)
            valid_imgs.append(out)
            valid_idx.append(i)
            raw_cells.append(cell)

        if valid_imgs:
            batch     = np.concatenate(valid_imgs, axis=0)
            all_probs = self.model.predict(batch, verbose=0)

            for j, idx in enumerate(valid_idx):
                probs      = all_probs[j]
                confidence = float(np.max(probs))
                digit      = int(np.argmax(probs)) + 1
                if confidence < self.threshold:
                    digits[idx]     = 0
                    solidities[idx] = None
                else:
                    digits[idx] = self._post_process(digit, raw_cells[j])

        return digits, solidities

    def set_threshold(self, threshold: float):
        self.threshold = threshold
        print(f"[Classifier] Threshold updated to {threshold}")