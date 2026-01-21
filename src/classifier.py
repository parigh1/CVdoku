import numpy as np
import cv2
from tensorflow.keras.models import load_model


class DigitClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def pre_process_smart(self, img):
        """
        Refined preprocessing pipeline:
        1. Crop standard borders.
        2. Find the digit contour.
        3. Center the digit to match MNIST training data format.
        """
        # 1. Initial aggressive crop to remove grid lines (50x50 -> 40x40)
        # The sudoku lines are usually on the very edge
        img = img[5:45, 5:45]

        # 2. Convert to Grayscale & Invert (Black background, White digit)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)

        # 3. Threshold to isolate the digit
        _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        # 4. Find the largest blob (the digit)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- GATEKEEPER: EMPTY CHECK ---
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        # Get bounding box to check aspect ratio
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(h) / w

        # LOGIC FIX FOR "1":
        # - If area is very small (<50), usually noise.
        # - BUT if it is small AND tall (aspect_ratio > 2.5), it is likely a '1', so keep it.
        if area < 50 and aspect_ratio < 2.5:
            return None

        # 6. Center the digit (MNIST style)
        # Create a black square canvas (32x32)
        digit_canvas = np.zeros((32, 32), dtype=np.uint8)

        # Crop the digit from the original image
        roi = thresh[y:y + h, x:x + w]

        # Resize it to fit in the center (keep aspect ratio)
        # We target a height of 20px (leaving buffer space)
        scale = 20.0 / max(w, h)
        nh = int(h * scale)
        nw = int(w * scale)
        roi_resized = cv2.resize(roi, (nw, nh))

        # Paste it into the center of the 32x32 canvas
        y_off = (32 - nh) // 2
        x_off = (32 - nw) // 2
        digit_canvas[y_off:y_off + nh, x_off:x_off + nw] = roi_resized

        # 7. Final Norm & Reshape for Model
        digit_final = digit_canvas / 255.0
        digit_final = digit_final.reshape(32, 32, 1)

        return digit_final

    def predict(self, boxes):
        results = []
        batch_images = []
        valid_indices = []

        # 1. Preprocess all boxes
        for i, box in enumerate(boxes):
            processed_img = self.pre_process_smart(box)

            if processed_img is None:
                results.append(0)  # Mark as Empty immediately
            else:
                results.append(-1)  # Placeholder for AI prediction
                batch_images.append(processed_img)
                valid_indices.append(i)

        # 2. Batch Predict (Only valid digits)
        if len(batch_images) > 0:
            batch_images = np.array(batch_images)
            predictions = self.model.predict(batch_images, verbose=0)

            for idx, pred in zip(valid_indices, predictions):
                class_index = np.argmax(pred)
                prob_value = np.amax(pred)

                # Confidence Threshold
                if prob_value > 0.6:
                    results[idx] = class_index
                else:
                    results[idx] = 0

        return results