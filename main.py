import cv2
import os
import sys
from src.vision import VisionEngine
from src.classifier import DigitClassifier


def main():
    # --- FIX: SMART PATH FINDING ---
    # Get the folder where main.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the model
    model_path = os.path.join(base_dir, 'models', 'digit_model.h5')

    # Debug: Print where we are looking to be sure
    print(f"Looking for model at: {model_path}")

    if not os.path.exists(model_path):
        print("ERROR: File not found! Please check if 'models/digit_model.h5' exists.")
        return

    # 1. Initialize Engines
    vision = VisionEngine()
    classifier = DigitClassifier(model_path)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    print("------------------------------------------------")
    print("Press 's' to Scan the board (when Green Box is stable)")
    print("Press 'q' to Quit")
    print("------------------------------------------------")

    while True:
        success, img = cap.read()
        if not success: break

        # Vision Pipeline
        img_thresh = vision.pre_process(img)
        board_contour = vision.find_board_contour(img_thresh)

        if board_contour is not None:
            vision.draw_contours(img, board_contour)

            # --- USER TRIGGER: Press 's' to Solve ---
            if cv2.waitKey(1) & 0xFF == ord('s'):
                print("Scanning...")

                # 1. Warp
                img_warped = vision.get_warp(img, board_contour)

                # 2. Slice
                img_warped_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
                boxes = vision.split_boxes(img_warped_gray)

                # 3. Predict
                numbers = classifier.predict(boxes)

                # 4. Display Results in Terminal
                print("Detected Sudoku Grid:")
                import numpy as np
                grid = np.array(numbers).reshape(9, 9)
                print(grid)

        cv2.imshow("AR Sudoku", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            # 3. Predict
            numbers = classifier.predict(boxes)

            # --- NEW: Visual Debugging ---
            # Draw the detected numbers ON TOP of the warped board
            img_detected = img_warped.copy()
            img_detected = vision.display_numbers(img_detected, numbers, color=(255, 0, 255))
            cv2.imshow("Detected Numbers", img_detected)
            # -----------------------------

            print("Detected Sudoku Grid:")
            import numpy as np
            grid = np.array(numbers).reshape(9, 9)
            print(grid)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()