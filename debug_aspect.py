"""Quick script to measure aspect ratios of specific cells."""
import os, sys, cv2, numpy as np
from src.vision import VisionEngine
from src.classifier import DigitClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "digit_model.h5")
WARP_SIZE  = 630
CELL_SIZE  = WARP_SIZE // 9

vision     = VisionEngine()
classifier = DigitClassifier(MODEL_PATH, threshold=0.85)
cap        = cv2.VideoCapture(0)

print("Press SPACE to capture aspect ratios of all cells. Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    thresh = vision.pre_process(frame)
    bc     = vision.find_board_contour(thresh)
    disp   = frame.copy()
    if bc is not None:
        vision.draw_contour(disp, bc)
    cv2.imshow("Live", disp)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord(' ') and bc is not None:
        warped = vision.get_warp(frame, bc)
        cells  = vision.split_boxes(warped)

        # Problem cells: (1,1), (3,7), (4,5), (8,8) — 0-indexed
        problem = [(1,1),(3,7),(4,5),(8,8)]
        print("\nAspect ratios for 1→7 confused cells:")
        for r,c in problem:
            cell = cells[r*9+c]
            asp  = classifier._get_digit_aspect(cell)

            # Also get raw CNN probs
            proc = classifier.preprocess_cell(cell)
            if proc is not None:
                probs = classifier.model.predict(proc, verbose=0)[0]
                top2  = np.argsort(probs)[-3:][::-1]
                prob_str = " | ".join([f"{i+1}:{probs[i]:.3f}" for i in top2])
            else:
                prob_str = "BLANK"

            print(f"  Row{r+1} Col{c+1}: aspect={asp:.3f}  CNN top3=[{prob_str}]")
        break

cap.release()
cv2.destroyAllWindows()