"""CVdoku — Cell Debugger v3. Press SPACE to capture, Q to quit."""
import os, sys, cv2, numpy as np
from src.vision import VisionEngine
from src.classifier import DigitClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "digit_model.h5")
WARP_SIZE  = 630
CELL_SIZE  = WARP_SIZE // 9

def draw_debug_grid(cells, predictions, confidences, has_bars):
    sz, pad = 80, 4
    canvas  = np.zeros((9*(sz+pad)+pad, 9*(sz+pad)+pad, 3), dtype=np.uint8)
    for i, (cell, pred, conf, bar) in enumerate(zip(cells, predictions, confidences, has_bars)):
        r, c = i//9, i%9
        x = c*(sz+pad)+pad; y = r*(sz+pad)+pad
        disp = cv2.cvtColor(cv2.resize(cell,(sz,sz)), cv2.COLOR_GRAY2BGR)
        col  = (0,200,0) if pred>0 else (0,0,180)
        cv2.rectangle(disp,(0,0),(sz-1,sz-1),col,2)
        if pred>0:
            cv2.putText(disp,str(pred),(sz//2-8,sz//2+6),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(disp,f"{conf:.2f}",(2,sz-14),cv2.FONT_HERSHEY_SIMPLEX,0.28,(0,220,220),1,cv2.LINE_AA)
            cv2.putText(disp,f"bar:{'Y' if bar else 'N'}",(2,sz-4),cv2.FONT_HERSHEY_SIMPLEX,0.28,(180,180,0),1,cv2.LINE_AA)
        else:
            cv2.putText(disp,"---",(sz//2-14,sz//2+4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(80,80,80),1,cv2.LINE_AA)
        canvas[y:y+sz, x:x+sz] = disp
    return canvas

def main():
    if not os.path.exists(MODEL_PATH): sys.exit("[ERROR] Run train.py first")
    vision     = VisionEngine()
    classifier = DigitClassifier(MODEL_PATH, threshold=0.85)
    cap        = cv2.VideoCapture(0)
    if not cap.isOpened(): sys.exit("[ERROR] Cannot open camera")
    print("\nSPACE = capture   Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        thresh = vision.pre_process(frame)
        bc     = vision.find_board_contour(thresh)
        disp   = frame.copy()
        if bc is not None:
            vision.draw_contour(disp, bc)
            cv2.putText(disp,"Board detected — SPACE to analyse",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        else:
            cv2.putText(disp,"No board",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,200),2)
        cv2.imshow("Live",disp)
        key = cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        if key==ord(' ') and bc is not None:
            print("[Debug] Analysing...")
            warped = vision.get_warp(frame, bc)
            cells  = vision.split_boxes_adaptive(warped)
            predictions, confidences, has_bars, proc_cells = [], [], [], []
            for cell in cells:
                # Binary for display
                c = cv2.resize(cell,(96,96))
                c = cv2.GaussianBlur(c,(5,5),0)
                c = cv2.adaptiveThreshold(c,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,5)
                proc_cells.append(c)
                # Has horizontal bar?
                bar = classifier._has_top_horizontal_bar(cell)
                has_bars.append(bar)
                # Predict via classifier (includes _post_process)
                processed = classifier.preprocess_cell(cell)
                if processed is None:
                    predictions.append(0); confidences.append(0.0)
                else:
                    probs = classifier.model.predict(processed, verbose=0)[0]
                    conf  = float(np.max(probs))
                    digit = int(np.argmax(probs))+1
                    confidences.append(conf)
                    if conf < classifier.threshold:
                        predictions.append(0)
                    else:
                        predictions.append(classifier._post_process(digit, cell))

            cv2.imshow("Cell Analysis", draw_debug_grid(proc_cells, predictions, confidences, has_bars))

            print("\n  Detected board:")
            print("  ┌───────┬───────┬───────┐")
            for r in range(9):
                if r in (3,6): print("  ├───────┼───────┼───────┤")
                s = "  │"
                for c in range(9):
                    v = predictions[r*9+c]
                    s += f" {v if v>0 else '.'}"
                    if c in (2,5): s += " │"
                print(s+" │")
            print("  └───────┴───────┴───────┘")
            filled = sum(1 for p in predictions if p>0)
            print(f"\n  Detected {filled}/81 digits")

            expected = [
                [5,9,3,7,8,0,0,0,0],[6,1,0,3,9,0,0,7,0],[0,0,8,4,0,5,6,3,0],
                [9,0,0,0,0,7,0,1,0],[0,4,0,0,0,1,0,9,6],[0,8,6,0,0,0,0,2,0],
                [4,0,0,0,7,0,0,0,3],[0,0,9,0,6,0,0,0,4],[0,0,5,2,4,8,9,6,1],
            ]
            errors = []
            for r in range(9):
                for c in range(9):
                    exp=expected[r][c]; got=predictions[r*9+c]
                    if exp!=0 and got!=0 and exp!=got:
                        errors.append(f"  Row{r+1} Col{c+1}: expected {exp}, got {got}")
                    elif exp!=0 and got==0:
                        errors.append(f"  Row{r+1} Col{c+1}: expected {exp}, got BLANK")
            if errors:
                print(f"\n  Errors ({len(errors)}):")
                for e in errors: print(e)
            else: 
                print("\n  ✓ All detected digits match expected!")

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__": main()