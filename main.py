import cv2
import os
import numpy as np
from src.vision import VisionEngine
from src.classifier import DigitClassifier
from src.solver import SudokuSolver  


def main():
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'digit_model.h5')

    if not os.path.exists(model_path):
        print("ERROR: File not found! Please check if 'models/digit_model.h5' exists.")
        return

    
    vision = VisionEngine()
    classifier = DigitClassifier(model_path)
    solver = SudokuSolver()  

    cap = cv2.VideoCapture(0)
    
    cap.set(3, 1280)
    cap.set(4, 720)

    print("------------------------------------------------")
    print("Press 's' to Scan & Solve")
    print("Press 'q' to Quit")
    print("------------------------------------------------")

    while True:
        success, img = cap.read()
        if not success: break

        
        img_thresh = vision.pre_process(img)
        board_contour = vision.find_board_contour(img_thresh)

        if board_contour is not None:
            vision.draw_contours(img, board_contour)

            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                print("\n SCANNING")

                
                img_warped = vision.get_warp(img, board_contour)

                
                img_warped_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
                boxes = vision.split_boxes(img_warped_gray)
                numbers = classifier.predict(boxes)

                
                grid = np.array(numbers).reshape(9, 9)
                print("Detected Grid:")
                print(grid)

                
                grid_list = grid.tolist()

                try:
                    print("Solving")
                    if solver.solve(grid_list):
                        print("\n SOLVED SUDOKU ")
                        solved_grid = np.array(grid_list)
                        print(solved_grid)
                        print("----------------------")
                    else:
                        print("Unsolvable! (Likely a detection error)")
                except Exception as e:
                    print(f"Error solving: {e}")

        cv2.imshow("AR Sudoku", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
