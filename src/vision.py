import cv2
import numpy as np


class VisionEngine:
    def __init__(self):
        self.width = 450
        self.height = 450

    def pre_process(self, img):
        """Converts to grayscale, blurs, and thresholds."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        return thresh

    def find_board_contour(self, img_thresh):
        """Finds the largest 4-sided polygon."""
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    return approx
        return None

    def _reorder_corners(self, my_points):
        """
        Private Helper: Reorders points to [Top-Left, Top-Right, Bottom-Left, Bottom-Right]
        Logic:
        - Sum(x,y): TL is min, BR is max
        - Diff(y-x): TR is min, BL is max
        """
        my_points = my_points.reshape((4, 2))
        my_points_new = np.zeros((4, 1, 2), np.int32)

        add = my_points.sum(1)
        my_points_new[0] = my_points[np.argmin(add)]  # TL
        my_points_new[3] = my_points[np.argmax(add)]  # BR

        diff = np.diff(my_points, axis=1)
        my_points_new[1] = my_points[np.argmin(diff)]  # TR
        my_points_new[2] = my_points[np.argmax(diff)]  # BL

        return my_points_new

    def get_warp(self, img, contour):
        """
        Warps the perspective of the found contour to a flat square.
        """
        pts1 = self._reorder_corners(contour)

        # Define the target mapping points (a perfect square)
        pts2 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])

        # Get and Apply the Matrix
        matrix = cv2.getPerspectiveTransform(np.float32(pts1), pts2)
        img_warped = cv2.warpPerspective(img, matrix, (self.width, self.height))

        return img_warped

    def draw_contours(self, img, contour):
        if contour is not None:
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 4)
        return img

    def split_boxes(self, img):
        rows = np.vsplit(img, 9)
        boxes = []
        for r in rows:
            cols = np.hsplit(r, 9)
            for box in cols:
                # --- CRITICAL FIX: CROP BORDERS ---
                # Remove 5 pixels from every side to cut out grid lines
                box = box[5:45, 5:45]
                # ----------------------------------
                boxes.append(box)
        return boxes

    def display_numbers(self, img, numbers, color=(0, 255, 0)):
        """
        Overlay the detected numbers onto the warped image for visual verification.
        """
        secW = int(img.shape[1] / 9)
        secH = int(img.shape[0] / 9)
        for x in range(0, 9):
            for y in range(0, 9):
                if numbers[(y * 9) + x] != 0:
                    cv2.putText(img, str(numbers[(y * 9) + x]),
                                (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv2.LINE_AA)
        return img