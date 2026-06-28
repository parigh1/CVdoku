"""
CVdoku — Vision Engine
=======================
Handles everything between the raw camera frame and 81 cell images:
  1. Pre-process  → grayscale, blur, adaptive threshold
  2. Find board   → largest quadrilateral contour in frame
  3. Warp         → perspective-correct the board to a square
  4. Split        → slice the warped square into 81 cells
  5. Draw         → highlight the detected board on the live feed

Design notes:
- All internal processing is done on grayscale images.
- The warp output is always WARP_SIZE × WARP_SIZE pixels.
- Cell images are returned as grayscale crops ready for the classifier.
"""

import cv2
import numpy as np

WARP_SIZE = 630   # Output size of the perspective-corrected board (pixels)
CELL_SIZE = WARP_SIZE // 9   # 70px per cell — more resolution per digit


class VisionEngine:

    # ── Pre-processing ─────────────────────────────────────────────────────────

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        """
        Convert raw BGR frame to a binary image optimised for contour detection.

        Pipeline:
            BGR → Grayscale → Gaussian blur → Adaptive threshold → Dilate
        """
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresh  = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
        # Small dilation closes tiny gaps in the grid lines
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        return dilated

    # ── Board detection ────────────────────────────────────────────────────────

    def find_board_contour(self, thresh: np.ndarray) -> np.ndarray | None:
        """
        Find the largest quadrilateral contour — that's the Sudoku grid.

        Returns:
            numpy array of shape (4, 2) with the four corner points,
            or None if no suitable contour is found.
        """
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Sort by area descending — the board is the biggest thing in frame
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:   # Only check the 5 largest
            area = cv2.contourArea(contour)
            if area < 10000:           # Too small — not a Sudoku board
                continue

            perimeter = cv2.arcLength(contour, closed=True)
            approx    = cv2.approxPolyDP(contour, 0.02 * perimeter, closed=True)

            if len(approx) == 4:       # Quadrilateral found
                return approx.reshape(4, 2).astype(np.float32)

        return None

    def order_corners(self, pts: np.ndarray) -> np.ndarray:
        """
        Order four corner points as: [top-left, top-right, bottom-right, bottom-left].
        This order is required by getPerspectiveTransform.
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left     (smallest x+y)
        rect[2] = pts[np.argmax(s)]   # bottom-right (largest x+y)

        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]   # top-right    (smallest y-x)
        rect[3] = pts[np.argmax(d)]   # bottom-left  (largest y-x)

        return rect

    # ── Perspective warp ───────────────────────────────────────────────────────

    def get_warp(self, img: np.ndarray, board_contour: np.ndarray) -> np.ndarray:
        """
        Apply a perspective warp to extract and straighten the Sudoku board.

        Args:
            img           : Original BGR frame.
            board_contour : Four corner points from find_board_contour().

        Returns:
            Grayscale warped image of shape (WARP_SIZE, WARP_SIZE).
        """
        src = self.order_corners(board_contour)

        dst = np.array([
            [0,           0          ],
            [WARP_SIZE-1, 0          ],
            [WARP_SIZE-1, WARP_SIZE-1],
            [0,           WARP_SIZE-1],
        ], dtype=np.float32)

        M       = cv2.getPerspectiveTransform(src, dst)
        warped  = cv2.warpPerspective(img, M, (WARP_SIZE, WARP_SIZE))
        gray    = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Store the transform matrix — needed by the overlay module
        self.last_M   = M
        self.last_src = src

        return gray

    def get_inverse_warp_matrix(self) -> np.ndarray | None:
        """
        Return the inverse perspective matrix for AR overlay projection.
        Only valid after get_warp() has been called.
        """
        if not hasattr(self, "last_M"):
            return None
        return cv2.invert(self.last_M)[1]

    # ── Cell splitting ─────────────────────────────────────────────────────────

    def split_boxes(self, warped_gray: np.ndarray) -> list:
        """
        Slice the warped board into 81 individual cell images.

        Each cell is centre-cropped by 4px on each side to remove grid lines,
        then returned as a grayscale numpy array.

        Returns:
            List of 81 numpy arrays, row-major order (left-to-right, top-to-bottom).
        """
        cells  = []
        margin = 2   # pixels to trim from each edge (removes grid lines)

        for row in range(9):
            for col in range(9):
                y1 = row * CELL_SIZE + margin
                y2 = (row + 1) * CELL_SIZE - margin
                x1 = col * CELL_SIZE + margin
                x2 = (col + 1) * CELL_SIZE - margin

                cell = warped_gray[y1:y2, x1:x2]
                cells.append(cell)

        return cells

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def draw_contour(self, img: np.ndarray, board_contour: np.ndarray) -> None:
        """
        Draw the detected board boundary on the live frame (in-place).
        Green polygon + corner dots so the user can see what's been detected.
        """
        pts = board_contour.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        # Corner markers
        for pt in board_contour:
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(img, (cx, cy), radius=8, color=(0, 255, 0), thickness=-1)

    def draw_grid_overlay(self, img: np.ndarray, board_contour: np.ndarray) -> None:
        """
        Draw a faint 9×9 grid overlay on the detected board area.
        Helps the user align the puzzle correctly before scanning.
        """
        src    = self.order_corners(board_contour)
        dst    = np.array([
            [0,           0          ],
            [WARP_SIZE-1, 0          ],
            [WARP_SIZE-1, WARP_SIZE-1],
            [0,           WARP_SIZE-1],
        ], dtype=np.float32)
        M_inv  = cv2.getPerspectiveTransform(dst, src)

        for i in range(1, 9):
            # Vertical lines
            p1 = np.array([[i * CELL_SIZE, 0]], dtype=np.float32).reshape(-1, 1, 2)
            p2 = np.array([[i * CELL_SIZE, WARP_SIZE]], dtype=np.float32).reshape(-1, 1, 2)
            # Horizontal lines
            p3 = np.array([[0, i * CELL_SIZE]], dtype=np.float32).reshape(-1, 1, 2)
            p4 = np.array([[WARP_SIZE, i * CELL_SIZE]], dtype=np.float32).reshape(-1, 1, 2)

            for pa, pb in [(p1, p2), (p3, p4)]:
                ta = cv2.perspectiveTransform(pa, M_inv)[0][0].astype(int)
                tb = cv2.perspectiveTransform(pb, M_inv)[0][0].astype(int)
                cv2.line(img, tuple(ta), tuple(tb), (0, 255, 0), 1, cv2.LINE_AA)