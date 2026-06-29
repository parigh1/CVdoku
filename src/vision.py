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

    def compute_perspective_matrix(self, board_contour: np.ndarray) -> np.ndarray:
        """
        Compute (and cache) just the forward perspective matrix from a
        board contour, without warping any image.

        Used during live AR tracking after solving: every frame we need a
        fresh inv_M to keep the overlay aligned with the puzzle's CURRENT
        position, but the digit values are already locked in, so there's
        no need to pay for a full warpPerspective + grayscale conversion
        just to throw the image away.
        """
        src = self.order_corners(board_contour)
        dst = np.array([
            [0,           0          ],
            [WARP_SIZE-1, 0          ],
            [WARP_SIZE-1, WARP_SIZE-1],
            [0,           WARP_SIZE-1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        self.last_M   = M
        self.last_src = src
        return M

    def get_warp(self, img: np.ndarray, board_contour: np.ndarray) -> np.ndarray:
        """
        Apply a perspective warp to extract and straighten the Sudoku board.

        Args:
            img           : Original BGR frame.
            board_contour : Four corner points from find_board_contour().

        Returns:
            Grayscale warped image of shape (WARP_SIZE, WARP_SIZE).
        """
        M      = self.compute_perspective_matrix(board_contour)
        warped = cv2.warpPerspective(img, M, (WARP_SIZE, WARP_SIZE))
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
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

    # ── Grid-line-aware splitting ──────────────────────────────────────────────

    def _profile_peaks(self, profile: np.ndarray, board_size: int) -> list:
        """
        Find the center position of each high-intensity run in a 1D profile
        (a row-sum or column-sum of a morphologically-isolated line mask).

        Used to locate the actual pixel position of each grid line instead
        of assuming they sit at perfect multiples of CELL_SIZE.
        """
        if profile.max() == 0:
            return []

        above   = profile > (profile.max() * 0.3)
        runs    = []
        start   = None
        for i, val in enumerate(above):
            if val and start is None:
                start = i
            elif not val and start is not None:
                runs.append((start + i - 1) / 2.0)
                start = None
        if start is not None:
            runs.append((start + len(above) - 1) / 2.0)

        # Merge runs that are too close together to be distinct grid lines
        # (avoids double-counting a thick or slightly broken line).
        min_gap = board_size * 0.04
        merged  = []
        for p in runs:
            if merged and (p - merged[-1]) < min_gap:
                merged[-1] = (merged[-1] + p) / 2.0
            else:
                merged.append(p)
        return merged

    def _detect_grid_lines(self, warped_gray: np.ndarray):
        """
        Detect the 10 horizontal + 10 vertical grid lines in the warped board
        via morphological line isolation, instead of assuming they fall at
        perfect multiples of CELL_SIZE.

        Perspective warp corrects tilt, but not residual lens distortion or
        paper curl — so on a real camera frame the 9 cells are rarely
        perfectly equal after warping. Finding the real line positions makes
        cell extraction robust to that residual distortion.

        Returns:
            (h_lines, v_lines) — each a sorted list of 10 pixel positions,
            or None if confident detection failed (caller should fall back
            to uniform division).
        """
        size    = warped_gray.shape[0]
        blurred = cv2.GaussianBlur(warped_gray, (5, 5), 0)
        thresh  = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=5
        )

        # Long, thin kernels isolate only lines that span most of the board —
        # digit strokes are far shorter than this, so they get erased.
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size // 4, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size // 4))

        h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

        h_lines = self._profile_peaks(h_mask.sum(axis=1).astype(np.float64), size)
        v_lines = self._profile_peaks(v_mask.sum(axis=0).astype(np.float64), size)

        if len(h_lines) != 10 or len(v_lines) != 10:
            return None

        return sorted(h_lines), sorted(v_lines)

    def split_boxes_adaptive(self, warped_gray: np.ndarray) -> list:
        """
        Slice the warped board into 81 cells using detected grid-line
        positions, with a proportional margin (≈6% of each cell's own
        size) instead of a fixed pixel trim.

        Falls back to the uniform split_boxes() if grid lines can't be
        confidently detected (e.g. faint lines, heavy glare, motion blur) —
        this method is a strict improvement, never a regression.
        """
        lines = self._detect_grid_lines(warped_gray)
        if lines is None:
            return self.split_boxes(warped_gray)

        h_lines, v_lines = lines
        cells = []

        for row in range(9):
            y1, y2   = h_lines[row], h_lines[row + 1]
            margin_y = max(1, int((y2 - y1) * 0.06))
            for col in range(9):
                x1, x2   = v_lines[col], v_lines[col + 1]
                margin_x = max(1, int((x2 - x1) * 0.06))

                crop = warped_gray[
                    int(y1) + margin_y : int(y2) - margin_y,
                    int(x1) + margin_x : int(x2) - margin_x,
                ]
                cells.append(crop)

        return cells

    def estimate_quality(self, warped_gray: np.ndarray, board_contour: np.ndarray) -> float:
        """
        Cheap per-frame quality score, used to pick the BEST frame(s) out
        of a capture window instead of requiring an unbroken run of N good
        frames in a row.

        Combines:
          - Sharpness (Laplacian variance) — drops sharply under motion
            blur (verified empirically: ~4400 sharp -> ~3 under heavy blur).
          - Corner-angle regularity — penalises a contour that barely
            qualifies as a quadrilateral (partial occlusion, glare wiping
            out an edge, a corner clipped at the frame boundary).

        Higher is better. Cheap enough to compute every frame — no CNN
        involved.
        """
        sharpness = cv2.Laplacian(warped_gray, cv2.CV_64F).var()

        src    = self.order_corners(board_contour)
        angles = []
        for i in range(4):
            p_prev = src[(i - 1) % 4]
            p_curr = src[i]
            p_next = src[(i + 1) % 4]
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles.append(angle)
        angle_penalty = sum(abs(a - 90) for a in angles)

        return sharpness - (angle_penalty * 2.0)

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