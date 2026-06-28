"""
CVdoku — AR Overlay Engine
===========================
Projects the solved Sudoku digits back onto the original camera frame
using inverse perspective warping.

How it works:
    1. Create a blank WARP_SIZE × WARP_SIZE canvas
    2. Draw each solved digit into its cell position on the canvas
    3. Inverse-warp the canvas back into the original camera perspective
    4. Create a mask from the warped canvas (non-zero pixels)
    5. Composite the warped digits onto the live frame

The result: solved digits appear to "sit" on the physical puzzle,
correctly skewed to match whatever angle the camera is viewing it from.
"""

import cv2
import numpy as np

WARP_SIZE = 630
CELL_SIZE = WARP_SIZE // 9   # 70px per cell

# Colour palette (BGR)
COLOUR_SOLVED  = (0, 230, 0)     # Bright green  — digits we filled in
COLOUR_GIVEN   = (200, 200, 200) # Light grey    — digits already in puzzle
COLOUR_OUTLINE = (0, 0, 0)       # Black outline  — improves readability


class OverlayEngine:

    def __init__(self):
        # Font settings
        self.font       = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.2
        self.thickness  = 2

    # ── Public API ─────────────────────────────────────────────────────────────

    def draw(
        self,
        frame:        np.ndarray,
        original_grid: list,
        solved_grid:  list,
        inv_M:        np.ndarray,
    ) -> np.ndarray:
        """
        Composite the AR solution overlay onto the live camera frame.

        Args:
            frame         : Original BGR camera frame (not modified).
            original_grid : 9×9 grid before solving (0 = blank, 1-9 = given).
            solved_grid   : 9×9 grid after solving (all cells filled).
            inv_M         : Inverse perspective matrix from VisionEngine.

        Returns:
            New BGR frame with solved digits overlaid on the puzzle.
        """
        # Step 1 — render digits onto a flat WARP_SIZE canvas
        canvas = self._build_canvas(original_grid, solved_grid)

        # Step 2 — warp canvas back into the original camera perspective
        h, w   = frame.shape[:2]
        warped = cv2.warpPerspective(canvas, inv_M, (w, h))

        # Step 3 — composite: only paint pixels where we drew digits
        mask        = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask     = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask_3ch    = cv2.merge([mask, mask, mask])

        output      = frame.copy()
        bg          = cv2.bitwise_and(output,     cv2.bitwise_not(mask_3ch))
        fg          = cv2.bitwise_and(warped,     mask_3ch)
        output      = cv2.add(bg, fg)

        return output

    def draw_status(
        self,
        frame:  np.ndarray,
        state:  str,
        fps:    float = 0.0,
        msg:    str   = "",
    ) -> None:
        """
        Draw HUD elements on the frame (in-place):
        - Top-left: FPS counter
        - Top-right: state label (SCANNING / SOLVED / ERROR)
        - Bottom-left: optional message

        Args:
            frame : BGR frame to draw on.
            state : One of "SCANNING", "SOLVED", "ERROR", "IDLE".
            fps   : Current frames per second.
            msg   : Optional extra message shown at bottom-left.
        """
        h, w = frame.shape[:2]

        state_colours = {
            "IDLE":     (180, 180, 180),
            "SCANNING": (0,   200, 255),
            "SOLVED":   (0,   230, 0  ),
            "ERROR":    (0,   0,   230),
        }
        colour = state_colours.get(state, (255, 255, 255))

        # FPS — top left
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (12, 30), self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )

        # State badge — top right
        label       = f"[ {state} ]"
        (tw, th), _ = cv2.getTextSize(label, self.font, 0.7, 2)
        cv2.putText(
            frame, label,
            (w - tw - 12, 30), self.font, 0.7, colour, 2, cv2.LINE_AA
        )

        # Controls hint — bottom left
        hints = "S: scan   R: reset   Q: quit"
        cv2.putText(
            frame, hints,
            (12, h - 12), self.font, 0.5, (180, 180, 180), 1, cv2.LINE_AA
        )

        # Optional extra message — bottom centre
        if msg:
            (mw, _), _ = cv2.getTextSize(msg, self.font, 0.55, 1)
            cv2.putText(
                frame, msg,
                ((w - mw) // 2, h - 12),
                self.font, 0.55, (0, 200, 255), 1, cv2.LINE_AA
            )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_canvas(self, original_grid: list, solved_grid: list) -> np.ndarray:
        """
        Render all 81 digits onto a WARP_SIZE × WARP_SIZE black canvas.

        - Cells that were blank in original_grid and filled in solved_grid
          are drawn in green (the AR magic).
        - Cells that were already given are skipped (not drawn) — we don't
          want to paint over digits that are physically printed on the puzzle.
        """
        canvas = np.zeros((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8)

        for row in range(9):
            for col in range(9):
                original_val = original_grid[row][col]
                solved_val   = solved_grid[row][col]

                # Skip given cells — they're already on the physical puzzle
                if original_val != 0:
                    continue

                # Skip if solver didn't fill this cell (shouldn't happen)
                if solved_val == 0:
                    continue

                self._draw_digit(canvas, row, col, solved_val, COLOUR_SOLVED)

        return canvas

    def _draw_digit(
        self,
        canvas: np.ndarray,
        row:    int,
        col:    int,
        digit:  int,
        colour: tuple,
    ) -> None:
        """
        Draw a single digit centred within its cell on the canvas.
        A thin black outline is drawn first for readability over any background.
        """
        text = str(digit)

        # Cell centre pixel coordinates on the canvas
        cx = col * CELL_SIZE + CELL_SIZE // 2
        cy = row * CELL_SIZE + CELL_SIZE // 2

        (tw, th), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.thickness
        )

        # Top-left of text so it appears centred in the cell
        tx = cx - tw // 2
        ty = cy + th // 2

        # Black outline first (draw with slightly larger thickness)
        cv2.putText(
            canvas, text, (tx, ty),
            self.font, self.font_scale,
            COLOUR_OUTLINE, self.thickness + 2, cv2.LINE_AA
        )

        # Coloured digit on top
        cv2.putText(
            canvas, text, (tx, ty),
            self.font, self.font_scale,
            colour, self.thickness, cv2.LINE_AA
        )