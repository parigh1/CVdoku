"""
CVdoku — Real-Time AR Sudoku Solver
=====================================
Entry point. Orchestrates the full pipeline:

    Camera → VisionEngine → DigitClassifier → SudokuSolver → OverlayEngine

Usage:
    python main.py
    python main.py --camera 1
    python main.py --model models/digit_model.h5 --width 1280 --height 720

Controls:
    S — Scan and solve the puzzle currently in frame
    R — Reset (clear the current solution)
    Q — Quit
"""

import os
import sys
import time
import argparse
import copy

import cv2
import numpy as np

from src.vision     import VisionEngine
from src.classifier import DigitClassifier
from src.solver     import SudokuSolver
from src.overlay    import OverlayEngine


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CVdoku — Real-Time AR Sudoku Solver"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--model", type=str,
        default=os.path.join(os.path.dirname(__file__), "models", "digit_model.h5"),
        help="Path to trained digit model (default: models/digit_model.h5)"
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Camera frame width (default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Camera frame height (default: 720)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Digit confidence threshold 0-1 (default: 0.85)"
    )
    parser.add_argument(
        "--stability-frames", type=int, default=7,
        help="Frames to vote across before locking scan (default: 7)"
    )
    return parser.parse_args()


# ── Frame stability voting ─────────────────────────────────────────────────────

class FrameVoter:
    """
    Collects digit predictions across N frames and returns a consensus grid.

    For each of the 81 cells, we keep a running vote count per digit (0-9).
    After N frames, the digit with the most votes wins each cell.
    This eliminates single-frame noise from camera shake or lighting flicker.
    """

    def __init__(self, n_frames: int = 7):
        self.n_frames  = n_frames
        self.votes     = np.zeros((9, 9, 10), dtype=np.int32)  # [row][col][digit]
        self.collected = 0

    def add(self, flat_predictions: list) -> None:
        """Add one frame's worth of predictions (list of 81 ints)."""
        grid = np.array(flat_predictions, dtype=np.int32).reshape(9, 9)
        for r in range(9):
            for c in range(9):
                self.votes[r, c, grid[r, c]] += 1
        self.collected += 1

    def is_ready(self) -> bool:
        return self.collected >= self.n_frames

    def get_result(self) -> list:
        """Return the consensus 9×9 grid as a list of lists."""
        result = np.argmax(self.votes, axis=2)   # shape (9, 9)
        return result.tolist()

    def reset(self) -> None:
        self.votes[:]  = 0
        self.collected = 0


# ── Application states ─────────────────────────────────────────────────────────

class State:
    IDLE      = "IDLE"
    VOTING    = "VOTING"     # Collecting frames for stability vote
    SOLVING   = "SOLVING"    # Running the solver (one-shot)
    SOLVED    = "SOLVED"     # Solution found, overlay active
    ERROR     = "ERROR"      # Something went wrong


# ── Main application ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Validate model path ────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"\n[ERROR] Model not found: {args.model}")
        print("  Run 'python train.py' first to generate the model.\n")
        sys.exit(1)

    # ── Initialise modules ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  CVdoku — AR Sudoku Solver")
    print("=" * 50)

    vision     = VisionEngine()
    classifier = DigitClassifier(args.model, threshold=args.threshold)
    solver     = SudokuSolver()
    overlay    = OverlayEngine()
    voter      = FrameVoter(n_frames=args.stability_frames)

    # ── Open camera ───────────────────────────────────────────────────────────
    print(f"[Camera] Opening camera index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}")
        sys.exit(1)

    print("[Camera] Ready.")
    print("\nControls: S = scan & solve   R = reset   Q = quit\n")

    # ── State variables ────────────────────────────────────────────────────────
    state         = State.IDLE
    original_grid = None    # Grid as detected (with blanks)
    solved_grid   = None    # Grid after solving
    inv_M         = None    # Inverse perspective matrix for AR
    status_msg    = ""      # Message shown on HUD
    board_contour = None    # Last detected contour

    # FPS tracking
    fps       = 0.0
    fps_timer = time.time()
    frame_count = 0

    # ── Main loop ──────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        # ── FPS ───────────────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps        = frame_count / elapsed
            frame_count = 0
            fps_timer  = time.time()

        # ── Vision — run every frame to show detection feedback ───────────────
        img_thresh    = vision.pre_process(frame)
        board_contour = vision.find_board_contour(img_thresh)

        # ── Draw detection feedback ───────────────────────────────────────────
        if board_contour is not None and state in (State.IDLE, State.VOTING):
            vision.draw_contour(frame, board_contour)
            vision.draw_grid_overlay(frame, board_contour)

        # ── Frame voting loop ─────────────────────────────────────────────────
        if state == State.VOTING:
            if board_contour is not None:
                warped_gray = vision.get_warp(frame, board_contour)
                cells       = vision.split_boxes(warped_gray)
                predictions = classifier.predict(cells)
                voter.add(predictions)

                remaining = args.stability_frames - voter.collected
                status_msg = f"Scanning... {remaining} frames left"

                if voter.is_ready():
                    state = State.SOLVING

            else:
                voter.reset()
                status_msg = "Board lost — hold steady"

        # ── Solve ─────────────────────────────────────────────────────────────
        if state == State.SOLVING:
            consensus = voter.get_result()

            solvable, reason = solver.is_solvable(consensus)

            if not solvable:
                print(f"[Solver] Cannot solve: {reason}")
                status_msg = reason
                state      = State.ERROR
            else:
                # Deep copy before solving — keep original for overlay
                original_grid = copy.deepcopy(consensus)
                inv_M         = vision.get_inverse_warp_matrix()

                print("[Solver] Solving (with conflict recovery)...")
                solved_attempt, success = solver.solve_with_recovery(consensus)

                if success:
                    solved_grid = solved_attempt
                    state       = State.SOLVED
                    status_msg  = "Solved!"
                    print("[Solver] Solution found.")
                    _print_grid(original_grid, solved_grid)
                else:
                    status_msg = "Unsolvable — rescan or improve lighting"
                    state      = State.ERROR
                    print("[Solver] No solution found even after recovery.")

        # ── AR overlay ────────────────────────────────────────────────────────
        if state == State.SOLVED and solved_grid is not None and inv_M is not None:
            frame = overlay.draw(frame, original_grid, solved_grid, inv_M)

        # ── HUD ───────────────────────────────────────────────────────────────
        display_state = state if state != State.VOTING else "SCANNING"
        overlay.draw_status(frame, display_state, fps, status_msg)

        # ── Show ──────────────────────────────────────────────────────────────
        cv2.imshow("CVdoku — AR Sudoku Solver", frame)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[CVdoku] Quitting.")
            break

        elif key == ord('s'):
            if state in (State.IDLE, State.ERROR, State.SOLVED):
                if board_contour is not None:
                    voter.reset()
                    state      = State.VOTING
                    status_msg = "Scanning..."
                    print("\n[CVdoku] Starting scan...")
                else:
                    status_msg = "No board detected — point camera at puzzle"
                    print("[CVdoku] No board in frame.")

        elif key == ord('r'):
            state         = State.IDLE
            original_grid = None
            solved_grid   = None
            inv_M         = None
            status_msg    = ""
            voter.reset()
            print("[CVdoku] Reset.")

    cap.release()
    cv2.destroyAllWindows()


# ── Debug helper ───────────────────────────────────────────────────────────────

def _print_grid(original: list, solved: list) -> None:
    """Pretty-print the solved grid to terminal. Green = filled in by solver."""
    print("\n  Solved Sudoku:")
    print("  ┌───────┬───────┬───────┐")
    for r in range(9):
        if r in (3, 6):
            print("  ├───────┼───────┼───────┤")
        row_str = "  │"
        for c in range(9):
            val = solved[r][c]
            sep = "│" if c in (2, 5) else " "
            if original[r][c] == 0:
                row_str += f" \033[32m{val}\033[0m"   # Green for solved digits
            else:
                row_str += f" {val}"
            if c < 8:
                row_str += sep if c in (2, 5) else ""
        row_str += " │"
        print(row_str)
    print("  └───────┴───────┴───────┘\n")


if __name__ == "__main__":
    main()