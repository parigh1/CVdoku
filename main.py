"""
CVdoku — Real-Time AR Sudoku Solver
=====================================
Entry point. Orchestrates the full pipeline:

    Camera → VisionEngine → DigitClassifier → SudokuSolver → OverlayEngine

Three phases, each independently testable:

  1. CAPTURE — watch a short rolling window, score every frame's quality
     CHEAPLY (sharpness + contour regularity, no CNN), keep the best few
     seen anywhere in that window. A bad moment in between doesn't reset
     anything — unlike requiring N consecutive good frames in a row.

  2. SOLVE — run the CNN + solver once, only on the few best frames.

  3. AR TRACKING — once solved, the digit values are locked in. Every
     subsequent frame just re-detects the board's CURRENT position and
     recomputes the perspective matrix (cheap — no CNN), so the overlay
     stays correctly aligned even if the camera moves after solving.

Usage:
    python main.py
    python main.py --camera 1
    python main.py --model models/digit_model.h5 --width 1280 --height 720

Controls:
    S — Start a scan (looks for the best frame(s) over the next ~2-3s)
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
        "--capture-window", type=float, default=2.5,
        help="Seconds to watch for the best frame(s) (default: 2.5)"
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of best frames to vote across (default: 3)"
    )
    return parser.parse_args()


# ── Best-frame capture ──────────────────────────────────────────────────────────

class BestFrameCapture:
    """
    Watches a short time window and keeps the best-quality frames seen,
    instead of requiring N consecutive clean frames in a row.

    Why: the old approach ran the CNN on every single frame and reset its
    entire vote tally back to zero the instant the board was lost for
    even one frame — so any hand shake or momentary blur meant starting
    over. This collects cheap quality scores (sharpness + corner
    regularity, no CNN) across the whole window, keeps only the top-K
    frames by quality, and runs the CNN just once at the end on those few
    winners. One good instant ANYWHERE in the window is enough — gaps in
    between don't reset anything.
    """

    def __init__(self, top_k: int = 3, window_seconds: float = 2.5,
                 early_exit_quality: float = 800.0):
        self.top_k              = top_k
        self.window_seconds      = window_seconds
        self.early_exit_quality = early_exit_quality
        self.reset()

    def reset(self) -> None:
        self.candidates  = []   # list of (quality, warped_gray, contour)
        self.start_time  = None

    def start(self) -> None:
        self.reset()
        self.start_time = time.time()

    def add(self, quality: float, warped_gray: np.ndarray, contour: np.ndarray) -> None:
        self.candidates.append((quality, warped_gray.copy(), contour.copy()))
        self.candidates.sort(key=lambda c: c[0], reverse=True)
        self.candidates = self.candidates[: self.top_k * 3]  # keep a little headroom

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def is_ready(self) -> bool:
        if self.start_time is None:
            return False
        # Early exit: once we already have top_k frames that are clearly
        # good (sharp, well-aligned), no need to wait out the full window.
        good_enough = sum(1 for q, _, _ in self.candidates if q >= self.early_exit_quality)
        if good_enough >= self.top_k:
            return True
        return self.elapsed() >= self.window_seconds

    def best_frames(self) -> list:
        """Top-K (warped_gray, contour) pairs by quality, best first."""
        return [(wg, ct) for _, wg, ct in self.candidates[: self.top_k]]

    def has_any(self) -> bool:
        return len(self.candidates) > 0


def vote_across_frames(vision, classifier, frames_and_contours: list) -> tuple:
    """
    Run the classifier on each of the given (warped_gray, contour) pairs
    and majority-vote per cell across them.

    Returns:
        (consensus_grid, flat_solidities)
        consensus_grid  — 9x9 nested list (matches what solver.py expects)
        flat_solidities — flat length-81 list (matches solver.py's
                          row*9+col indexing for phantom-digit recovery)
    """
    votes      = np.zeros((9, 9, 10), dtype=np.int32)
    all_preds  = []
    all_solids = []

    for warped_gray, _ in frames_and_contours:
        cells         = vision.split_boxes_adaptive(warped_gray)
        preds, solids = classifier.predict_with_solidity(cells)
        all_preds.append(preds)
        all_solids.append(solids)
        grid = np.array(preds, dtype=np.int32).reshape(9, 9)
        for r in range(9):
            for c in range(9):
                votes[r, c, grid[r, c]] += 1

    consensus_grid = np.argmax(votes, axis=2).tolist()        # 9x9 nested
    consensus_flat = [consensus_grid[r][c] for r in range(9) for c in range(9)]

    # For solidity: use whichever frame's prediction matched the per-cell
    # consensus first (list is already quality-sorted, best frame first).
    flat_solidities = [None] * 81
    for i in range(81):
        for preds, solids in zip(all_preds, all_solids):
            if preds[i] == consensus_flat[i]:
                flat_solidities[i] = solids[i]
                break

    return consensus_grid, flat_solidities


# ── Application states ─────────────────────────────────────────────────────────

class State:
    IDLE       = "IDLE"
    CAPTURING  = "CAPTURING"  # Watching for the best frame(s) in a time window
    SOLVING    = "SOLVING"    # Running the solver (one-shot)
    SOLVED     = "SOLVED"     # Solution found, AR overlay live-tracking
    ERROR      = "ERROR"      # Something went wrong


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
    capture    = BestFrameCapture(top_k=args.top_k, window_seconds=args.capture_window)

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
    inv_M         = None    # Inverse perspective matrix for AR (refreshed every
                             # frame while SOLVED — see "AR tracking" below)
    status_msg    = ""      # Message shown on HUD
    board_contour = None    # Last detected contour
    last_solidities = [None] * 81

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

        # ── Vision — run every frame (cheap: no CNN here) ──────────────────────
        img_thresh    = vision.pre_process(frame)
        board_contour = vision.find_board_contour(img_thresh)

        # ── Draw detection feedback ───────────────────────────────────────────
        if board_contour is not None and state in (State.IDLE, State.CAPTURING):
            vision.draw_contour(frame, board_contour)
            vision.draw_grid_overlay(frame, board_contour)

        # ── Phase 1: capture — collect best frames, no CNN yet ─────────────────
        if state == State.CAPTURING:
            if board_contour is not None:
                warped_gray = vision.get_warp(frame, board_contour)
                quality     = vision.estimate_quality(warped_gray, board_contour)
                capture.add(quality, warped_gray, board_contour)

            remaining = max(0.0, args.capture_window - capture.elapsed())
            status_msg = f"Capturing best frame... {remaining:.1f}s  (found {len(capture.candidates)})"

            if capture.is_ready():
                state = State.SOLVING

        # ── Phase 2: solve — CNN + solver run ONCE, on the best frames only ────
        if state == State.SOLVING:
            if not capture.has_any():
                status_msg = "No board detected during capture — point camera at puzzle"
                print(f"[CVdoku] {status_msg}")
                state = State.ERROR
            else:
                print(f"[CVdoku] Classifying {len(capture.best_frames())} best frame(s)...")
                consensus, last_solidities = vote_across_frames(
                    vision, classifier, capture.best_frames()
                )

                solvable, reason = solver.is_solvable(consensus)

                if not solvable:
                    print(f"[Solver] Cannot solve: {reason}")
                    status_msg = reason
                    state      = State.ERROR
                else:
                    original_grid = copy.deepcopy(consensus)

                    print("[Solver] Solving (with conflict recovery)...")
                    solved_attempt, success = solver.solve_with_recovery(
                        consensus, solidities=last_solidities
                    )

                    if success:
                        solved_grid = solved_attempt
                        state       = State.SOLVED
                        status_msg  = "Solved! Move the camera freely — AR will track the board."
                        print("[Solver] Solution found.")
                        _print_grid(original_grid, solved_grid)
                    else:
                        status_msg = "Unsolvable — rescan or improve lighting"
                        state      = State.ERROR
                        print("[Solver] No solution found even after recovery.")

        # ── Phase 3: AR tracking — re-detect position every frame, no CNN ──────
        if state == State.SOLVED:
            if board_contour is not None:
                # Digit values are already locked in (solved_grid). All we
                # need every frame is a FRESH perspective matrix from the
                # board's current position — cheap (no warpPerspective of
                # the actual image, no classification), so the overlay
                # tracks the puzzle live even if the camera moves.
                vision.compute_perspective_matrix(board_contour)
                inv_M = vision.get_inverse_warp_matrix()
            # If the board is briefly out of view, keep the last known
            # inv_M rather than dropping the overlay entirely — avoids
            # flicker on momentary occlusion/motion blur.

        # ── AR overlay ────────────────────────────────────────────────────────
        if state == State.SOLVED and solved_grid is not None and inv_M is not None:
            frame = overlay.draw(frame, original_grid, solved_grid, inv_M)

        # ── HUD ───────────────────────────────────────────────────────────────
        display_state = "SCANNING" if state == State.CAPTURING else state
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
                capture.start()
                state      = State.CAPTURING
                status_msg = "Capturing best frame..."
                print("\n[CVdoku] Starting capture window...")

        elif key == ord('r'):
            state         = State.IDLE
            original_grid = None
            solved_grid   = None
            inv_M         = None
            status_msg    = ""
            capture.reset()
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