"""
CVdoku — Sudoku Solver
=======================
v2: Added conflict resolver that tries 1↔7 and 5↔6 swaps
    when the board has conflicts — catches remaining CNN misreads.
"""

import copy


class SudokuSolver:

    # ── Public API ─────────────────────────────────────────────────────────────

    def solve(self, grid: list) -> bool:
        """
        Solve the Sudoku puzzle in-place using backtracking.
        Returns True if solved, False if unsolvable.
        """
        empty = self._find_empty(grid)
        if empty is None:
            return True

        row, col = empty

        for num in range(1, 10):
            if self._is_valid(grid, row, col, num):
                grid[row][col] = num
                if self.solve(grid):
                    return True
                grid[row][col] = 0

        return False

    def solve_with_recovery(self, grid: list) -> tuple:
        """
        Try to solve the board. If it has conflicts, attempt automatic
        correction by trying common CNN misread swaps:
            1 ↔ 7  (most common — thin digit confusion)
            5 ↔ 6  (second most common — rounded digit confusion)
            8 ↔ 9  (occasional — similar shape)

        Returns:
            (solved_grid, True)  if solution found
            (None, False)        if unsolvable even after recovery
        """
        # First try: direct solve
        if self.is_valid_board(grid):
            attempt = copy.deepcopy(grid)
            if self.solve(attempt):
                return attempt, True

        # Recovery: try swapping common misread pairs
        swap_pairs = [(1, 7), (5, 6), (8, 9), (3, 8), (2, 7)]

        for a, b in swap_pairs:
            recovered = self._try_swap(grid, a, b)
            if recovered is not None:
                attempt = copy.deepcopy(recovered)
                if self.solve(attempt):
                    print(f"[Solver] Recovered by swapping {a}↔{b}")
                    return attempt, True

        # Try all combinations of two swaps
        for i, (a1, b1) in enumerate(swap_pairs):
            for a2, b2 in swap_pairs[i+1:]:
                recovered = self._try_swap(grid, a1, b1)
                if recovered is not None:
                    recovered2 = self._try_swap(recovered, a2, b2)
                    if recovered2 is not None:
                        attempt = copy.deepcopy(recovered2)
                        if self.solve(attempt):
                            print(f"[Solver] Recovered by swapping {a1}↔{b1} and {a2}↔{b2}")
                            return attempt, True

        return None, False

    def _try_swap(self, grid: list, a: int, b: int) -> list:
        """
        Return a new grid with all occurrences of digit `a` swapped with `b`.
        Only returns the swapped grid if it's valid; otherwise returns None.
        """
        new_grid = copy.deepcopy(grid)
        for r in range(9):
            for c in range(9):
                if new_grid[r][c] == a:
                    new_grid[r][c] = b
                elif new_grid[r][c] == b:
                    new_grid[r][c] = a

        if self.is_valid_board(new_grid):
            return new_grid
        return None

    def is_valid_board(self, grid: list) -> bool:
        """Check for duplicate digits in any row, column, or 3×3 box."""
        for i in range(9):
            row_vals = [grid[i][j] for j in range(9) if grid[i][j] != 0]
            if len(row_vals) != len(set(row_vals)):
                return False

            col_vals = [grid[j][i] for j in range(9) if grid[j][i] != 0]
            if len(col_vals) != len(set(col_vals)):
                return False

        for box_row in range(3):
            for box_col in range(3):
                box_vals = []
                for r in range(3):
                    for c in range(3):
                        val = grid[box_row*3+r][box_col*3+c]
                        if val != 0:
                            box_vals.append(val)
                if len(box_vals) != len(set(box_vals)):
                    return False

        return True

    def count_givens(self, grid: list) -> int:
        """Count filled cells."""
        return sum(1 for row in grid for val in row if val != 0)

    def is_solvable(self, grid: list) -> tuple:
        """
        Sanity check before solving.
        Returns (True, "") or (False, reason).
        """
        givens = self.count_givens(grid)

        if givens < 17:
            return False, f"Only {givens} digits detected — likely a scan error (need 17+)"

        if givens == 81:
            return False, "Board is already complete"

        # Note: we don't reject on conflict here anymore —
        # solve_with_recovery handles that case
        return True, ""

    # ── Private helpers ────────────────────────────────────────────────────────

    def _find_empty(self, grid: list):
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    return (i, j)
        return None

    def _is_valid(self, grid: list, row: int, col: int, num: int) -> bool:
        if num in grid[row]:
            return False
        if num in [grid[i][col] for i in range(9)]:
            return False
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if grid[r][c] == num:
                    return False
        return True