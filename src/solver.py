class SudokuSolver:

    # ── Public API ─────────────────────────────────────────────────────────────

    def solve(self, grid: list) -> bool:
        """
        Solve the Sudoku puzzle in-place using backtracking.

        Args:
            grid: 9×9 list of lists. 0 = empty, 1-9 = given digit.

        Returns:
            True  if a solution was found (grid is now filled).
            False if no solution exists (grid is left partially modified).
        """
        empty = self._find_empty(grid)
        if empty is None:
            return True   # No empty cells left — solved!

        row, col = empty

        for num in range(1, 10):
            if self._is_valid(grid, row, col, num):
                grid[row][col] = num

                if self.solve(grid):
                    return True

                grid[row][col] = 0   # Backtrack

        return False   # Trigger backtracking in caller

    def is_valid_board(self, grid: list) -> bool:
        """
        Check whether the given (partially filled) board is valid —
        i.e. no duplicate digits in any row, column, or 3×3 box.

        Used to catch gross classifier errors before attempting to solve.

        Returns:
            True if the board is valid, False if there's a conflict.
        """
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
                        val = grid[box_row * 3 + r][box_col * 3 + c]
                        if val != 0:
                            box_vals.append(val)
                if len(box_vals) != len(set(box_vals)):
                    return False

        return True

    def count_givens(self, grid: list) -> int:
        """
        Count how many cells are already filled in.
        A real Sudoku puzzle typically has 17-40 givens.
        Fewer than 17 is mathematically unsolvable (non-unique solution).
        """
        return sum(1 for row in grid for val in row if val != 0)

    def is_solvable(self, grid: list) -> tuple:
        """
        Combined sanity check before attempting to solve.

        Returns:
            (True, "")          if the board looks solvable.
            (False, reason_str) if something is wrong.
        """
        givens = self.count_givens(grid)

        if givens < 17:
            return False, f"Only {givens} digits detected — likely a scan error (need 17+)"

        if givens == 81:
            return False, "Board is already complete"

        if not self.is_valid_board(grid):
            return False, "Board has conflicts — likely a misread digit"

        return True, ""

    # ── Private helpers ────────────────────────────────────────────────────────

    def _find_empty(self, grid: list) -> tuple | None:
        """
        Find the next empty cell (value == 0).
        Scans row-by-row, left to right.

        Returns (row, col) or None if no empty cell exists.
        """
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    return (i, j)
        return None

    def _is_valid(self, grid: list, row: int, col: int, num: int) -> bool:
        """
        Check whether placing `num` at (row, col) violates any Sudoku rule.

        Checks:
            1. Row   — num not already in this row
            2. Column — num not already in this column
            3. Box   — num not already in the 3×3 sub-grid
        """
        # Row check
        if num in grid[row]:
            return False

        # Column check
        if num in [grid[i][col] for i in range(9)]:
            return False

        # 3×3 box check
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if grid[r][c] == num:
                    return False

        return True