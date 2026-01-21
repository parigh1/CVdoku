class SudokuSolver:
    def __init__(self):
        pass

    def solve(self, board):
        """
        Solves the sudoku board using backtracking.
        Returns True if solved, False if unsolvable.
        The board is modified in-place.
        """
        find = self.find_empty(board)

        # Base Case: If no empty spots are left, we are done!
        if not find:
            return True
        else:
            row, col = find

        # Try numbers 1 through 9
        for i in range(1, 10):
            if self.is_valid(board, i, (row, col)):
                board[row][col] = i

                # Recursively try to solve the rest
                if self.solve(board):
                    return True

                # Backtrack: If it didn't work, reset to 0 and try next number
                board[row][col] = 0

        return False

    def is_valid(self, board, num, pos):
        """
        Checks if placing 'num' at 'pos' (row, col) is valid.
        """
        # Check Row
        for i in range(len(board[0])):
            if board[pos[0]][i] == num and pos[1] != i:
                return False

        # Check Column
        for i in range(len(board)):
            if board[i][pos[1]] == num and pos[0] != i:
                return False

        # Check 3x3 Box
        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if board[i][j] == num and (i, j) != pos:
                    return False

        return True

    def find_empty(self, board):
        """
        Finds the next empty cell (marked with 0).
        """
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    return (i, j)  # row, col
        return None