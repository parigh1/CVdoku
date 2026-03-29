#CVdoku

A simple computer vision project that detects and solves Sudoku puzzles in real-time. Just point your camera at a Sudoku grid, and the program recognizes the numbers, solves the puzzle, and shows you the result.
How it Works

    Vision: The program uses your webcam to find the Sudoku grid and squares it up.

    Recognition: A custom CNN model (trained on digits) looks at each cell to see which numbers are already there.

    Solving: It uses a backtracking algorithm to find the solution for the empty spots instantly.

Features

    Real-time grid detection and perspective warping.

    Digit recognition using a trained machine learning model.

    Fast Sudoku solving engine.

Tech Used

    Python: The core language.

    OpenCV: For image processing and camera handling.

    TensorFlow/Keras: For identifying the numbers.

    Backtracking Algorithm: To solve the logic of the puzzle.
