# Assignment 1 - Planning & Scheduling
# Part 2 - A* Sliding Tile Puzzle
# Michael McAleer (R00143621)

There are no additional packages required or program dependencies other than
Python 3.x to be in use.  The program only needs the following files to run:
    - Node.py
    - Puzzle.py
    - PuzzleSearch.py
    - run_solver.py

To run the 'A* Sliding Tile Puzzle' solver run the following from the
command line whilst in the 'AStarPuzzleSolver' directory:

    $ python run_solver.py

The 'run_solver.py' file has a range of starting states for the puzzle solver
including:
    - Randomly generated start
    - Easy difficulty start
    - Medium difficulty start
    - List of puzzles sorted in ascending difficulty

To run any of these, uncomment out the blocks of code pertaining to each of the
possible starting states.

If you do not require the path from the initial state to the goal state output
to screen once puzzle has been solved, set 'solution_output' to False in
run_solver.py.

To change heuristic distance calculation from 'manhattan' to 'misplaced' just
uncomment the required 'heuristic' option in run_solver.py

If randomly generating puzzles, the puzzle solver will perform a check on
initialisation to ensure that the puzzle is solvable, if not, the program will
exit.