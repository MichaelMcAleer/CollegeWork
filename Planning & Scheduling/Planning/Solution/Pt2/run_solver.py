# Assignment 1 - Planning & Scheduling
# Part 2 - A* Sliding Tile Puzzle
# Michael McAleer (R00143621)
import random
import PuzzleSearch as Puzz

# # Puzzle options
heuristic = 'manhattan'
# heuristic = 'misplaced'
solution_output = True

# # >>>>>>>> Sliding puzzle start states
# Random start
m_random = list(range(0, 9))
random.shuffle(m_random)
# Easy/Medium start
m_easy = [[1, 2, 3], [4, 5, 6], [0, 7, 8]]
m_medium = [[4, 0, 1], [5, 8, 2], [7, 6, 3]]
# Ascending difficulty puzzles
puzzle_list = [[1, 2, 3, 4, 0, 5, 7, 8, 6],
               [1, 2, 3, 7, 4, 5, 0, 8, 6],
               [1, 2, 3, 4, 8, 0, 7, 6, 5],
               [1, 6, 2, 5, 3, 0, 4, 7, 8],
               [5, 1, 2, 6, 3, 0, 4, 7, 8],
               [1, 2, 6, 3, 5, 0, 4, 7, 8],
               [3, 5, 6, 1, 4, 8, 0, 7, 2],
               [4, 3, 6, 8, 7, 1, 0, 5, 2],
               [3, 0, 2, 6, 5, 1, 4, 7, 8],
               [0, 1, 2, 3, 4, 5, 6, 7, 8],
               [5, 0, 3, 2, 8, 4, 6, 7, 1],
               [8, 7, 4, 3, 2, 0, 6, 5, 1],
               [8, 7, 6, 5, 4, 3, 0, 2, 1],
               [8, 7, 6, 5, 4, 3, 2, 1, 0]]

# Uncomment out the code below to run either the easy/medium difficulty
# puzzles, the set of puzzles in ascending difficulty, or the randomly
# generated start state

# # >>>>>>>> Start with Easy/Medium Puzzles
p = Puzz.PuzzleSearch(m_medium, heuristic, solution_output)
p.solve_puzzle()

# # >>>>>>>> Start with gradually rising difficulty puzzles
# for game in puzzle_list:
#     m_game = Puzz.PuzzleSearch.convert_to_matrix(game, (3, 3))
#     p = Puzz.PuzzleSearch(m_game, heuristic, solution_output)
#     p.solve_puzzle()

# # >>>>>>>> RANDOM START
# m_random = Puzz.PuzzleSearch.convert_to_matrix(m_random, (3, 3))
# p = Puzz.PuzzleSearch(m_random, heuristic, solution_output)
# p.solve_puzzle()
