# Assignment 1 - Planning & Scheduling
# Part 2 - A* Sliding Tile Puzzle
# Michael McAleer (R00143621)
from heapq import heappush, heappop
import time
import sys

from Puzzle import Puzzle


class PuzzleSearch:
    """
    This class handles the puzzle and the forward search A* algorithm used to
    solve the puzzle.
    """

    def __init__(self, puzzle_matrix, heuristic, output_solution):
        """
        Initialise the forward search algorithm with a given sliding puzzle
        matrix.

        :param puzzle_matrix: (list) The 1-D array containing number positions
        :param heuristic: (str) The method of distance calculation, valid
        values are 'manhattan' and 'misplaced'
        :param output_solution: (bool) Output the puzzle solution to console
        """
        self.p = Puzzle(puzzle_matrix, heuristic)
        self.output_solution = output_solution
        self.closed_list = list()
        self.open_list = list()

    @staticmethod
    def convert_to_matrix(state, dimension):
        """
        Given a 1-D flat puzzle, convert it to the X/Y 2-D matrix.

        :param state: (list) The 1-D puzzle state
        :param dimension: (tuple) The X Y dimensions of the puzzle
        :return: (nested list) the 2-D puzzle matrix
        """
        x, y = dimension[0], dimension[1]
        i = 0
        new_list = list()
        while i < len(state):
            new_list.append(state[i:i + x])
            i += y
        return new_list

    @staticmethod
    def return_path_to_state(state):
        """
        Given a Node state, return the path from the initial state to the
        current state.

        :param state: (Node) The Node object with state defined
        :return: (list) The list of nodes in order from start state to given
        state
        """
        current_state = state
        # Create path list and add goal state as final state in path
        path = list()
        path.append(current_state)
        # While there are parent nodes, add each of these to the list at
        # index 0
        while current_state.parent:
            previous_state = current_state.parent
            path.insert(0, previous_state)
            current_state = previous_state
        # # All parent nodes are exhausted, add initial state at index 0
        # path.insert(0, current_state)
        return path

    def solve_puzzle(self):
        """
        Using the A* forward search algorithm, solve the given sliding puzzle.

        Results are output to screen, if graphical representation is required
        in output set output_solution to True.
        """
        # Set counters
        nodes_visited = 1
        nodes_expanded = 1
        iterations = 0

        # Set while loop condition and start timer
        solution_found = False
        time_start = time.clock()
        while not solution_found:
            # Set upper limit on the amount of loops the forward search
            # algorithm can reach before the program exits
            if iterations > 5000:
                sys.exit("5000 iterations reached, a solution could not be "
                         "found within the set upper limit.")

            # Expand current state to determine possible moves
            moves = self.p.get_possible_moves(self.p.current_node)
            # Add current state to closed lists
            self.closed_list.append(self.p.current_node.state)

            for move in moves:
                nodes_visited += 1
                # Make the move and calculate the F cost of the potential next
                # state
                state = self.p.node_move_result(self.p.current_node, move)
                self.p.calculate_node_cost(state)
                # Check if the target state is equal to the goal state
                if self.p.is_goal_state(state):
                    # Stop the timer
                    time_end = time.clock()
                    solution_found = True
                    state.move_from_parent = move
                    # Get the path from the initial state to the goal state
                    path_to_solution = self.return_path_to_state(state)
                    # Output results summary to screen
                    print("--------------------------\n"
                          "Puzzle initial state:\n"
                          "--------------------------\n{}".format(self.p.root))

                    print("--------------------------\n"
                          "Solution Found:\n"
                          "--------------------------\n"
                          "Nodes visited: {}\n"
                          "Nodes Expanded: {}\n"
                          "Cost of solution: {}\n"
                          "Moves required to complete puzzle: {}\n"
                          "Time taken (s): {}\n"
                          "--------------------------".format(
                            nodes_visited, nodes_expanded, state.f_cost,
                            len(path_to_solution) - 1,
                            (time_end - time_start)))

                    # If the detailed solution output is set to True, output
                    # path from start to finish
                    if self.output_solution:
                        move_cnt = 0
                        print("Solution moves:")
                        for node in path_to_solution:
                            print("--------------------------")
                            if node.move_from_parent:
                                print("{}. Blank position move: {}\n"
                                      "--------------------------".format(
                                        move_cnt, node.move_from_parent))
                            print(node)
                            move_cnt += 1
                        print("--------------------------\n"
                              "Goal state reached in {} moves and {} seconds\n"
                              "--------------------------".format(
                                len(path_to_solution) - 1,
                                (time_end - time_start)))

                # Push the state onto the heap with respective F score if the
                # state has not already been expanded and added to closed list
                if state.state in self.closed_list:
                    pass
                else:
                    heappush(self.open_list, (state.f_cost, state, move))

            # Take the next state with lowest F cost from heap
            f_score, next_state, move = heappop(self.open_list)
            self.p.current_node = next_state
            self.p.current_node.move_from_parent = move
            nodes_expanded += 1
