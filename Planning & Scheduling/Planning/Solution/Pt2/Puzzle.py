# Assignment 1 - Planning & Scheduling
# Part 2 - A* Sliding Tile Puzzle
# Michael McAleer (R00143621)
import math
import sys
import time

from Node import Node


class Puzzle:
    """
    This is the puzzle as a whole, with knowledge of the initial state,
    current state, goal state, costs etc.  Heuristic costs are also calculated
    here as well as potential moves from a given state, and the logic to make
    those moves. The puzzle class also has the ability to determine if a given
    puzzle is solvable or not, if not, the puzzle solver will quit gracefully.
    """

    def __init__(self, matrix, heuristic):
        """
        Initialise the sliding puzzle.

        :param matrix: (nested-list of X Y dimensions) The state of the puzzle
        :param heuristic: (str) The chosen heuristic for calculating distance
        to the goal state, valid values are 'manhattan' and 'misplaced'.
        """
        # Run a check to determine the distance metric is valid
        if heuristic not in ['manhattan', 'misplaced']:
            sys.exit("Invalid heuristic provided, exiting...")

        # Set the heuristic, root, present, and goal states
        self.heuristic = heuristic
        self.root = Node(matrix)
        self.current_node = self.root
        self.goal = self.calculate_goal_state()

        # Check that the given puzzle is solvable
        if not self.is_solvable(self.root):
            print("--------------------------\n"
                  "Puzzle initial state:\n"
                  "--------------------------\n{}\n"
                  "--------------------------".format(self.root))
            time.sleep(1)
            sys.exit("The given problem is not solvable")

        # Calculate the cost of puzzle's initial state (root node)
        self.calculate_node_cost(self.root)

    @staticmethod
    def is_solvable(node):
        """
        Checks if the given state is solvable.

        Note: This method has been taken from 'Artificial Intelligence: A
        Modern Approach' code repo:
            https://github.com/aimacode/aima-python/blob/master/search.py

        :param node: (Node) The Node object with state defined
        :return: (Boolean) If the puzzle is solvable
        """
        inversion = 0
        state = node.flat
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1
        return inversion % 2 == 0

    @staticmethod
    def convert_to_matrix(state, dimension):
        """
        Given a flat (1-D) puzzle, convert it to a matrix of the XY dimensions
        provided.

        :param state: (list) The flat puzzle
        :param dimension: (tuple) The XY dimensions to use for the puzzle
        :return: (nested list) The 2-D puzzle matrix
        """
        x, y = dimension[0], dimension[1]
        i = 0
        new_list = list()
        while i < len(state):
            new_list.append(state[i:i + x])
            i += y
        return new_list

    @staticmethod
    def pos_by_val(node, value):
        """
        Given a Node and a target tile value return the X Y position of that
        tile.

        :param node: (Node) The Node object with state defined
        :param value: (int) The value of the tile for which an index is
        required
        :return: (int, int) The X Y co-ordinates of the given value
        """
        tile_cnt = int(math.sqrt(len(node.flat)))
        x = node.flat.index(value) % tile_cnt
        y = math.floor(node.flat.index(value) / tile_cnt)
        return x, y

    def calculate_goal_state(self):
        """
        Calculate the goal state for the given puzzle. The goal state is
        determined to be all numbers in sequence from 1-X with the 0 (blank)
        tile located in the last position (bottom right corner).

        :return: (nested list) The goal state matrix
        """
        cnt = 1
        goal = list()
        for y_i in range(0, self.root.y):
            x_list = list()
            for x_i in range(0, self.root.x):
                x_list.append(cnt)
                cnt += 1
            goal.append(x_list)
        # Change last number in puzzle to 0 (blank) tile
        goal[self.root.y - 1][self.root.x - 1] = 0
        return Node(goal)

    def is_goal_state(self, node):
        """
        Determine if a given node state matches the expected goal state.

        :param node: (Node) The Node object with state defined
        :return: (Boolean) If the providedc state matches the goal state
        """
        return True if node.state == self.goal.state else False

    def heuristic_misplaced(self, node):
        """
        Return the heuristic cost to goal state for a given state.  The cost
        is determined to be the quantity of misplaced tiles minus 1 as the
        0 (blank) tile should not be counted as misplaced.

        :param node: (Node) The Node object with state defined
        :return: (int) The quantity of misplaced tiles
        """
        return sum(a != b for (a, b) in zip(node.flat, self.goal.flat)) - 1

    def heuristic_manhattan(self, node):
        """
        Return the heuristic cost to goal state for a given state.  The cost
        is determined to be the manhattan distance of each misplaced tile from
        its determined position in the goal state. The 0 (blank) tile is not
        included in this calculation.

        :param node: (Node) The Node object with state defined
        :return: (int) The cumulative manhattan distance of all tiles
        """
        distance = 0
        for i in node.flat:
            # exclude 0 (Blank) tile
            if i != 0:
                # Calculate position of current tile with regards to matrix
                # and get equivalent position of tile in goal state matrix
                x_i, y_i = self.pos_by_val(node, i)
                goal_x_i, goal_y_i = self.pos_by_val(self.goal, i)
                # Using positions of current and goal state, calculate
                # manhattan distance between the two, add to cumulative total h
                distance += abs(goal_x_i - x_i) + abs(goal_y_i - y_i)
        return distance

    def calculate_node_cost(self, node):
        """
        Calculate the heuristic cost of the given node from the goal state.

        :param node: (Node) The Node object with state defined
        :return: (int) The heuristic cost of the current state
        """
        if self.heuristic == 'misplaced':
            node.h_cost = self.heuristic_misplaced(node)
        if self.heuristic == 'manhattan':
            node.h_cost = self.heuristic_manhattan(node)

        # If the current node has a parent, set the current node's g(N) cost
        # as the f(N) cost of the parent to maintain the cost to current state
        # from the initial state.
        if node.parent:
            node.g_cost = node.parent.f_cost
        node.f_cost += node.h_cost + node.g_cost

    @staticmethod
    def get_possible_moves(node):
        """
        Given a node state, determine all the possible moves that can be made
        using the 0 (blank) tile.

        :param node: (Node) The Node object with state defined
        :return: (list) The possible moves that can be made
        """
        # Start with all possible moves, get the X Y position of the free cell,
        # and determine its index within the equivalent 1-D array
        possible_moves = ['Up', 'Down', 'Left', 'Right']
        x, y = node.free_cell[0], node.free_cell[1]
        free_cell_index = x + (y * 3)

        # Free cell is on left-most boundary
        if free_cell_index % node.x == 0:
            possible_moves.remove('Left')
        # Free cell is on the top row
        if free_cell_index < node.x:
            possible_moves.remove('Up')
        # Free cell is on the right most boundary
        if free_cell_index % node.x == 2:
            possible_moves.remove('Right')
        # Free cell is on the bottom row
        if free_cell_index > ((len(node.flat)-1) - node.x):
            possible_moves.remove('Down')
        return possible_moves

    def node_move_result(self, node, direction):
        """
        Given a node state and a direction of movement, move the free cell (0)
        to the target position.

        :param node: (Node) The Node object with state defined
        :param direction: (str) The direction of travel
        :return: (Node) The target state after move has been made
        """
        # Get the index of the blank tile in the flattened puzzle state
        f_cell = node.flat.index(0)
        n_state = list(node.flat)
        action = {'Up': (node.x * -1), 'Down': node.x, 'Left': -1, 'Right': 1}
        # Get targeted position (by index) for blank tile to move to
        n_index = f_cell + action[direction]
        # Apply the move by switching the neighbour tile and blank tile
        n_state[f_cell], n_state[n_index] = n_state[n_index], n_state[f_cell]

        new_state = Node(self.convert_to_matrix(n_state, (self.root.x,
                                                          self.root.y)))
        new_state.parent = node

        return new_state
