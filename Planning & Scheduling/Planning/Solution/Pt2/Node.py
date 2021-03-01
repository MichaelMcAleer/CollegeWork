# Assignment 1 - Planning & Scheduling
# Part 2 - A* Sliding Tile Puzzle
# Michael McAleer (R00143621)


class Node(object):
	"""
	This is the individual state of a sliding puzzle during the puzzle solve
	process. The node stores information on:
	- Its parent node,
	- The move required to get from the parent to the present state
	- The state of the node (number positions in an X x Y matrix
	- The flattened state of the node (1-D array)
	- The X and Y dimensions of the puzzle
	- The position of the free cell (tuple)
	- The f(N), g(N) and h(N) cost of the node
		- h(N) = the heuristic cost of the current state only
		- g(N) = the cumulative cost of all previous nodes
		- f(N) = the combined cost of the current state plus all previous nodes
	"""

	def __init__(self, state):
		"""
		Initialise the sliding puzzle node state.

		:param state: (nested-list of X Y dimensions) The state of the puzzle
		"""
		self.parent = None
		self.move_from_parent = None
		self.state = state
		self.flat = [col for row in state for col in row]
		self.x = len(state[0])
		self.y = len(state)
		self.h_cost = 0
		self.g_cost = 0
		self.f_cost = self.h_cost + self.g_cost
		for y_i in range(self.y):
			for x_i in range(self.x):
				if self.state[y_i][x_i] == 0:
					self.free_cell = (x_i, y_i)

	def __str__(self):
		"""
		String representation of the current node state, the 0 (blank) tile is
		replaced with '-' for ease of reading.

		:return: (str) The current state of the node
		"""
		response = str()
		for y_i in range(0, self.y):
			line = ""
			for x_i in range(0, self.x):
				val = self.state[y_i][x_i]
				if val == 0:
					val = "-"
				line += str(val) + " "
			if y_i != self.y - 1:
				line += "\n"
			response += line
		return response

	def __lt__(self, other):
		"""
		This assists in occasions where potential moves have same F value.

		:param other: (Node) The other state in the comparison
		:return: (Boolean) If the current F cost is less than the other Node F
		cost
		"""
		return self.f_cost < other.f_cost
