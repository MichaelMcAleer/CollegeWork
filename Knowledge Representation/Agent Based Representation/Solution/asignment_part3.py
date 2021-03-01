try:
    from aima_src import *
except ImportError:
    from agents import *

warnings.filterwarnings("ignore")
# global variable to remember to turn if our dog hits the boundary

# -----------------------------------------------------------------------------
# Classes built for things in Outlaw 2-D game, additional classes added for
# perceptions of environment things:


class Dog(Thing):
    pass


class Bark(Thing):
    pass


class Chest(Thing):
    pass


class XMark(Thing):
    pass


class Gold(Thing):
    pass


class Glitter(Thing):
    pass


class Snake(Thing):
    pass


class Hiss(Thing):
    pass


class Sheriff(Thing):
    pass


class Stench(Thing):
    pass


class Horse(Thing):
    pass


class Neigh(Thing):
    pass


# -----------------------------------------------------------------------------
# Outlaw - the agent of the game

class OutLaw(Agent):
    # # Agent Attributes
    location = (0, 0)
    luck = random.random()
    # Agent states & performance measures
    gold = 0
    life = 100
    steps = 0
    dog = False
    found_Horse = False
    treasure_found = False
    bitten_by_snake = False
    escaped_by_horse = False
    bribed_sheriff = False
    saved_by_dog = False
    locked_up = False
    chest_found = False

    def move_forward(self, location):
        """
        Move forward to given environment location
        :param location: (tuple) the location the agent needs to move to
        """
        self.location = location

    @staticmethod
    def open(env_chest):
        """
        Check if Thing detected in the environment is a chest and can be opened
        :param env_chest: (thing) The chest
        :return: (boolean) True if chest class else False
        """
        return True if isinstance(env_chest, Chest) else False

    @staticmethod
    def find(env_gold):
        """
        Check if Thing detected in the environment is gold and can be taken
        :param env_gold: (thing) The gold
        :return: (boolean) True if gold class else False
        """
        return True if isinstance(env_gold, Gold) else False

    @staticmethod
    def bitten(env_snake):
        """
        Check if Thing detected in the environment is a snake and can bite the
        agent
        :param env_snake: (thing) The snake
        :return: (boolean) True if snake class else False
        """
        return True if isinstance(env_snake, Snake) else False

    @staticmethod
    def arrested(env_sheriff):
        """
        Check if Thing detected in the environment is a Sheriff and can arrest
        agent
        :param env_sheriff: (thing) The sheriff
        :return: (boolean) True if sheriff class else False
        """
        return True if isinstance(env_sheriff, Sheriff) else False

    @staticmethod
    def escaped(env_horse):
        """
        Check if Thing detected in the environment is a Horse and agent can
        escape on
        :param env_horse: (thing) The horse
        :return: (boolean) True if horse class else False
        """
        return True if isinstance(env_horse, Horse) else False

    @staticmethod
    def rescue(env_dog):
        """
        Check if Thing detected in the environment is a dog and can be rescued
        by agent
        :param env_dog: (thing) The dog
        :return: (boolean) True if dog class else False
        """
        return True if isinstance(env_dog, Dog) else False


# -----------------------------------------------------------------------------
# WildWest - The 2-D game environment

class WildWest(XYEnvironment):

    def __init__(self, agent_program, width=5, height=5):
        """
        On Environment initialisation, setup the world and define the agent's
        primary goal and escape goal
        :param agent_program: (method) the program which controls agent's
        perceptions and actions
        :param width: (int) The environment width (# of cells)
        :param height: (int) The environment height (# of cells)
        """
        super().__init__(width, height)
        self.init_world(agent_program)
        # Set Goals
        self.agent_goal = Gold
        self.agent_escape_goal = Horse
        # Get the adjacent (neighbour) cells for every cell in environment
        self.grid_links = self.get_grid_links(width, height)

    def init_world(self, program):
        """
        Initialise the environment agent, things and thing percepts.
        Add agent to start position (0,0), add things to random locations, and
        add thing percepts to adjacent cells if not gold or chest, add percept
        only to own cell if gold or chest.

        :param program: (method) the program which controls agent's
        perceptions and actions
        """
        print("---------------------------------------------------")
        print(">> Setting up Wild West Environment")
        # Initialise Outlaw
        outlaw = OutLaw(program)
        self.add_thing(outlaw, (0, 0))
        print("\tThe Outlaw's luck today is {}".format(outlaw.luck))
        # Initialise Dog
        location = self.random_location_inbounds(exclude=(0, 0))
        self.add_thing(Dog(), location, True)
        self.add_thing_percept('Dog', location)
        # Initialise Chest
        location = self.random_location_inbounds(exclude=(0, 0))
        self.add_thing(Chest(), location, True)
        self.add_thing(XMark(), location, True)
        # Initialise Horse
        location = self.random_location_inbounds(exclude=(0, 0))
        self.add_thing(Horse(), location, True)
        self.add_thing_percept('Horse', location)
        # Initialise Gold
        location = self.random_location_inbounds(exclude=(0, 0))
        self.add_thing(Gold(), location, True)
        self.add_thing(Glitter(), location, True)
        # Initialise Snake
        location = self.random_location_inbounds(exclude=(0, 0))
        self.add_thing(Snake(), location, True)
        self.add_thing_percept('Snake', location)
        # Initialise Sheriff
        location = self.random_location_inbounds(exclude=(0, 0))
        self.add_thing(Sheriff(), location, True)
        self.add_thing_percept('Sheriff', location)
        print(">> Starting game")
        print("---------------------------------------------------")

    def run(self, search_type='bfs', default_moves=True):
        """
        WildWest runtime adaption

        This is where the search algorithm is controlled from, where the
        goal is first to find the Gold, then find the Horse, not necessarily
        in that order however, the agent will return to the Horse if found
        before the Gold, when the Gold is found.

        :param search_type: (string) The search algorithm to use, possible
        values are 'bfs' and 'dfs', default to 'bfs' if not specified when
        called.
        - 'bfs' = Breadth First Search
        - 'dfs = Depth First Search
        :param default_moves: If the agent will move according to standard
        binary tree traversal of BFS - 0 => 1 => 2 => 3, or if agent will make
        only legal moves (not move to a square other than an adjacent one on
        the way to next cell in BFS).
        :return: Run results
        """
        # Set search type
        bfs, dfs = False, False
        if 'bfs' in search_type:
            bfs = True
        elif 'dfs' in search_type:
            dfs = True

        # Get agent and set start location
        agent = self.agents[0]
        start = agent.location
        # Get the grid mapping for all cells in environment
        neighbors = self.get_grid_links(self.width, self.height)
        # Initialise container to store the queue of cells to visit in search
        frontier = deque([start])
        # Initialise dictionary to store all previously visited cells, avoids
        # visiting same cells twice where applicable
        previous = {start: None}
        escaping = False
        horse_location = None

        # Run the search until completion
        while frontier:
            # Agent has died, do not continue run, return results of run
            if not agent.alive:
                return self.output_results()

            # If the search algorithm is BFS, use popleft() to implement FIFO
            # queue control
            if bfs:
                s = frontier.popleft()
            # If the search algorithm is BFS, use pop() to implement LIFO
            # queue control
            if dfs:
                s = frontier.pop()

            # Get agent perceptions of contents of current location
            agent_percept = self.percept_at_location(agent.location)
            # If the agent percepts Gold or a Horse, handle the agent's action
            # correctly
            for p in agent_percept:
                if isinstance(p, Gold):
                    # If the agent has found the gold but has not yet found
                    # the horse to escape on...
                    if not horse_location:
                        print("Agent has found gold, now to find the horse")
                        # Agent now needs to find horse to escape on
                        escaping = True
                    # Else the agent has also found the horse, return to the
                    # location of the horse
                    elif horse_location:
                        escaping = True
                        self._find_treasure(agent)
                        print("Found Gold, returning to horse to escape")
                        path_to_escape = None
                        # Dependent on the search algorithm used, get the best
                        # path to horse using same search algorithm
                        if bfs:
                            path_to_escape = self.breadth_first(agent.location,
                                                                horse_location,
                                                                neighbors)
                        elif dfs:
                            path_to_escape = self.depth_first(agent.location,
                                                              horse_location,
                                                              neighbors)
                        # Move to the target cell containing horse
                        for cell in path_to_escape:
                            self.execute_action(agent, 'move_forward', cell)
                            agent.steps += 1
                        agent.escaped_by_horse = True

                if isinstance(p, Horse):
                    # The agent is not trying to escape, has found the horse
                    # but has not yet found the gold
                    if not escaping:
                        print("Found the horse, find the gold to escape with!")
                        horse_location = agent.location
                        agent.found_Horse = True
                    # Else the agent is escaping, gold has been found and now
                    # the horse has been found to escape on
                    if escaping:
                        print("Agent found the horse and escapes!")
                        agent.found_Horse = True
                        agent.escaped_by_horse = True
                        return self.output_results()

            # Handle the agent's actions depending on the state of the current
            # cell they are on, if they are not determined as being 'done' with
            # with the environment
            if not self.is_done():
                actions = []
                for agent in self.agents:
                    if agent.alive:
                        actions.append(agent.program(self.percept(agent)))
                    else:
                        actions.append("")
                for (agent, action) in zip(self.agents, actions):
                    self.execute_action(agent, action)

            # For all the neighbours of the current cell
            for s2 in neighbors[s]:
                # Move back to previous cell before going to next node, not
                # possible to traverse between nodes on the same level, have
                # to move back first
                if not default_moves:
                    current_neighbours = neighbors.get(tuple(agent.location))
                    # If the target cell is not a neighbour of the current
                    # cell...
                    if s2 not in current_neighbours and s2 != (0, 0):
                        print("No path to next cell {}, finding next best "
                              "route".format(s2))
                        path_to_s2 = self.breadth_first(agent.location,
                                                        s2,
                                                        neighbors)
                        # Delete first and last in path, currently on first
                        # cell and the last cell is the destination
                        if len(path_to_s2) >= 3:
                            del path_to_s2[0]
                            del path_to_s2[-1]
                        else:
                            del path_to_s2[0]
                        print("Route to {} found: {}".format(s2, path_to_s2))
                        for cell in path_to_s2:
                            # Move agent to target cell via adjacent cells
                            self.execute_action(agent, 'move_forward', cell)
                            agent.steps += 1

                # If agent is not done and the target cell has not been visited
                # already...
                if not self.is_done() and s2 not in previous:
                    # Get agent's perceptions of target cell
                    s2_percept = self.percept_at_location(s2)
                    # Append the target cell to the queue
                    frontier.append(s2)
                    # Add the current cell to the previously visited cells
                    previous[s2] = s
                    skip_cell = False
                    # If the agent percepts something in the target cell...
                    if s2_percept:
                        for percept in s2_percept:
                            # If the agent percepts the sheriff, avoid that
                            # cell
                            if isinstance(percept, Sheriff):
                                print("Sheriff in next location, avoiding!")
                                skip_cell = True
                            # If the agent percepts the Snake but has not yet
                            # found the dog to kill the snake, avoid that cell
                            if isinstance(percept, Snake) and not agent.dog:
                                print("Snake in next location, avoiding!")
                                skip_cell = True
                    # If the agent does not have to skip cell, move to the
                    # target cell
                    if not skip_cell:
                        self.execute_action(agent, 'move_forward', s2)
                        agent.steps += 1
        # Once all cells have been visited or cells avoided because of the
        # snake or sheriff, output the results to the screen
        if not frontier:
            if not agent.treasure_found and agent.found_Horse:
                print("The Outlaw found the horse to escape on but "
                      "did not find the Gold!")
            if not agent.found_Horse and agent.treasure_found:
                print("The Outlaw found the Gold but did not find the horse"
                      "to escape on!")
            else:
                print("The Outlaw did not find the Gold or the horse to "
                      "escape on...")
            return self.output_results()

    def output_results(self):
        """
        The results of the current agent performance during run.
        :return: (dict) Agent statistics on run
        """
        agent = self.agents[0]
        run_results = {
            'agent_escaped': agent.escaped_by_horse,
            'agent_steps': agent.steps,
            'treasure_found': agent.treasure_found,
            'chest_found': agent.chest_found,
            'agent_gold': agent.gold,
            'dog_found': agent.dog,
            'horse_found': agent.found_Horse,
            'bitten_by_snake': agent.bitten_by_snake,
            'saved_by_dog': agent.saved_by_dog,
            'agent_bribed': agent.bribed_sheriff,
            'agent_locked_up': agent.locked_up}
        return run_results

    @staticmethod
    def get_grid_links(width, height):
        """
        For all cells in the environment, return all adjacent (neighbour) cells
        :param width: (int) The environment width
        :param height: (int) The environment height
        :return: (dict) All cells and their neighbours
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def neighbors(x, y):
            # For all neighbours of location (x,y) return all adjacent
            # neighbours if the neighbour is withing the width/height boundary
            for (dx, dy) in directions:
                (nx, ny) = (x + dx, y + dy)
                if 0 <= nx < width and 0 <= ny < height:
                    yield (nx, ny)

        # Repeat for all cells in width/height range
        return {(x, y): list(neighbors(x, y))
                for x in range(width) for y in range(height)}

    def path(self, previous, s):
        """
        Return a list of states that lead to cell s, according to the
        previous dict.
        :param previous: (dict) The dictionary of previously visited cells
        :param s: (tuple) The goal cell (state)
        :return: (list) Path to cell if possible else empty dict
        """
        return [] if (s is None) else (
            self.path(previous, previous[s]) + [s])

    def breadth_first(self, start, goal, neighbors):
        """
        Breadth first search algorithm, used when determining legal moves to
        next cell or path back to horse.
        :param start: (tuple) The start location
        :param goal: (tuple) The end location
        :param neighbors: (dict) The dictionary of all neighbour cells for each
        cell
        :return: (list) The path from the start cell to the end cell if
        possible else empty list
        """
        frontier = deque([start])
        previous = {start: None}
        while frontier:
            s = frontier.popleft()
            if s == goal:
                return self.path(previous, s)
            for s2 in neighbors[s]:
                if s2 not in previous:
                    frontier.append(s2)
                    previous[s2] = s

    def depth_first(self, start, goal, neighbors):
        """
        Depth first search algorithm, used when determining legal moves to
        next cell or path back to horse.
        :param start: (tuple) The start location
        :param goal: (tuple) The end location
        :param neighbors: (dict) The dictionary of all neighbour cells for each
        cell
        :return: (list) The path from the start cell to the end cell if
        possible else empty list
        """
        frontier = deque([start])
        previous = {start: None}
        while frontier:
            s = frontier.pop()
            if s == goal:
                return self.path(previous, s)
            for s2 in neighbors[s]:
                if s2 not in previous:
                    frontier.append(s2)
                    previous[s2] = s

    def add_thing_percept(self, thing, location):
        """
        Add thing percepts to all adjacent cells i.e. above, below, left, right
        :param thing: (string) the environment thing
        :param location: (tuple) the thing location
        """
        x, y = location

        def percept():
            pass

        if thing == 'Dog':
            def percept():
                return Bark()
        elif thing == 'Horse':
            def percept():
                return Neigh()
        elif thing == 'Snake':
            def percept():
                return Hiss()
        elif thing == 'Sheriff':
            def percept():
                return Stench()

        self.add_thing(percept(), (x - 1, y), True)
        self.add_thing(percept(), (x, y - 1), True)
        self.add_thing(percept(), (x + 1, y), True)
        self.add_thing(percept(), (x, y + 1), True)

    def percept(self, agent):
        """
        Percept things at current agent location and all adjacent locations
        :param agent: (agent) The environment agent
        :return: (list) The things at agent location, empty list if none
        """
        x, y = agent.location
        result = list()

        result.append(self.list_things_at((x - 1, y)))
        result.append(self.list_things_at((x + 1, y)))
        result.append(self.list_things_at((x, y - 1)))
        result.append(self.list_things_at((x, y + 1)))
        result.append(self.list_things_at((x, y)))

        return result

    def percept_at_location(self, location):
        """
        Pecept things at a given location only.
        :param location: (tuple) The target location
        :return: (list) The things at provided location if things, else empty
        list
        """
        return self.list_things_at(location)

    def _open_chest(self, agent):
        """
        The logic for opening a chest in the game environment
        :param agent: (agent) The environment agent
        """
        # List the things at the current agent location of class type chest,
        # if chests returned run check that agent can open chest, if True,
        # run chest open logic
        x, y = agent.location
        items = self.list_things_at((x, y), tclass=Chest)
        if len(items) != 0:
            if agent.open(items[0]):
                agent.chest_found = True
                print("The Outlaw opened {} at location: {}".format(
                    str(items[0])[1:-1], agent.location))
                # Run random number gen for snake or treasure, player has a 20%
                # chance of finding a snake in the chest, 80% chance to find
                # gold
                random_num = random.random()
                if random_num <= 0.2:
                    # 20% chance of finding snake in the box
                    print("The Outlaw found a snake in the chest!")
                    # If the agent has found the dog, the dog will kill
                    # the snake
                    if agent.dog:
                        print("The Outlaw's trusty dog companion fought the "
                              "snake and killed it!")
                        agent.saved_by_dog = True
                    # Else the agent gets bitten by the snake, implement
                    # countdown on life if bitten
                    else:
                        print("The Outlaw got bitten by the snake!")
                        agent.bitten_by_snake = True
                else:
                    # 80% chance of finding gold in the box, add gold to agent
                    # gold attribute
                    chest_gold = random.randint(50, 1000)
                    agent.gold += chest_gold
                    print("The Outlaw found {} pieces of gold in the "
                          "chest!".format(chest_gold))

                # Delete treasure chest from environment
                self.delete_thing(items[0])

    def _find_treasure(self, agent):
        """
        The logic for finding gold in the game environment
        :param agent: (agent) The environment agent
        """
        # List the things at the current agent location of class type gold,
        # if gold returned run check that agent can 'find' gold, if True,
        # run find gold logic
        x, y = agent.location
        items = self.list_things_at((x, y), tclass=Gold)
        if len(items) != 0:
            if agent.find(items[0]):
                # Generate a random number between 1000 and 10000 for the
                # amount of gold the agent will find, add it to the agent gold
                # attribute and set treasure_found to True so agent can now
                # escape when horse is found
                chest_gold = random.randint(1000, 10000)
                agent.gold += chest_gold
                agent.treasure_found = True
                print("The Outlaw found the treasure of {} pieces of gold at "
                      "location: {}".format(chest_gold,
                                            agent.location))
                self.delete_thing(items[0])

    def _bitten_by_snake(self, agent):
        """
        The logic for getting bit by a snake in the game environment
        :param agent: (agent) The environment agent
        """
        # List the things at the current agent location of class type snake,
        # if snake returned run check that agent can bit by snake, if True,
        # run bit by snake logic
        x, y = agent.location
        items = self.list_things_at((x, y), tclass=Snake)
        if len(items) != 0:
            if agent.bitten(items[0]):
                print("The Outlaw got cornered by an angry snake at "
                      "{}!".format(agent.location))
                # If the agent has rescues a dog in the game, the dog can kill
                # the snake and save the agent
                if agent.dog:
                    print("The Outlaw's trusty dog companion fought the snake "
                          "and killed it!")
                    self.delete_thing(items[0])
                    agent.saved_by_dog = True
                # Else the agent gets bitten by the snake, initialise countdown
                # on life attribute if bitten
                else:
                    print("The Outlaw got bitten by a snake! The venom starts "
                          "to take effect on the Outlaw's health")
                    agent.bitten_by_snake = True
                    self.delete_thing(items[0])

    def _arrested_by_sheriff(self, agent):
        """
        The logic for getting arrested by the sheriff in the game environment
        :param agent: (agent) The environment agent
        """
        # List the things at the current agent location of class type sheriff,
        # if sheriff returned run check that agent can be arrested by sheriff,
        # if True run chest open logic
        x, y = agent.location
        items = self.list_things_at((x, y), tclass=Sheriff)
        if len(items) != 0:
            if agent.arrested(items[0]):
                print("The Outlaw encountered the crooked Sheriff at "
                      "location: {}".format(agent.location))
                # If the agent has gold in their possession, attempt to bribe
                # the sheriff
                if agent.gold:
                    print("The Outlaw tried his luck bribing the Sheriff...")
                    # If the agent's luck is less than the randomly
                    # generated number, the agent's luck isn't great enough and
                    # the sheriff arrests the outlaw, game over
                    if random.random() < agent.luck:
                        print("The Outlaw's luck ran out, Sheriff threw him "
                              "in jail")
                        agent.locked_up = True
                        agent.alive = False
                    # Else the agent't luck is greater than the randomly
                    # generated number, sheriff takes bribe of half of the
                    # current gold the outlaw has in possession
                    else:
                        bribe_amount = int(agent.gold / 2)
                        agent.gold -= bribe_amount
                        print("The Outlaw is in luck, Sheriff accepted bribe "
                              "of {}".format(bribe_amount))
                        self.delete_thing(items[0])
                        agent.bribed_sheriff = True
                # Agent has no gold to attempt to bribe the sheriff with, gets
                # thrown in jail and game over
                else:
                    print("The Outlaw had no gold to bribe the Sheriff and "
                          "got thrown in jail...")
                    self.delete_thing(items[0])
                    agent.locked_up = True
                    agent.alive = False

    def _horse_escape(self, agent):
        """
        The logic for escaping on a horse in the game environment
        :param agent: (agent) The environment agent
        """
        # List the things at the current agent location of class type horse,
        # if horse returned run check that agent can escape on horse, if True,
        # run escape on horse logic
        x, y = agent.location
        items = self.list_things_at((x, y), tclass=Horse)
        if len(items) != 0:
            if agent.escaped(items[0]):
                print("The Outlaw found the Horse to escape on at location: "
                      "{}".format(agent.location))
                # Agent has already found the gold, escape on horse
                if agent.treasure_found and agent.gold > 0:
                    print("The Outlaw escaped with {} gold pieces! Game "
                          "Over!".format(agent.gold))
                    agent.alive = False
                    agent.escaped_by_horse = True
                # Agent has found the gold but has dropped it all
                elif agent.treasure_found and agent.gold == 0:
                    print("The Outlaw dropped all his gold but decides to "
                          "cut his losses because there is no treasure "
                          "left. The Outlaw escaped broke but free!")
                    self.delete_thing(items[0])
                    agent.alive = False
                    agent.escaped_by_horse = True
                # Else the gold has not been found, game continues until gold
                # is found
                else:
                    print("The Outlaw hasn't found any gold yet to escape on "
                          "the horse with, the search continues!")
                    agent.found_Horse = True

    def _dog_rescue(self, agent):
        """
        The logic for rescuing the dog in the game environment
        :param agent: (agent) The environment agent
        """
        # List the things at the current agent location of class type dog,
        # if dog returned run check that agent can rescue dog, if True,
        # run rescue dog logic
        x, y = agent.location
        items = self.list_things_at((x, y), tclass=Dog)
        if len(items) != 0:
            if agent.rescue(items[0]):
                print("The Outlaw rescued a dog at location: {}".format(
                    agent.location))
                agent.dog = True
                self.delete_thing(items[0])

    def execute_action(self, agent, action, location=None):
        """
        Change the state of the environment based on what the agent does.

        :param agent: (agent) The environment agent
        :param action: (string) The action for the agent to carry out
        :param action: (tuple) If move_forward, the location to move to
        """
        if action == 'move_forward':
            print("Outlaw decided to move forwards at location {} to "
                  "location {}".format(agent.location, location))
            # Control the reduction in Outlaw gold pieces value per step
            # taken only if the Outlaw has found the treasure and the value
            # of gold is above 0
            if agent.treasure_found and agent.gold > 0:
                # Minus 100 gold from total gold held
                agent.gold -= 100
                if agent.gold <= 0:
                    # Set gold to 0 as gold cannot be a negative value
                    agent.gold = 0
                    print("The Outlaw has dropped all his gold!")
                else:
                    print("100 pieces of gold dropped! Outlaw now has {} "
                          "pieces of gold!".format(agent.gold))

            # 50/50 chance of running this block of code for the dog going
            # to find anti-venom for the snakebite
            if (agent.dog and agent.bitten_by_snake and
                    random.random() >= 0.5):
                # If the Outlaw's luck is good, the dog will find the
                # anti-venom
                if random.random() < agent.luck:
                    print("Man's best friend has found some anti-venom "
                          "for the snake bite! The venom no longer "
                          "affects the Outlaw!")
                    agent.bitten_by_snake = False
                    agent.saved_by_dog = True

            # If the Outlaw has been bitten by a snake handle the health
            # score reduction, if the health is 0 then game over
            if agent.bitten_by_snake:
                agent.life -= 10
                print("The Outlaw continues to feel the effects of the "
                      "snake bite and loses another 10 health points, "
                      "current health now at {}".format(agent.life))
                if agent.life == 0:
                    print("The snake venom has killed the outlaw! Game "
                          "Over!")
                    agent.alive = False
            if agent.alive:
                agent.move_forward(location)

        # Open Chest
        elif action == 'open':
            self._open_chest(agent)

        # Found the gold treasure
        elif action == 'find':
            self._find_treasure(agent)

        # Get bitten by snake
        elif action == 'bitten':
            self._bitten_by_snake(agent)

        # Get arrested by Sheriff
        elif action == 'arrested':
            self._arrested_by_sheriff(agent)

        # Escape on horse
        elif action == 'escaped':
            self._horse_escape(agent)

        # Rescue the dog
        elif action == 'rescue':
            self._dog_rescue(agent)

    def is_done(self):
        """
        Run a check to determine if the agent/game is done, this is determined
        by the current alive status of agent attribute.

        :return: (boolean) True if agent is dead, else False
        """
        return not any(agent.is_alive() for agent in self.agents)


def outlaw_program(percepts):
    """
    The program to control the agent actions based on agent perception of
    current cell. After agent carries out action on game environment, perform
    random choice of next agent move up (50%) or down (50%)

    :param percepts: (list) What the agent currently percepts
    :return: (string) The agent action determined by cell perception
    """
    on_square = percepts[4]
    for p in on_square:
        if isinstance(p, Dog):
            return 'rescue'
        elif isinstance(p, Chest):
            return 'open'
        elif isinstance(p, Gold):
            return 'find'
        elif isinstance(p, Snake):
            return 'bitten'
        elif isinstance(p, Sheriff):
            return 'arrested'
        elif isinstance(p, Horse):
            return 'escaped'
        # If environment 'percept'
        elif isinstance(p, Bark):
            print("Outlaw can hear a dog bark nearby...")
        elif isinstance(p, Hiss):
            print("Outlaw can hear a snake hissing close by!")
        elif isinstance(p, Stench):
            print("Outlaw can smell a Sheriff close by!")
        elif isinstance(p, Neigh):
            print("Outlaw can hear a Horse neighing nearby...")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialise Environment & Start Game
    wild_west = WildWest(outlaw_program, width=7, height=7)
    results = wild_west.run(search_type='bfs', default_moves=True)

    print("---------------------------------------------------")
    print(">> Game Results:")
    print("\tAgent Escaped: {}"
          "\n\tSteps Taken: {}"
          "\n\tTreasure Found: {} "
          "\n\tChest Found: {}"
          "\n\tGold Found: {}"
          "\n\tDog Found: {}"
          "\n\tHorse Found: {}"
          "\n\tBitten by Snake: {}"
          "\n\tSaved by Dog: {}"
          "\n\tBribed Sheriff {}"
          "\n\tLocked up by Sheriff: {}".format(
            results['agent_escaped'], results['agent_steps'],
            results['treasure_found'], results['chest_found'],
            results['agent_gold'], results['dog_found'],
            results['horse_found'], results['bitten_by_snake'],
            results['saved_by_dog'], results['agent_bribed'],
            results['agent_locked_up']))
    print(">> End game")
    print("---------------------------------------------------")
