try:
    from aima_src import *
except ImportError:
    from agents import *

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Classes built for 'things' in Outlaw 1-D game


class Dog(Thing):
    pass


class Chest(Thing):
    pass


class Gold(Thing):
    pass


class Snake(Thing):
    pass


class Sheriff(Thing):
    pass


class Horse(Thing):
    pass


# -----------------------------------------------------------------------------
# Outlaw - the agent of the game

class OutLaw(Agent):
    """
    The Outlaw agent, has a number of attributes to determine game performance
    """
    # Agent Attributes
    location = 0
    luck = random.random()
    # Agent states & performance measures
    life = 100
    gold = 0
    steps = 0
    dog = False
    found_Horse = False
    treasure_found = False
    bitten_by_snake = False
    escaped_by_horse = False
    chest_found = False
    saved_by_dog = False
    bribed_sheriff = False
    locked_up = False

    def move_down(self):
        """
        Move agent 1 cell down from current location
        """
        self.steps += 1
        self.location += 1

    def move_up(self):
        """
        Move agent 1 cell up from current location
        """
        self.steps += 1
        self.location -= 1

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
# WildWest - The 1-D game environment

class WildWest(Environment):

    def run(self, steps=1000):
        """
        Sub-class of environment run which has alterations to return agent
        performance measure.

        :param steps: (int) the max (ceiling) amount of steps the agent should
        perform, this eliminates the possibility of infinite runs or endless
        loop
        :return: (dict) the agent performance stats when game is done
        """
        for step in range(steps):
            if self.is_done():
                run_results = dict()
                for agent in self.agents:
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
            self.step()

    def percept(self, agent):
        """
        Percept things at current agent location
        :param agent: (agent) The environment agent
        :return: (list) The things at agent location, empty list if none
        """
        return self.list_things_at(agent.location)

    @staticmethod
    def _move_down(agent):
        """
        Move agent 1 cell down from current location. There is also control
        here for step (move) based logic in the environment for:
        -Agent dropping gold per step
        -Agent losing health per step
        -Dog finding anti-venom

        If the agent is at game boundary of 15 then do not move any further.

        :param agent: (agent) The environment agent
        """
        # If the agent is at the game boundary do not move past it
        if agent.location == 15:
            print("The Outlaw decided to move up at location {} to location: "
                  "{} but cannot because game boundary reached.".format(
                    agent.location, agent.location + 1))
            return

        print("The Outlaw decided to move down at location {} to location: "
              "{}".format(agent.location, agent.location + 1))

        # Control the reduction in Outlaw gold pieces value per step taken
        # only if the Outlaw has found the treasure and the value of gold
        # is above 0
        if agent.treasure_found and agent.gold > 0:
            # Minus 100 gold from total gold held
            agent.gold -= 100
            if agent.gold <= 0:
                # If the agent gold goes below zero, hard-code to zero value,
                # do not let fall into negative value
                agent.gold = 0
                print("The Outlaw has dropped all his gold!")
            else:
                # Else output the current amount of gold the agent holds
                print("100 pieces of gold dropped! Outlaw now has {} pieces "
                      "of gold!".format(agent.gold))

        # 50/50 chance of running this block of code for the dog going to
        # find anti-venom for the snakebite
        if agent.dog and agent.bitten_by_snake and random.random() >= 0.5:
            # If the Outlaw's luck is good, the dog will find the
            # anti-venom
            if random.random() < agent.luck:
                print("Man's best friend has found some anti-venom for "
                      "the snake bite! The venom no longer affects the "
                      "The Outlaw!")
                agent.bitten_by_snake = False

        # If the Outlaw has been bitten by a snake handle the health score
        # reduction, minus 10 health per move until the health is 0 then game
        # over
        if agent.bitten_by_snake:
            agent.life -= 10
            print("The Outlaw continues to feel the effects of the snake "
                  "bite and loses another 10 health points, current "
                  "health now at {}".format(agent.life))
            if agent.life == 0:
                print("The snake venom has killed the outlaw! Game Over!")
                agent.alive = False

        # Move the agent down one cell after all turn based logic is complete
        agent.move_down()

    @staticmethod
    def _move_up(agent):
        """
        Move agent 1 cell up from current location. There is also control
        here for step (move) based logic in the environment for:
        -Agent dropping gold per step
        -Agent losing health per step
        -Dog finding anti-venom

        If the agent is at game boundary of -15 then do not move any further.

        :param agent: (agent) The environment agent
        """
        # If the agent is at the game boundary do not move past it
        if agent.location == -15:
            print("The Outlaw decided to move up at location {} to location: "
                  "{} but cannot because game boundary reached.".format(
                    agent.location, agent.location - 1))
            return

        print("The Outlaw decided to move up at location {} to location: "
              "{}".format(agent.location, agent.location - 1))

        # Control the reduction in Outlaw gold pieces value per step taken
        # only if the Outlaw has found the treasure and the value of gold
        # is above 0
        if agent.treasure_found and agent.gold > 0:
            # Minus 100 gold from total gold held
            agent.gold -= 100
            if agent.gold <= 0:
                # If the agent gold goes below zero, hard-code to zero value,
                # do not let fall into negative value
                agent.gold = 0
                print("The Outlaw has dropped all his gold!")
            else:
                # Else output the current amount of gold the agent holds
                print("100 pieces of gold dropped! Outlaw now has {} pieces "
                      "of gold!".format(agent.gold))

        # 50/50 chance of running this block of code for the dog going to
        # find anti-venom for the snakebite
        if agent.dog and agent.bitten_by_snake and random.random() >= 0.5:
            # If the Outlaw's luck is good, the dog will find the
            # anti-venom
            if random.random() < agent.luck:
                print("Man's best friend has found some anti-venom for "
                      "the snake bite! The venom no longer affects the "
                      "The Outlaw!")
                agent.bitten_by_snake = False

        # If the Outlaw has been bitten by a snake handle the health score
        # reduction, minus 10 health per move until the health is 0 then game
        # over
        if agent.bitten_by_snake:
            agent.life -= 10
            print("The Outlaw continues to feel the effects of the snake "
                  "bite and loses another 10 health points, current "
                  "health now at {}".format(agent.life))
            if agent.life == 0:
                print("The snake venom has killed the outlaw! Game Over!")
                agent.alive = False

        # Move the agent down up cell after all turn based logic is complete
        agent.move_up()

    def _open_chest(self, agent):
        """
        The logic for opening a chest in the game environment
        :param agent: (agent) The environment agent
        """
        # List the things at the current agent location of class type chest,
        # if chests returned run check that agent can open chest, if True,
        # run chest open logic
        items = self.list_things_at(agent.location, tclass=Chest)
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
                    print("The Outlaw found a snake in the chest!")
                    # If the agent has found the dog, the dog will kill the
                    # snake
                    if agent.dog:
                        print("The Outlaw's trusty dog companion fought the "
                              "snake and killed it!")
                        agent.saved_by_dog = True
                    # Else the agent gets bitten by the snake, initialise
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
        items = self.list_things_at(agent.location, tclass=Gold)
        if len(items) != 0:
            if agent.find(items[0]):
                # Generate a random number between 1000 and 10000 for the
                # amount of gold the agent will find, add it to the agent gold
                # attribute and set treasure_found to True so agent can now
                # escape
                chest_gold = random.randint(1000, 10000)
                agent.gold += chest_gold
                agent.treasure_found = True
                print("The Outlaw found the treasure of {} pieces of gold at "
                      "location: {}".format(chest_gold,
                                            agent.location))
                self.delete_thing(items[0])
            # If the agent has already found the horse they can now escape the
            # environment and 'win' the game
            if agent.found_Horse:
                print("The Outlaw found the gold and escaped on the horse "
                      "with {} pieces of gold! Game Over!".format(agent.gold))
                # Set agent alive to 'False' so is_done() is triggered on
                # next agent 'step'
                agent.alive = False
                agent.escaped_by_horse = True
            # Else the agent has not yet found the horse to escape on, game
            # continues until horse is found
            else:
                print("Find the Horse to escape with the loot and watch "
                      "out for the Sheriff! There is a hole in the gold "
                      "bag, the Outlaw loses 100 gold pieces per step...")

    def _bitten_by_snake(self, agent):
        """
        The logic for getting bit by a snake in the game environment
        :param agent: (agent) The environment agent
        """
        # List the things at the current agent location of class type snake,
        # if snake returned run check that agent can bit by snake, if True,
        # run bit by snake logic
        items = self.list_things_at(agent.location, tclass=Snake)
        if len(items) != 0:
            if agent.bitten(items[0]):
                print("The Outlaw got cornered by an angry snake at "
                      "{}!".format(agent.location))
                # If the agent has rescues a dog in the game, the dog can kill
                # the snake and save the agent
                if agent.dog:
                    print("The Outlaw's trusty dog companion fought the snake "
                          "and killed it!")
                    agent.saved_by_dog = True
                    self.delete_thing(items[0])
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
        items = self.list_things_at(agent.location, tclass=Sheriff)
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
                        agent.alive = False
                        agent.locked_up = True
                    # Else the agent't luck is greater than the randomly
                    # generated number, sheriff takes bribe of half of the
                    # current gold the outlaw has in possession
                    else:
                        bribe_amount = int(agent.gold / 2)
                        agent.gold -= bribe_amount
                        agent.bribed_sheriff = True
                        print("The Outlaw is in luck, Sheriff accepted bribe "
                              "of {}".format(bribe_amount))
                        self.delete_thing(items[0])
                # Agent has no gold to attempt to bribe the sheriff with, gets
                # thrown in jail and game over
                else:
                    print("The Outlaw had no gold to bribe the Sheriff and "
                          "got thrown in jail...")
                    agent.alive = False
                    agent.locked_up = True
                    self.delete_thing(items[0])

    def _horse_escape(self, agent):
        """
        The logic for escaping on a horse in the game environment
        :param agent: (agent) The environment agent
        """
        # List the things at the current agent location of class type horse,
        # if horse returned run check that agent can escape on horse, if True,
        # run escape on horse logic
        items = self.list_things_at(agent.location, tclass=Horse)
        if len(items) != 0:
            if agent.escaped(items[0]):
                print("The Outlaw found the Horse to escape on at location: "
                      "{}".format(agent.location))
                # Agent has already found the gold, escape on horse
                if agent.treasure_found and agent.gold > 0:
                    print("The Outlaw escaped with {} gold pieces! Game "
                          "Over!".format(agent.gold))
                    agent.escaped_by_horse = True
                    agent.alive = False
                    self.delete_thing(items[0])
                # Agent has found the gold but has dropped it all
                elif agent.treasure_found and agent.gold == 0:
                    print("The Outlaw dropped all his gold but decides to "
                          "cut his losses because there is no treasure "
                          "left. The Outlaw escaped broke but free!")
                    agent.escaped_by_horse = True
                    agent.alive = False
                    self.delete_thing(items[0])
                # Else the gold has not been found, game continues until gold
                # is found
                else:
                    print("The Outlaw hasn't found any gold yet to escape on "
                          "the horse with, the search continues!")
                    agent.found_Horse = True
                    self.delete_thing(items[0])

    def _dog_rescue(self, agent):
        """
        The logic for rescuing the dog in the game environment
        :param agent: (agent) The environment agent
        """
        # List the things at the current agent location of class type dog,
        # if dog returned run check that agent can rescue dog, if True,
        # run rescue dog logic
        items = self.list_things_at(agent.location, tclass=Dog)
        if len(items) != 0:
            if agent.rescue(items[0]):
                print("The Outlaw rescued a dog at location: {}".format(
                    agent.location))
                agent.dog = True
                self.delete_thing(items[0])

    def execute_action(self, agent, action):
        """
        Change the state of the environment based on what the agent does.

        :param agent: (agent) The environment agent
        :param action: (string) The action for the agent to carry out
        """
        # Move down in environment
        if action == 'move down':
            self._move_down(agent)

        # Move up in environment
        elif action == 'move up':
            self._move_up(agent)

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

        elif action == 'escaped':
            self._horse_escape(agent)

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
    for p in percepts:
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

    return 'move down' if random.random() > 0.5 else 'move up'


if __name__ == "__main__":
    # Initialise the agent & environment, add agent to start position of 0
    wild_west = WildWest()
    outlaw = OutLaw(outlaw_program)
    wild_west.add_thing(outlaw, 0)
    # Initialise the environment 'things'
    things = [Dog(), Chest(), Chest(), Gold(),
              Snake(), Snake(), Sheriff(), Horse()]
    # Select 8 random positions within game boundary (-15 => 15) to place
    # 'things'
    positions = random.sample(range(-15, 15), 8)
    print("---------------------------------------------------")
    print(">> Setting up Wild West Environment")
    for t in things:
        wild_west.add_thing(t, positions[0])
        print("\t{} added to location {}".format(t, positions[0]))
        del positions[0]
    print("\tThe Outlaw's luck today is {}".format(outlaw.luck))
    print(">> Starting game")
    print("---------------------------------------------------")
    results = wild_west.run()
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
