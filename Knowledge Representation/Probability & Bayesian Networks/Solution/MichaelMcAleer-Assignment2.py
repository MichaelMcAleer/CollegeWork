from probability import *
import itertools

# Set shortened constants for True/False
T, F = True, False


def print_table(table):
    """
    Print a formatted table.

    :param table: (list) Nested list, each list represents a row in the table.
    """
    longest_cols = [
        (max([len(str(row[i])) for row in table]) + 3)
        for i in range(len(table[0]))
    ]
    row_format = "".join(
        ["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
    counter = 0
    for row in table:
        if counter == 0:
            dash = '-' * 48
            print(dash)
            print(row_format.format(*row))
            print(dash)
        else:
            print(row_format.format(*row))
        counter += 1


def q_one():
    """
    Calculate the probability of rolling snake eyes in dice, two consecutive
    rolls on the '1' side.
    """
    # Assign each side of a dice their probability
    p = ProbDist(freqs={'1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1})

    # Create joint probability distribution object with two die
    variables = ['die1', 'die2']
    j = JointProbDist(variables)

    # For two rolls of a die, get all possible combinations
    dice_sides = [1, 2, 3, 4, 5, 6]
    two_dice_rolls = [p for p in itertools.product(dice_sides, repeat=2)]

    # For each combination calculate the probability dependent on the
    # probability of each side of the dice from p, assign that combo and
    # probability to joint probability object
    for combo in two_dice_rolls:
        j[combo[0], combo[1]] = p[str(combo[0])] * p[str(combo[1])]

    # Output the probability of rolling two ones
    print("**Question 1**")
    print("Probability of Snake Eyes is: {}".format(j[1, 1]))


def q_two(bn):
    """
    Output CPT for being late and probability of various events.

    :param bn: (bayesNet) The complete Bayes network for being late.
    """
    # Calculate CPT for being late
    late_cpt = bn.variable_node('Late').cpt

    # Get headings and rows for CPT, output using print_table()
    headings = ['Rain', 'Traffic', 'Motorway', 'Probability Late']
    table = list()
    table.append(headings)
    # For each combination in the CPT, extract tuple values and assign
    # probability, append to table list for output to table
    for k, v in late_cpt.items():
        row = [str(k[0]), str(k[1]), str(k[2]), v]
        table.append(row)
    # Output table
    print("\n**Question 2**")
    print_table(table)

    # Calculate probability that the MotorWay was taken by invoking True
    # probability of variable node 'MotorWay'
    print("\nProbability you took the motorway: {}".format(
        bn.variable_node('MotorWay').p(T, 'MotorWay')))

    # Calculate probability the boss does not call if you are late by False
    # probability of variable node 'BossCalls' given that 'Late' is True
    print("Probability the boss does not call given that you are late: "
          "{}".format(bn.variable_node('BossCalls').p(F, {'Late': T})))

    # Calculate probability you are late then its raining, there is traffic,
    # and you took the motorway by invoking the variable node 'Late' given
    # that Rain/Traffic/MotorWay all resolve to True
    print("Probability you are late when it's raining & there is traffic as "
          "you took the motorway: {}".format(
            bn.variable_node('Late').p(T, {'Rain': T,
                                           'Traffic': T,
                                           'MotorWay': T})))


def q_three(bn):
    """
    Calculate by enumeration various probabilities of certain events.
    Probabilities are output as a percentage instead of 0-1 scale.

    :param bn: (bayesNet) The complete Bayes network for being late.
    """
    # Calculate the probability it is raining when the boss calls
    ans_dist_a = enumeration_ask('Rain', {'BossCalls': T}, bn)
    # Calculate the probability of traffic when the boss calls
    ans_dist_b = enumeration_ask('Traffic', {'BossCalls': T}, bn)
    # Calculate the probability of using the MotorWay when the boss calls
    ans_dist_c = enumeration_ask('MotorWay', {'BossCalls': T}, bn)
    # Calculate the probability it is raining and there is traffic when the
    # boss calls
    ans_dist_d = enumeration_ask('BossCalls', {'Rain': T, 'Traffic': T}, bn)

    print("\n**Question 3**")
    # Output pt1
    print("It is raining when the bass calls {:.3f}% of the time".format(
        ans_dist_a[T] * 100))
    print("It is raining when the bass calls {:.3f}% of the time".format(
        ans_dist_a[F] * 100))

    # Output pt2
    print("\nThere is traffic when the Boss calls {:.3f}% of the time".format(
        ans_dist_b[T] * 100))
    print("There is traffic when the Boss calls {:.3f}% of the time".format(
        ans_dist_b[F] * 100))

    # Output pt3
    print("\nI am using the Motorway when the Boss calls around {:.3f}% of "
          "the time".format(ans_dist_c[T] * 100))
    print("I am not using the Motorway when the Boss calls around {:.3f}% of "
          "the time".format(ans_dist_c[F] * 100))

    # Output pt4
    print("\nThe Boss calls when it is raining and there is Traffic around "
          "{:.3f}% of the time".format(ans_dist_d[T] * 100))
    print("The Boss does not calls when it is raining and there is Traffic "
          "around {:.3f}% of the time".format(ans_dist_d[F] * 100))


# Construct the Bayes Network, parent nodes first to be defined, then all
# subsequent children
workCommute = BayesNet([
    ('Rain', '', 0.41),
    ('Traffic', '', 0.15),
    ('MotorWay', '', 0.01),
    ('Late', 'Rain Traffic MotorWay',
     {(T, T, T): 0.8,
      (T, T, F): 0.98,
      (T, F, T): 0.2,
      (T, F, F): 0.3,
      (F, T, T): 0.25,
      (F, T, F): 0.24,
      (F, F, T): 0.001,
      (F, F, F): 0.05}),
    ('BossCalls', 'Late', {T: 0.8, F: 0.1})
])

# Call three question parts
q_one()
q_two(workCommute)
q_three(workCommute)
