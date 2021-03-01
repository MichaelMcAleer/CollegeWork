import csv
import random
import sys
import time

CNF_PROBLEM = None
# Set probability constants
NOVELTY_WP = 0.4
NOVELTY_P = 0.3


def read_instance(file_name):
    """
    Read the contents of a user-specified .CNF file.

    :param file_name: (string) Path to the SAT problem .CNF file
    :return: (list) Variable list and clauses from .CNF file
    """
    file = open(file_name, 'r')
    total_variables = -1
    total_clauses = -1
    clause = []
    variables = []
    current_clause = []

    for line in file:
        data = line.split()

        if len(data) == 0:
            continue
        if data[0] == 'c':
            continue
        if data[0] == 'p':
            total_variables = int(data[2])
            total_clauses = int(data[3])
            continue
        if data[0] == '%':
            break
        if total_variables == -1 or total_clauses == -1:
            print("Error, unexpected data")
            sys.exit(0)

        for var_i in data:
            literal = int(var_i)
            if literal == 0:
                clause.append(current_clause)
                current_clause = []
                continue
            var = literal
            if var < 0:
                var = -var
            if var not in variables:
                variables.append(var)
            current_clause.append(literal)

    if total_variables != len(variables):
        print("Unexpected number of variables in the problem")
        print("Variables", total_variables, "len: ", len(variables))
        print(variables)
        sys.exit(0)
    if total_clauses != len(clause):
        print("Unexpected number of clauses in the problem")
        sys.exit(0)
    file.close()
    return [variables, clause]


def output_solution_to_file(success, solution, search_steps, search_time):
    """
    Write solution details to text file.

    :param success: (bool) If the algorithm was successful
    :param solution: (dict) The current key/value solution for the SAT problem.
    :param search_steps: (int) The amount of steps required to solve the SAT
    problem.
    :param search_time: (float) The amount of time required to solve the SAT
    problem.
    """

    def _write_sol_to_file(file_name, write_type, result):
        """
        The main file writing inner function.
        """
        written = False
        while not written:
            try:
                f = open(file_name, write_type)
                f.write(result)
                f.close()
                written = True
            except PermissionError:
                print("Write Error - Retrying")
                pass

    f_name = 'Part2_Novelty+.txt'
    _write_sol_to_file(f_name, 'w+', '\nc')
    _write_sol_to_file(f_name, 'a+', '\nc Solution for {}'.format(CNF_PROBLEM))
    _write_sol_to_file(f_name, 'a+', '\nc Solution success {}'.format(success))
    _write_sol_to_file(f_name, 'a+', '\nc')

    sol_list = list()
    for k, v in solution.items():
        if v:
            sol_list.append(str(k))
        else:
            sol_list.append(str('-{}'.format(k)))

    while sol_list:
        sol_line = sol_list[:10]
        sol_list = sol_list[10:]
        _write_sol_to_file(f_name, 'a+', '\nv {}'.format(str(sol_line)))

    _write_sol_to_file(f_name, 'a+', '\nc')
    _write_sol_to_file(f_name, 'a+', '\nc Steps: {}'.format(search_steps))
    _write_sol_to_file(f_name, 'a+', '\nc Time: {}'.format(search_time))


def output_solution_to_csv(result):
    """
    Write run result details to CSV file.

    :param result (list) A list to be written to CSV, each line represents a
    run of the GSAT Tabu Search algorithm.
    """
    written = False
    while not written:
        try:
            filename = ('Part2_Novelty+.csv'.format())
            my_file = open(filename, 'a+', newline='')
            with my_file:
                writer = csv.writer(my_file,
                                    delimiter=',',
                                    quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerow(result)
            my_file.close()
            written = True
        except PermissionError:
            print("Write Error - Retrying")
            pass


def solution_status(instance, solution):
    """
    Given a solution to a SAT problem, determine if the solution solves the
    problem.

    :param instance: (list) The return value from read_instance() containing
    variables and clauses from user-specified .CNF file
    :param solution: (dict) The solution for the SAT problem
    :return: (boolean & list) If the solution satisfies the problem and the
    amount of unsatisfied clauses. If problem satisfied, the list will be empty
    """
    clause = instance[1]
    unsat_clause = 0
    unsat_clause_list = list()
    for clause_i in clause:
        clause_status = False
        for var in clause_i:
            # For variable in clause, check if one of the clauses will evaluate
            # to True, if so, set the clause_status to True
            if var < 0:
                # If that variable is les than zero, get the value of the
                # variable from the list, double negative var to get plus and
                # retrieve from list by key.
                if (1 - solution[-var]) == 1:
                    # If 1 - 0 (double negative value) == 1, var is true,
                    # clause will resolve as true
                    clause_status = True
            else:
                if solution[var] == 1:
                    # Variable true so clause will automatically resolve to
                    # True
                    clause_status = True
        # if clause_status remains False, clause is unsatisfied, append clause
        # to list for return
        if not clause_status:
            unsat_clause += 1
            unsat_clause_list.append(clause_i)
    # If there are unsat clauses return false and all unsat clauses
    if unsat_clause > 0:
        return False, unsat_clause_list
    # Else return true and an empty list
    return True, unsat_clause_list


def generate_random_solution(num_variables):
    """
    Generate a random solution for the SAT problem by randomising values of all
    variables.

    :param num_variables: (int) The amount of variables that need to be
    randomised
    :return: (dict) All variables and their 1/0 (True/False) designation
    """
    # Initialise the random solution dict
    rand_sol_list = {}
    # Start the variable designation at 1
    var_num = 1
    # While there are still variables to be designated with 1/0
    while var_num <= num_variables:
        # Assign the current variable with a random choice of 1/0
        rand_sol_list[var_num] = random.choice((0, 1))
        # Increment counter
        var_num += 1
    return rand_sol_list


def run_novelty_plus_sat_solver(file_in):
    """
    Given a SAT problem file, solve the SAT problem using Novelty+ SAT solving
    technique.  The probabilities are defined as constants at the top of the
    file. Novelty+ algorithm will run until solution to SAT problem is found or
    100000 iterations has been reached, whichever comes first.

    :param file_in: (string) Path to the SAT problem .CNF file
    :return: (list) Results from the Novelty+ algorithm run. Results include in
    this order: success (boolean), solution (dict), steps taken (int),
    algorithm run time (float), unsatisfied clauses (int)
    """
    # Read in SAT problem from file
    sat_problem = read_instance(file_in)
    # Generate a random solution for each of the variables in SAT problem
    sol = generate_random_solution(len(sat_problem[0]))
    # Get the success of random SAT solution and amount of unsatisfied clauses
    success, result = solution_status(sat_problem, sol)
    # Initialise iteration counter
    iteration_count = 1
    # Start the process timer
    start_time = time.time()
    # Initialise the Tabu list
    tabu_dict = dict()
    # While the SAT problem remains unsolved, the iteration counter is less
    # than 100000, and the tune time is less than 60s continue to attempt to
    # solve SAT problem
    while not success and iteration_count <= 100000 and (
            (time.time() - start_time) <= 60):
        # Select random clause from unsat clauses
        r_clause = random.choice(result)
        # WP Random selection, if less than or equal to 0.4, use random
        # selection for flip
        if random.random() <= NOVELTY_WP:
            # Select a random variable from the clause
            r_var = random.choice(r_clause)
            # If the random var is negative set as the positive equivalent
            if r_var < 0:
                r_var = -r_var
            # Flip the variable value
            sol[r_var] = 1 - sol[r_var]
            # Add the flipped variable to the tabu dict
            tabu_dict[str(r_clause)] = r_var
            # Using new solution, get the success and unsat clauses results
            success, result = solution_status(sat_problem, sol)
        else:
            # Initialise Iteration Best Improvement (IBI) tuples to store
            # information on best and 2nd best variable flip.
            # Format: Variable, T/F (1/0), current unsat clauses
            ibi_1st = (None, None, float('inf'))
            ibi_2nd = (None, None, float('inf'))
            # Flip each variable and determine the improvement of unsat clauses
            # Add best improvement and 2nd best to above tuples
            for var in r_clause:
                # Make a copy of the current solution so it remains unchanged
                # during best variable determination
                temp_sol = sol.copy()
                # If the random var is negative set as the positive equivalent
                if var < 0:
                    var = -var
                # Flip the value of the variable from 0 to 1 or vice versa
                temp_sol[var] = 1 - temp_sol[var]
                # Determine the effectiveness of flipping this variable value
                temp_success, temp_result = solution_status(
                    sat_problem, temp_sol)
                # If the amount of unsat clauses is less than the current
                # lowest value found by other clause variables
                if len(temp_result) < ibi_1st[2]:
                    # Set the old best as the second best
                    ibi_2nd = ibi_1st
                    # Set current change as the best
                    ibi_1st = (var, temp_sol[var], len(temp_result))
                # Else the amount of unsat clauses isn't as good as the current
                # best but beats the second best amount of unsat clauses
                elif ibi_1st[2] <= len(temp_result) < ibi_2nd[2]:
                    ibi_2nd = (var, temp_sol[var], len(temp_result))

            # If best solution is not the most recently flipped variable apply
            # that change
            if str(r_clause) in tabu_dict:
                if ibi_1st[1] != tabu_dict[str(r_clause)]:
                    sol[ibi_1st[0]] = ibi_1st[1]
                    tabu_dict[str(r_clause)] = ibi_1st[1]

            else:
                # If a random float between 0 and 1 is less than or equal to
                # the NOVELTY_P value of 0.3, then apply the second best
                # improvement
                if random.random() <= NOVELTY_P:
                    sol[ibi_2nd[0]] = ibi_2nd[1]
                # Else the random value is greater than NOVELTY_P value, apply
                # the best improvement
                else:
                    sol[ibi_1st[0]] = ibi_1st[1]
        # Increment the run iteration counter
        iteration_count += 1

    # Run complete either by success or max iterations reached
    # End the process timer
    run_time = time.time() - start_time
    # Return the results, either the SAT problem has been solved or max
    # iterations has been reached
    return success, sol, iteration_count, run_time, len(result)


if __name__ == '__main__':
    # Path to CNF Sat file is not an optional argument
    try:
        file_input = sys.argv[1]
    except IndexError:
        sys.exit("You must provide a path to a CNF sat problem, exiting...")

    # If user has not specified to write to file and CSV, default to True
    try:
        write_to_file = sys.argv[2]
    except IndexError:
        write_to_file = True
    try:
        write_to_csv = sys.argv[3]
    except IndexError:
        write_to_csv = True

    CNF_PROBLEM = file_input

    # Run the GSAT Tabu search algorithm output, results to file
    (sat_success, sat_solution, sat_search_steps, sat_search_time,
     unsat_clauses) = run_novelty_plus_sat_solver(file_input)

    if write_to_file:
        output_solution_to_file(sat_success, sat_solution,
                                sat_search_steps,
                                sat_search_time)
    if write_to_csv:
        output_solution_to_csv(
            ['Success', 'Search Steps', 'Search Time', 'Unsat Clauses'])
        output_solution_to_csv(
            [sat_success, sat_search_steps, sat_search_time,
             unsat_clauses])
