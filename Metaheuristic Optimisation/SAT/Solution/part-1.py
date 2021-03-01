import csv
import random
import sys
import time

CNF_PROBLEM = None


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

    f_name = 'Part1_GSAT_TabuSearch.txt'
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
            filename = ('Part1_GSAT_TabuSearch.csv'.format())
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
    Determine the correctness of the current solution.

    :param instance: (nested list) The original variables and clause values.
    :param solution: (dict) The current solution.
    :return: (boolean & int) Is the solution successful, unsatisfied clause
    count.
    """
    clause = instance[1]
    unsat_clause = 0
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
        # if clause_status remains False, clause is unsatisfied
        if not clause_status:
            unsat_clause += 1
    # If there are unsat clauses return false and the count
    if unsat_clause > 0:
        return False, unsat_clause
    # Else return true and the zero count
    return True, unsat_clause


def generate_random_solution(num_variables):
    """
    For a given number of variables, generate a random solution with True/False
    values.

    :param num_variables: (int) The number of random T/F variables required.
    :return: (dict) The randomly generated solution.
    """
    rand_sol_list = {}
    # Start count at 1 to start variable assignment with
    var_num = 1
    while var_num < (num_variables + 1):
        # Assign the current variable count value a random 0/1 values
        rand_sol_list[var_num] = random.choice((0, 1))
        var_num += 1
    return rand_sol_list


def tabu_control(tabu_dict, var=None):
    """
    Control the tabu list (dictionary) operation.  If no variable kwarg is
    provided, the operation will decrement all current variables values by 1
    until 0 is reached so no variable lasts on the list for more than 5 steps.
    If a variable kwarg is provided, only add variable to list as key and start
    counter at 5.

    :param tabu_dict: (dict) The existing tabu dictionary.
    :param var: (int) The variable to be added to the tabu list, default=None.
    :return: (dict) The updated tabu dictionary.
    """
    # If tabu dict is provided by not a variable to add.
    if tabu_dict and not var:
        # For each of the existing keys, decrement value by 1
        for k, v in tabu_dict.items():
            tabu_dict[k] -= 1
        # Efficient removal of 0 value keys without triggering 3.x RunTimeError
        # in for loop
        tabu_dict = {k: v for k, v in tabu_dict.items() if v != 0}
    # Else if variable has been included, add variable to dictionary
    elif var:
        tabu_dict[var] = 5
    return tabu_dict


def run_gsat_tabu_search(file_in):
    """
    The main GSAT with Tabu Search algorithm. Operates with using 10 restarts
    with 1000 iterations per restart.

    :param file_in: (string) The path to the CNF SAT problem.
    :return: (dict) Run-time results for solving the CNF problem with GSAT and
    Tabu Search.
    """
    # Read in the CNF SAT problem
    sat_problem = read_instance(file_in)
    # Generate the random solution
    sol = generate_random_solution(len(sat_problem[0]))
    # Determine the success of the random solution and the amount of
    # unsatisfied clauses
    success, result = solution_status(sat_problem, sol)
    # Initialise Tabu List & iteration/restart counters
    tabu_dict = dict()
    restart_count = 0
    iteration_count = 0
    # Set best improvement tuple: Variable, T/F (1/0), current unsat clauses
    iteration_best_improvement = (None, None, float('inf'))
    start_time = None
    end_time = None
    # Whilst the SAT problem has not been successfully solved, keep iterating
    # through GSAT algorithm
    while not success:
        # Start Timer
        start_time = time.time()
        # Decrement all variables in the tabu dictionary by 1 if any present
        # and increment iteration counter by 1
        tabu_dict = tabu_control(tabu_dict)
        iteration_count += 1
        # Initialise the found improvement bool to false
        improvement = False
        # For each key/value in the current solution
        for k, v in sol.items():
            # Create a copy of the current solution
            temp_sol = sol.copy()
            # Flip the current variable value (T=>F, F=>%)
            temp_sol[k] = 0 if v else 1
            # Determine the amount of unsat clauses after the variable flip
            temp_success, temp_result = solution_status(sat_problem, temp_sol)
            # The the current result matches or bests the current best
            # improvement, set that as the iteration best improvement
            if (temp_result < iteration_best_improvement[2]) and (
                    k not in tabu_dict.keys()):
                iteration_best_improvement = (k, temp_sol[k], temp_result)
                improvement = True

        # If an improvement has been found, apply the change to the main
        # solution
        if improvement:
            # Apply the best variable flip to current solution
            sol[iteration_best_improvement[0]] = iteration_best_improvement[1]
            # Add the flipped variable to the tabu list so it can't be flipped
            # again until 5 other flips have been carried out
            tabu_dict = tabu_control(tabu_dict, iteration_best_improvement[0])
            # Check the success of the current solution
            success, sol_result = solution_status(sat_problem, sol)
            if success:
                end_time = time.time()

        # If the current iteration count is 1000, generate new random solution
        # if the restart count is less than 10. If restart count not less than
        # 10, break out of the iteration/restart loop entirely to end run
        if iteration_count == 1000:
            if restart_count >= 10:
                break
            else:
                sol = generate_random_solution(len(sat_problem[0]))
                success, result = solution_status(sat_problem, sol)
                iteration_best_improvement = (None, None, result)
                # Restart counters & reset tabu dict to empty
                restart_count += 1
                iteration_count = 0
                tabu_dict = dict()

    # If the GSAT algorithm has been successful in solving the SAT problem,
    # end run
    if success:
        success, result = solution_status(sat_problem, sol)
        search_steps = iteration_count
        search_time = end_time - start_time

        return success, sol, search_steps, search_time
    # Else no successful solution was found before 1000 iterations & 10
    # restarts
    else:
        end_time = time.time()
        search_steps = iteration_count
        search_time = end_time - start_time
        return success, sol, search_steps, search_time


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
    output_solution_to_csv(['Success', 'Search Steps', 'Search Time'])
    # Run the GSAT Tabu search algorithm output, results to file
    sat_success, sat_solution, sat_search_steps, sat_search_time = (
        run_gsat_tabu_search(file_input))

    if write_to_file:
        output_solution_to_file(sat_success, sat_solution,
                                sat_search_steps,
                                sat_search_time)
    if write_to_csv:
        output_solution_to_csv(
            [sat_success, sat_search_steps, sat_search_time])
