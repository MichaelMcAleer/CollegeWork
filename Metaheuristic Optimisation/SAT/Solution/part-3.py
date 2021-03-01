import csv
import math
import random
import sys
import time
from multiprocessing.pool import ThreadPool

THREADS = 3000


def read_instance(file_location):
    """
    Read TSP city info from a file.

    :param file_location: (string) The location of the file relative to the
    current file location.
    :return: (dict) All the cities with ID as key and tuple x,y co-ordinates.
    """
    file = open(file_location, 'r')
    file_size = int(file.readline())
    inst = {}
    for line in file:
        (city_id, x, y) = line.split()
        inst[int(city_id)] = int(x), int(y)
    file.close()
    return inst, file_size


def write_distance_to_file(result):
    """
    Write distance/cost to file. Add while loop and try/except clause to ensure
    data is always written to CSV about the current run (there were sporadic
    write errors coming back, known issue in Python when files are accessed and
    written to in quick succession).

    :param result: (string) The result of the calculate distance function.
    """
    written = False
    while not written:
        try:
            filename = 'ILS_Results.csv'
            my_file = open(filename, 'a+', newline='')
            with my_file:
                writer = csv.writer(my_file,
                                    delimiter=',',
                                    quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerow([result])
            my_file.close()
            written = True
        except PermissionError:
            pass


def generate_random_tsp_solution(data):
    """
    Shuffle the list of city IDs to get a random solution to the TSP problem.

    :param data: (dict) The TSP city data read from the file.
    :return: (list) The random TSP solution.
    """
    # Generate a list using all the City IDs
    city_route = list(data.keys())
    # Shuffle the City IDs to produce the random solution and return
    random.shuffle(city_route)
    return city_route


def generate_nearest_neighbour_solution(data):
    """
    Select a random city ID and from there calculate the nearest-neighbour
    TSP solution. Adapted from lab-1.

    :param data: (dict) The TSP city data read from the file.
    :return: (list) The nearest-neighbour TSP solution.
    """
    # Generate a list using all the City IDs
    cities = list(data.keys())
    # Select a random City ID index in the range of the length of the list
    # created previously
    city_index = random.randint(0, len(data) - 1)
    # The first city in the solution is the city at the randomly selected index
    solution = [cities[city_index]]
    # Delete that city from the initial list of cities so it isn't selected
    # again
    del cities[city_index]
    # Set current city as the first city in the list to begin with
    current_city = solution[0]
    # While there are still cities remaining in the initial city list
    while len(cities) > 0:
        # Get the next city ID in the list and calculate the euclidean distance
        # between the current city and the next city
        next_city = cities[0]
        next_city_cost = euclidean_distance(data[current_city],
                                            data[next_city])
        # Initialise next city index variable
        next_city_index = 0
        # For indexes in the remaining city list
        for city_index in range(0, len(cities)):
            # Get the city and the euclidean distance between it and the
            # current city, if there is an improvement in the distance (cost),
            # set city to next city
            city = cities[city_index]
            cost = euclidean_distance(data[current_city], data[city])
            if next_city_cost > cost:
                next_city_cost = cost
                next_city = city
                next_city_index = city_index
        # For next loop set the next city as the current city to start with
        # next nearest neighbour selection
        current_city = next_city
        # Append the new current city to the nearest neighbour solution
        solution.append(current_city)
        # Delete the newly selected current city from existing city list
        del cities[next_city_index]

    # Once all cities have been selected return the solution
    return solution


def euclidean_distance(city1, city2):
    """
    Calculate distance between two cities. Adapted from lab-1.

    :param city1: (tuple) City one co-ordinates
    :param city2: (tuple) City two co-ordinates
    :return: (float) The euclidean distance between the two cities
    """
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def calculate_route_distance(data, route):
    """
    Calculate the cost of the entire tour. Adapted from lab-1.

    :param data: (dict) The TSP city data read from the file.
    :param route: (list) The TSP solution.
    :return: (float) The distance/cost of the entire tour.
    """
    # Get the distance between the first and last stops in route
    city_a = data.get(route[0])
    city_z = data.get(route[len(route) - 1])
    fitness = euclidean_distance(city_a, city_z)

    # Get the distance between all cities between first and last in route
    for i in range(0, len(route) - 1):
        city_a = data.get(route[i])
        city_b = data.get(route[i + 1])
        fitness += euclidean_distance(city_a, city_b)

    # Once all city distances in tour have been calculated return the total
    return fitness


def select_cities(tour):
    """
    Select a random edge and determine all possible combinations with two other
    random edges in the tour.
    :param tour: (list) The TSP solution.
    :return: (nested-list) All possible combinations.
    """
    # Select a random city index ID to determine first edge selection
    random_city = random.randrange(0, (len(tour)))

    def _all_edge_variations(tour_length):
        # Where the random city is in three for loop using range specifiers,
        # return that combination as a valid edge + 2 possibility. If the K
        # value selected is the last city in the tour, the edge will use city 0
        return ([[i, i + 1],
                 [j, j + 1],
                 [k, 0 if k == tour_length - 1 else k + 1]]
                for i in range(tour_length - 1)
                for j in range(i + 2, (tour_length - 1))
                for k in range(j + 2, (tour_length - 1) + (i > 0))
                if random_city in [i, j, k])

    # Return all possible combinations for the length of the tour
    return _all_edge_variations(len(tour) - 1)


def get_route_permutations(three_opt_cities):
    """
    Given a combination of three edges, return all possible 3-opt moves.

    :param three_opt_cities: (nested-list) The three edges.
    :return: (dict) All possible 3-opt moves.
    """
    # Flatten the nested city list and extract all city IDs
    cities = [city for route in three_opt_cities for city in route]
    a, b = cities[0], cities[1]
    c, d = cities[2], cities[3]
    e, f = cities[4], cities[5]

    # Return a dict with all possible 2/3-opt moves including the original
    # AB|CD|EF combination
    return {0: [[a, b], [c, d], [e, f]],
            1: [[a, d], [e, b], [c, f]],
            2: [[a, b], [c, e], [d, f]],
            3: [[a, d], [e, c], [b, f]],
            4: [[a, e], [b, d], [c, f]],
            5: [[a, c], [b, d], [e, f]],
            6: [[a, e], [d, c], [b, f]],
            7: [[a, c], [b, e], [d, f]]}


def calculate_three_edge_distance(raw_data, route, three_edge_indexes):
    """
    Calculate the distance between the cities in three edges.

    :param raw_data: (dict) The city IDs and respective x-y co-ordinates
    :param route: (list) The TSP solution
    :param three_edge_indexes:
    :return:
    """
    # Total distance variable to be returned from calculations
    total_edge_distance = 0

    # For each of the edges in the three edges
    for edge in three_edge_indexes:
        try:
            # Get the city positions from the raw data
            city_a_position = raw_data.get(route[edge[0]])
            city_b_position = raw_data.get(route[edge[1]])

            # Calculate the distance between the two cities
            edge_distance = euclidean_distance(city_a_position,
                                               city_b_position)

            # Add the edge distance to the total distance
            total_edge_distance += edge_distance
        except Exception as e:
            # There should always be three edges, if this is not the case exit
            # as it will have an adverse impact on cost calculations
            print(e, edge)
            sys.exit("There was an issue calculating the distance between the "
                     "three edges, exiting iterated local search algorithm.")

    return total_edge_distance


def make_3opt_change_to_route(route, three_opt_cities, key):
    """
    Given a desired K-opt change, make that change to the route. The original
    route is passed in, along with the three edges and the change to be made.

    :param route: (list) The original TSP route
    :param three_opt_cities: (nested list) The three edges comprised of six
    cities.
    :param key: (int) The key to specify which K-opt change to make.
    :return: (list) The updated TSP route
    """
    r = route
    # Flatten the nested city list and extract all city IDs
    cities = [city for perm in three_opt_cities for city in perm]
    a, b = cities[0], cities[1]
    c, d = cities[2], cities[3]
    e, f = cities[4], cities[5]

    # Given the supplied key, make the change to the TSP route using array
    # slicing functionality
    if key == 1:
        # AD EB CF
        return r[:a + 1] + r[d:e + 1] + r[b:c + 1] + r[f:]
    elif key == 2:
        # AB CE DF
        return r[:a + 1] + r[b:c + 1] + r[e:d - 1:-1] + r[f:]
    elif key == 3:
        # AD EC BF
        return r[:a + 1] + r[d:e + 1] + r[c:b - 1:-1] + r[f:]
    elif key == 4:
        # AE DB CF
        return r[:a + 1] + r[e:d - 1:-1] + r[b:c + 1] + r[f:]
    elif key == 5:
        # AC BD EF
        return r[:a + 1] + r[c:b - 1:-1] + r[d:e + 1] + r[f:]
    elif key == 6:
        # AE DC BF
        return r[:a + 1] + r[e:d - 1:-1] + r[c:b - 1:-1] + r[f:]
    elif key == 7:
        # AC BE DF
        return r[:a + 1] + r[c:b - 1:-1] + r[e:d - 1:-1] + r[f:]


def make_random_2opt_change_to_route(route):
    """
    For the pertubation phase, select two random edges to switch in the TSP
    route.

    :param route: (list) The original TSP route
    :return: (list) The updated TSP route
    """
    # Select the two cities to be the first cities of each edge
    a = random.randrange(1, len(route))
    b = random.randrange(1, len(route) + 1)

    # Whilst city B is within range of city A and possibly uses the same city
    # in its edge determination, select another city B
    while b in [a - 1, a, a + 1]:
        b = random.randrange(1, len(route))

    # Make the 2-opt change to the TSP route
    new_route = route[:]
    new_route[a:b] = route[b - 1:a - 1:-1]

    # Run check to ensure the length of the original route and the updated
    # route are the same, if not exit due to the impact on the success and
    # accuracy of the algorithm
    if len(new_route) == len(route):
        return new_route
    else:
        sys.exit("There was an issue performing the random 2-opt perturbation "
                 "move, exiting iterated local search algorithm.")


def run_tsp_local_search(raw_data, tsp_solution):
    """
    Run the local search algorithm on a given TSP solution, run for all
    possible edge + 2 combinations. Once complete, apply the best solution if
    one is found. This function makes use of Python's inbuilt multiprocessing
    module to run the inner calculation function on all possible combinations.

    :param raw_data: (dict) The city IDs and respective x-y co-ordinates
    :param tsp_solution: (list) The original TSP solution
    :return: (list) The updated TSP solution
    """
    improvement = True
    while improvement:
        improvement = False
        # Select a random edge and get all possible variations with two other
        # edges
        all_variations = select_cities(tsp_solution)

        def _inner_calculation(edge_variation):
            """
            The inner calculation function that controls the distance cost
            calculations of edge variation.
            :param edge_variation: (nested list) The three edges comprised of
            six cities.
            :return: (tuple) The best improvement, the best combination, and
            the key required to apply the best k-opt move on the TSP route.
            """
            # Initialise vars to hold best improvement results
            best_var, best_var_opt_key = None, None
            best_improv = 0
            # For the supplied three edges, get all possible k-opt changes for
            # those edges
            edge_permutations = get_route_permutations(edge_variation)
            # Calculate the ABCDEF distance of the original edge distance
            original_route = edge_permutations.get(0)
            orig_route_cost = calculate_three_edge_distance(
                raw_data, tsp_solution, original_route)

            # For the remaining possible edge permutations (the i value here
            # determines the key used for k-opt change later)
            for i in range(1, 8):
                # Calculate the distance between the cities in each edge
                edge_route_cost = calculate_three_edge_distance(
                    raw_data, tsp_solution, edge_permutations.get(i))
                # Determine the improvement in route cost, if any
                edge_improvement = orig_route_cost - edge_route_cost
                # If the improvement is better than the current best
                # improvement save this improvement run details to be returned
                if edge_improvement > best_improv:
                    best_improv = edge_improvement
                    best_var = edge_variation
                    best_var_opt_key = i
            # If a best improvement was found, return the details of the cost
            # improvement, the best variation of edges, and the permutation key
            # that resulted in that improvement
            if best_improv:
                return best_improv, best_var, best_var_opt_key

        # Initialise Python multiprocessing thread pool, set threads to user
        # specified value. Using the inner calculation method, pass in the
        # iterable all_variations list so it can be iterated through by the
        # specified amount of threads until complete. All results are saved in
        # a nested dictionary, this has post-processing carried out on it to
        # remove all None values which are returned when no improvement can be
        # found. The amount of threads used is default to global 3000 unless
        # overridden by user in system argument input
        pool = ThreadPool(THREADS)
        results = pool.map(_inner_calculation, all_variations)
        pool.close()
        pool.join()
        results = [x for x in results if x is not None]

        # Set the outer best variation variables so changes can be applied if
        # an improvement is found
        best_variation, best_variation_opt_key = None, None
        best_improvement = 0

        # For all results returned from the multiprocess inner_calculation,
        # iterate through each until the best possible improvement is found
        for result in results:
            distance_improvement = result[0]
            if distance_improvement > best_improvement:
                # If the current improvement is bested, make a temporary route
                # to ensure that the total length of the tour has not been
                # changed by the k-opt move
                temp_route = make_3opt_change_to_route(tsp_solution,
                                                       result[1],
                                                       result[2])
                if len(temp_route) == len(tsp_solution):
                    # If no route length change detected, store improvement
                    # details to be applied later
                    best_variation = result[1]
                    best_variation_opt_key = result[2]
                    improvement = True

        # If an improvement has been found...
        if improvement:
            # Apply that change to the main TSP solution
            tsp_solution = make_3opt_change_to_route(tsp_solution,
                                                     best_variation,
                                                     best_variation_opt_key)

            # Determine the current full tour cost, write to file for post-run
            # analysis
            distance = calculate_route_distance(raw_data, tsp_solution)
            write_distance_to_file(str(distance))

    return tsp_solution


def run_iterated_local_search(file_location, init_sol):
    """
    Run the full iterative local search algorithm with pertubation.

    :param file_location: (string) The location of the file with TSP city
    details.
    :param init_sol: (string) The type of the initial solution to be used,
    valid values are 'random' and 'nearest-neighbour'. Nearest-neighbour is the
    default.
    """
    # 1. Read in data
    raw_data, file_size = read_instance(file_location)

    # 2. Generate random solution or nearest neighbour sol (default)
    if init_sol == 'random':
        tsp_sol = generate_random_tsp_solution(raw_data)
    else:
        tsp_sol = generate_nearest_neighbour_solution(raw_data)

    # 3. Compute length of original solution
    orig_sol = calculate_route_distance(raw_data, tsp_sol)
    write_distance_to_file(str(orig_sol))

    # Start LS Timer
    ils_time_start = time.time()

    # 4. Run the local search algorithm
    best_sol = run_tsp_local_search(raw_data, tsp_sol)
    best_sol_cost = calculate_route_distance(raw_data, best_sol)

    # 5. Start first of 5 perturbation phases
    for i in range(0, 5):
        print("Pertubtion Phase {}".format(i + 1))
        # If the total run time has not exceeded 5 minutes, run the pertubation
        # phase
        if (time.time() - ils_time_start) < 300:
            # Make a random 2-opt change
            pert_tsp_sol_2opt = make_random_2opt_change_to_route(best_sol)
            # Run local search algorithm using the route output from the random
            # 2-opt move
            pert_tsp_sol = run_tsp_local_search(raw_data, pert_tsp_sol_2opt)
            # Calculate the distance of the route generated from pertubation
            pert_tsp_distance = calculate_route_distance(raw_data,
                                                         pert_tsp_sol)
            # With 5% chance, apply the change irregardless if there has been
            # an improvement in the total cost or not
            if random.random() <= 0.05:
                best_sol = pert_tsp_sol
            # Else only apply the if an improvment has been made to the overall
            # cost of the TSP route
            else:
                if pert_tsp_distance < best_sol_cost:
                    best_sol = pert_tsp_sol
        # 5 minute limit has been reached, exit iterative local search
        # algorithm and keep current best generated solution up until this
        # point
        else:
            print("Five minute limit has been reached")
            break

    # 6. Iterated local search complete, write results to file and output to
    # screen
    best_sol_cost = calculate_route_distance(raw_data, best_sol)
    print("Original route distance: {}".format(orig_sol))
    print("New route distance: {}".format(best_sol_cost))
    print("Improvement: {}".format(orig_sol - best_sol_cost))
    print("Route: {}".format(best_sol))


try:
    file_input = sys.argv[1]
except IndexError:
    sys.exit("You must provide a path to a TSP problem, exiting...")

try:
    initial_sol = sys.argv[2]
except IndexError:
    initial_sol = 'nearest_neighbour'

try:
    THREADS = int(sys.argv[3])
except IndexError:
    # Default global threads 3000 used
    pass

start = time.time()
run_iterated_local_search(file_input, initial_sol)
end = time.time()

print("Run Time: {}".format(end - start))
