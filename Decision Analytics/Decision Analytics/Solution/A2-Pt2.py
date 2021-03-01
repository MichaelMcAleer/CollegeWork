# Decision Analytics
# Assignment 2: Linear Programming - Part 2
# Michael McAleer - R00143621
import pandas as pd
from ortools.linear_solver import pywraplp

"""
A - Load the input data from the file 'Assignment_DA_2_b_data.xlsx'
"""
raw_data = 'Assignment_DA_2_b_data.xlsx'

flight_schedule = pd.read_excel(
    raw_data, sheet_name='Flight schedule', index_col=0)

taxi_distances = pd.read_excel(
    raw_data, sheet_name='Taxi distances', index_col=0)

terminal_capacity = pd.read_excel(
    raw_data, sheet_name='Terminal capacity', index_col=0)

# Extract actors from dataset into sets of unique values
flights = list(sorted(flight_schedule.index))
runways = list(sorted(taxi_distances.index))
terminals = list(sorted(taxi_distances.columns))

# Process arrival times to produce set of unique times, and to get an ordered
# list of all times
arrival_times_set = set(flight_schedule['Arrival'].to_list())
arrival_times_list = list()
for a_time in arrival_times_set:
    arrival_times_list.append(str(a_time))
arrival_times_list = sorted(arrival_times_list)

# Process departure times to produce set of unique times, and to get an ordered
# list of all times
departure_times_set = set(flight_schedule['Departure'].to_list())
departure_times_list = list()
for d_time in departure_times_set:
    departure_times_list.append(str(d_time))
departure_times_list = sorted(departure_times_list)

# Get the union of both arrival and departure times, get an ordered list of all
# unique times, this will be used for terminal occupancy
combined_times_set = arrival_times_set.union(departure_times_set)
combined_times_list = list()
for c_time in combined_times_set:
    combined_times_list.append(str(c_time))
combined_times_list = sorted(combined_times_list)

"""
B - Identify the decision variables of the problem for the Linear Programming 
model.

Note: For all decision variables binary integer values x{0,1} are used, these
will indicate selection of that option as True/False.

We need to decide what runway a given flight should land on at its associated
arrival time.

    Let W^tfr be the time {t=1->20} flight f {f=1->26} takes taxi route between
    runway and terminal r {r=1->9}. It is modelled this way so the decision 
    variables have an associated cost, a runway alone has no cost it is the 
    taxi route that has an associated cost.
    
                    -------------------------
                              Time
        -------     -------------------------    ------
        Flight      t1   | t2   | .... | tD      Taxi Route
        -------     -------------------------    ------
           f A   |  W111 + W211 + .... + WD11  |  rA tA
           f A   |  W112 + W212 + .... + WD12  |  rA tB
           f A   |  W113 + W213 + .... + WD13  |  rA tC
        
           f A   |  W114 + W214 + .... + WD14  |  rB tA
           f A   |  W115 + W215 + .... + WD15  |  rB tB
           f A   |  W116 + W216 + .... + WD16  |  rB tC
        
           f A   |  W117 + W217 + .... + WD17  |  rC tA
           f A   |  W118 + W218 + .... + WD18  |  rC tB
           f A   |  W119 + W219 + .... + WD19  |  rC tC
        -------     -------------------------    ------
           f B   |  W121 + W221 + .... + WD21  |  rA tA
           f B   |  W122 + W222 + .... + WD22  |  rA tB
           f B   |  W123 + W223 + .... + WD23  |  rA tC
        
           f B   |  W124 + W224 + .... + WD24  |  rB tA
           f B   |  W125 + W225 + .... + WD25  |  rB tB
           f B   |  W126 + W226 + .... + WD26  |  rB tC
        
           f B   |  W127 + W227 + .... + WD27  |  rC tA
           f B   |  W128 + W228 + .... + WD28  |  rC tB
           f B   |  W129 + W229 + .... + WD29  |  rC tC
        -------     -------------------------    ------
           ...   |  .... + .... + .... + ....  |  ...
        -------     -------------------------    ------
           f Z   |  W1Z1 + W3Z1 + .... + WDZ1  |  rA tA
           f Z   |  W1Z2 + W3Z2 + .... + WDZ2  |  rA tB
           f Z   |  W1Z3 + W3Z3 + .... + WDZ3  |  rA tC
        
           f Z   |  W1Z4 + W224 + .... + WDZ4  |  rB tA
           f Z   |  W1Z5 + W225 + .... + WDZ5  |  rB tB
           f Z   |  W1Z6 + W226 + .... + WDZ6  |  rB tC
        
           f Z   |  W1Z7 + W227 + .... + WDZ7  |  rC tA
           f Z   |  W1Z8 + W228 + .... + WDZ8  |  rC tB
           f Z   |  W1Z9 + W229 + .... + WDZ9  |  rC tC
        -------     -------------------------    ------
    
We need to determine what terminal a flight should go to between arriving and
derparting.

    To model this a combined list of all times, both arriving and departing, so
    a full sequence of times is available, these will indicate at which times 
    during the flight timeetable a given terminal is occupied.

    Let X^abc be the time a {a=1->31} terminal b {b=1,2,3} is occupied by 
    flight c {c=1->26}
    
                    -------------------------
                               Time
        -------     -------------------------    --------
         Flight     t1   | t2   | .... | tD      Terminal
        -------     -------------------------    --------
             A   |  X111 + X211 + .... + XD11  |   A
             A   |  X112 + X212 + .... + XD12  |   B
             A   |  X113 + X213 + .... + XD13  |   C
        -------     -------------------------    --------
             B   |  X121 + X221 + .... + XD21  |   A
             B   |  X122 + X222 + .... + XD22  |   B
             B   |  X123 + X223 + .... + XD23  |   C
        -------     -------------------------    --------
           ...   |  .... + .... + .... + ....  |  ...
        -------     -------------------------    --------
             Z   |  X1Z1 + X3Z1 + .... + XDZ1  |   A
             Z   |  X1Z2 + X3Z2 + .... + XDZ2  |   B
             Z   |  X1Z3 + X3Z3 + .... + XDZ3  |   C
        -------     -------------------------    --------

We need to decide what runway a given flight should leave on at its associated
departure time.

    Let Y^tfr be the time {t=1->20} flight f {f=1->26} lands on runway 
    r {r=1,2,3}
    
        (The table layout for this is the same as that for the arrival
        decision variables)

C - Identify the constraints of the problem for the Linear Programming model
    
    - Only one flight can occupy a runway at a time
                
        The combined sum of all flights for a given runway for a given time 
        must not exceed 1 - example, on runway A at 08:00 only one flight can 
        use that runway
            
            W111 + W112 + W113 + ... + W1Z9 ≤ 1
        
    - Flights must arrive and depart at their designated time
        
        The combined sum of each runway for that flight and time must equal 
        one
            
            W111 + W112 + W113 + ... + W119 = 1 
        
    - Flights must use the correct terminal for their runway/terminal taxi
        
        For each flight and terminal, we want to ensure that if any of the
        terminal/runway combinations are True then the associated Terminal
        is set as True in the terminal schedule, as this is a logical OR
        constraint it can be modelled in a linear constraint as follows:
              y = x1 ∨ x2 ∨ ... V xN
          becomes...
              y ≤ x1 + x2 + ... + xN
              y ≥ x1
              y ≥ x2
              y ≥ xN
              0 ≤ y ≤1
        Where y is the decision variable from the terminal schedule and xN
        is the list of terminal/runway decision variables
        
        Example:
            X111 = W111 ∨ W114 ∨ W117
            X111 ≤ W111 + W114 + W117
            X111 ≥ W111
            X111 ≥ W114
            X111 ≥ W117
            0 ≤ X111 ≤1
           
    - Flights must depart from the correct terminal/runway combination 
      dependent on the terminal they stayed at
        
        This is modelled the same as the previous constraint using the logical
        OR constraint, the y variable is still the decision variable from the
        terminal schedule, the OR comparison variables are instead the flight
        departure intervals for all runway combinations for a given flight and
        terminal.

        Example:
            X111 = Y111 ∨ Y114 ∨ Y117
            X111 ≤ Y111 + Y114 + Y117
            X111 ≥ Y111
            X111 ≥ Y114
            X111 ≥ Y117
            0 ≤ X111 ≤1
        
    - Flights must stay at the same terminal for the duration of their stay
        
        Get the arrival time, departure time, and all intervals in between.
        We use the first flight interval time as our y variable, this will be
        modelled as a logical AND constraint, so if y is True, xN must also
        be True, it can be modelled as a linear constraint as follows:
              y = x1 ∧ x2
          becomes...
              2y ≤ x1 + x2
              2y - x1 - x2 ≤ 0
              y ≤ x1
              y ≤ x2
              0 ≤ y ≤ 1

        Where y is the decision variable from the terminal schedule and xN
        is the list of terminal/runway decision variables
            
            Example:
                Y111 = Y211 ∧ Y311 ∧ Y411
                (3 * Y111) ≤ Y211 + Y311 + Y411
                (3 * Y111) - Y211 - Y311 - Y411 ≤ 0
                Y111 ≤ Y211
                Y111 ≤ Y311
                Y111 ≤ Y411
                0 ≤ Y111 ≤ 1

    - Terminals must not exceed their capacity at any time throughout the day
    
        For all flights and terminals and a given time interval, the sum of all
        intervals must not exceed the capacity of the terminal.
        
        Note: Originally in the code it was set so that the last interval of a
        flights stay at a terminal was not included so a flight could leave at
        a given time and another could arrive immediately. This was 
        subsequently removed as it skewed the results output from the last
        section making it look like terminals were exceeding their capacity.
        
            Y111 + Y121 + Y131 + ... + Y1Z1 ≤ capacity
    
    - All decision variables must be constrained to binary integer values 0,1
        
            0 ≤ W^tfr ≤ 1
            0 ≤ X^abc ≤ 1
            0 ≤ Y^tfr ≤ 1 

D - Identify the objective function for the Linear Programming model to 
minimise overall taxi distance.

    Minimise the total taxi distance between runway and terminals for both
    arrivals and departures, where distance is the cost.
    
        minimise(3[W111] + 5[W112] + 6[W113] + ... + 6[WDZ9] +
                 3[Y111] + 5[Y112] + 6[Y113] + ... + 6[YDZ9])
    
"""

"""
E - Implement and solve the Linear Programming model using the identified 
variables, constraints, and objective function.
"""

# Instantiate the solver
solver = pywraplp.Solver('LPWrapper',
                         pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

# Decision Variables
# ==================

# Instantiate LP decision variable holders
flight_arrivals = dict()
terminal_plan = dict()
flight_departures = dict()

# Vars for determining what runway a given flight should land on at its
# associated arrival time (W^tfr).
for a_time in arrival_times_list:
    for flight in flights:
        for runway in runways:
            for terminal in terminals:
                v_name = 'arr-{t}-{f}-({r}|{te})'.format(t=a_time, f=flight,
                                                         r=runway, te=terminal)
                flight_arrivals[(
                    a_time, flight, (runway, terminal))] = solver.IntVar(
                    0, 1, v_name)

# Vars for determining what terminal a flight should go to between arriving and
# derparting (X^abc).
for terminal in terminals:
    for t_time in combined_times_list:
        for flight in flights:
            v_name = '{te}-{tt}-{f}'.format(te=terminal, tt=t_time, f=flight)
            terminal_plan[(terminal, t_time, flight)] = solver.IntVar(
                0, 1, v_name)

# Vars for determining what runway a given flight should leave on at its
# associated departure time (Y^tfr).
for d_time in departure_times_list:
    for flight in flights:
        for runway in runways:
            for terminal in terminals:
                v_name = 'dep-{t}-{f}-({r}|{te})'.format(t=d_time, f=flight,
                                                         r=runway, te=terminal)
                flight_departures[(
                    d_time, flight, (runway, terminal))] = solver.IntVar(
                    0, 1, v_name)

# Constraints
# ===========
# - Only one flight can occupy a runway at a time
for runway in runways:
    # Set for arrivals
    # For each time in the list of arrivals
    for a_time in arrival_times_list:
        # Set constraint with ub of 1, this will limit the total amount
        # of flights for this arrival time and runway to 1
        c1 = solver.Constraint(0, 1)
        # For each flight/terminal combination
        for flight in flights:
            for terminal in terminals:
                # Set constraint coefficient on the runway/arrival time
                c1.SetCoefficient(flight_arrivals[(
                    a_time, flight, (runway, terminal))], 1)

    # Set for departures
    # For each time in the list of departures
    for d_time in departure_times_list:
        # Set constraint with ub of 1, this will limit the total amount
        # of flights for this departure time and runway to 1
        c2 = solver.Constraint(0, 1)
        # For each flight/terminal combination
        for flight in flights:
            for terminal in terminals:
                # Set constraint coefficient on the runway/departure time
                c2.SetCoefficient(flight_departures[(
                    d_time, flight, (runway, terminal))], 1)


# - Flights must arrive and depart at their designated time
for flight in flights:
    # Get the arrival and departure times of the current flight
    fa_time = str(flight_schedule['Arrival'][flight])
    fd_time = str(flight_schedule['Departure'][flight])

    # Arrival time constraint
    # Set constraint on the flight/arrival time with both lb and ub of 1, this
    # will enforce from all runway/terminal combinations one of them must be
    # selected in the solution
    c3 = solver.Constraint(1, 1)
    # For each runway/terminal combination
    for runway in runways:
        for terminal in terminals:
            # Set constraint coefficient of 1, this both ensures we will have
            # the plane arrive at the right time and that it can't land on
            # more than one runway
            c3.SetCoefficient(
                flight_arrivals[(
                    fa_time, flight, (runway, terminal))], 1)

    # Departure time constraint
    # Set constraint on the flight/arrival time with lb and ub of 1 for same
    # effect as arrival time constraint
    c4 = solver.Constraint(1, 1)
    # For each runwau/terminal combination
    for runway in runways:
        for terminal in terminals:
            # Set constraint coefficient of 1
            c4.SetCoefficient(
                flight_departures[(
                    fd_time, flight, (runway, terminal))], 1)

# - Flights must use the correct terminal for their runway/terminal taxi route
for flight in flights:
    # Get the arrival and departure times of the current flight
    fa_time = str(flight_schedule['Arrival'][flight])
    fd_time = str(flight_schedule['Departure'][flight])

    # Arrival constraint - Flight goes to correct terminal dependent on
    # runway/terminal taxi route
    # For each terminal
    for terminal in terminals:
        # Instantiate list to hold terminal/runway decision vars
        x_list = list()
        # For each runway
        for runway in runways:
            # Add the terminal/runway combination to the list of vars
            x_list.append(flight_arrivals[(
                fa_time, flight, (runway, terminal))])
        # Get the associated decision variable in the terminal schedule
        y = terminal_plan[(terminal, fa_time, flight)]

        # y ≤ x1 + x2 + ... + xN
        solver.Add(y <= sum(x_list))
        for x in x_list:
            # y ≥ xN
            solver.Add(y >= x)
        # 0 ≤ y ≤1
        solver.Add(0 <= y <= 1)

    # Departure Constraint - Flight leaves from correct terminal/runway
    # combination when departing
    for terminal in terminals:
        # Instantiate list to hold terminal/runway decision vars
        x_list = list()
        # For each runway
        for runway in runways:
            # Add the terminal/runway combination to the list
            x_list.append(flight_departures[(
                fd_time, flight, (runway, terminal))])
        # Get the associated decision variable in the terminal schedule
        y = terminal_plan[(terminal, fd_time, flight)]
        # Set logical OR constraint so that if any terminal/runway combination
        # is True, the associated terminal is used in the terminal schedule
        solver.Add(y <= sum(x_list))
        for x in x_list:
            solver.Add(y >= x)
        solver.Add(0 <= y <= 1)

# - Flights must stay at the same terminal for the duration of their stay
for flight in flights:
    # Get the current flight's arrival and departure time
    fa_time = str(flight_schedule['Arrival'][flight])
    fd_time = str(flight_schedule['Departure'][flight])
    # Get the associated indexes of the arrival and departure times from the
    # list of combined times
    fa_idx = combined_times_list.index(fa_time)
    fd_idx = combined_times_list.index(fd_time)
    # Create a new list from a slice of the combined times list, this will
    # get the intervals for which the current flight needs to be at a terminal
    f_intervals = combined_times_list[fa_idx:fd_idx + 1]
    # For each terminal
    for terminal in terminals:
        # Get the associated decision variable in the terminal schedule
        y = terminal_plan[(terminal, fa_time, flight)]
        # Instantiate list to hold terminal/runway decision vars
        x_list = list()
        # For each time interval in the flights required stay intervals
        for interval_time in f_intervals:
            # Add the terminal/interval/flight to the list of vars
            x_list.append(terminal_plan[(terminal, interval_time, flight)])
        # Remove the arrival interval time from the list of comparison vars
        x_list.remove(y)

        # len(xN) * y ≤ x1 + x2 + ... - xN
        solver.Add(len(x_list) * y <= sum(x_list))
        # len(xN) * y - x1 - x2 - ... - xN ≤ 0
        n = 0
        n += len(x_list) * y
        for x in x_list:
            n -= x
            # y ≤ xN
            solver.Add(y <= x)
        solver.Add(n <= 0)
        # 0 ≤ y ≤ 1
        solver.Add(0 <= y <= 1)

# - Terminals must not exceed their capacity at any time throughout the day
for terminal in terminals:
    # Get the capacity of the terminal
    capacity = float(terminal_capacity['Gates'][terminal])
    # For each time interval in the terminal schedule
    for interval in combined_times_list:
        # Set constraint with ub of the terminal capacity so it cannot be
        # exceeded by associated flight count
        c5 = solver.Constraint(0, capacity)
        # For each flight
        for flight in flights:
            # This was removed because it skewed results for the last part of
            # this section with capacity limits seeming to be exceeded, it does
            # not count last interval so flights can leave and arrive at
            # terminal at the same time

            # fd_time = str(flight_schedule['Departure'][flight])
            # if interval != fd_time:

            # Set the constraint coefficient as 1, so the total sum of
            # planes at a given terminal does not exceed the capacity ub
            # on the constraint.
            c5.SetCoefficient(
                terminal_plan[(terminal, interval, flight)], 1)

# Objective Function
# ==================
# Reduce the total taxi distance for flights between runway and terminal for
# both arrivals and departures
objective = solver.Objective()
# For each flight
for flight in flights:
    # For each flights arrival and departure time
    fa_time = str(flight_schedule['Arrival'][flight])
    fd_time = str(flight_schedule['Departure'][flight])
    # For runway/terminal combination
    for runway in runways:
        for terminal in terminals:
            # Set the associated cost of that combination for the flights
            # arrival
            objective.SetCoefficient(
                flight_arrivals[(fa_time, flight, (runway, terminal))],
                float(taxi_distances[terminal][runway]))
            # Set the associated cost of that combination for the flights
            # departure
            objective.SetCoefficient(
                flight_departures[(fd_time, flight, (runway, terminal))],
                float(taxi_distances[terminal][runway]))

objective.SetMinimization()
solver.Solve()

"""
F - Output the allocation of arrival runway, departure runway, and terminal 
for each flight.
"""
print('#-----------------------------------------------#')
print('|  FLIGHT RUNWAY -> TERMINAL -> RUNWAY SCHEDULE |')
print('#-----------------------------------------------#')
print('Note: Without a sample of correct results I cannot tell if there is '
      '\nmeant to be so many instances where the same letter runway and '
      '\nterminal are in use, it isnt all the time (Flight C, M, O etc.) but '
      '\nit still seems like an unlikely result.\n')

taxi_total_cost = 0

for flight in flights:
    arrival_terminal = None
    arrival_runway = None
    arrival_time = None

    departure_terminal = None
    departure_runway = None
    departure_time = None

    for a_time in arrival_times_list:
        for runway in runways:
            for terminal in terminals:
                if flight_arrivals[(
                        a_time, flight,
                        (runway, terminal))].solution_value() > 0:
                    arrival_runway = runway
                    arrival_time = a_time
                    arrival_terminal = terminal
                    taxi_total_cost += float(taxi_distances[terminal][runway])

    for d_time in departure_times_list:
        for runway in runways:
            for terminal in terminals:
                if flight_departures[(
                        d_time, flight,
                        (runway, terminal))].solution_value() > 0:
                    departure_runway = runway
                    departure_time = d_time
                    departure_terminal = terminal
                    taxi_total_cost += float(taxi_distances[terminal][runway])

    print('{f}: Arr: {a} onto {r1} | Taxi to {te} | Dep: {d} from {r2}'.format(
        f=flight, a=arrival_time, r1=arrival_runway, te=arrival_terminal,
        d=departure_time, r2=departure_runway))
    if arrival_terminal != departure_terminal:
        print(' *** Terminal arr/dep mismatch ***')

"""
G - Answer the question how much total taxi distance was incurred by all 
flights together and how many gates were occupied in each terminal at every 
time of the day.
"""
print('\n#------------------------------------------#')
print('|  Taxi Distance Total & Terminal Capacity |')
print('#------------------------------------------#')

print('\nTaxi Total Distance: {t}'.format(t=taxi_total_cost))
print('Sanity Check - Objective Solution Value: {s}\n'.format(
    s=solver.Objective().Value()))
terminal_capacity_info = dict()

for terminal in terminals:
    terminal_capacity_info[terminal] = list()
    for interval in combined_times_list:
        interval_total = 0
        for flight in flights:
            if terminal_plan[(
                    terminal, interval, flight)].solution_value() > 0:
                interval_total += 1
        terminal_capacity_info[terminal].append(interval_total)

for i in range(0, len(combined_times_list)):
    print('{ti} | Terminal A: {tA} | Terminal B: {tB} '
          '| Terminal C: {tC}'.format(
            ti=combined_times_list[i],
            tA=terminal_capacity_info['Terminal A'][i],
            tB=terminal_capacity_info['Terminal B'][i],
            tC=terminal_capacity_info['Terminal C'][i]))
