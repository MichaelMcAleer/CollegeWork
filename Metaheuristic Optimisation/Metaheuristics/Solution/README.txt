================================================
Genetic Algorithms - Travelling Salesman Problem
Solution Implemented by Michael McAleer
================================================

Submitted Files:
    -TSP_Solution_MMA.py
    -dataset
        -inst-0.tsp
        -inst-13.tsp
        -inst-16.tsp
    -Results
        -Results Default Config
        -Results Exploratory
    -README.txt
    -'Metaheuristic Optimisation - Assignment 1.pdf'

Compiling Environment/Compiling Steps
    -Script built using Python 3.7
    -All libraries used are standard Python libraries
    -No additonal dependencies/libraries required
    -Can run on Ubuntu/Windows without issues if Python 3.7 is in use

Execution Instructions
    -All configurations have been implemented in the one file
    -The Individual.py file has been incorporated into the TSP_Solution_MMA.py
     file
    -To run the script, from the command line issue the following:

    Format:
        $ python TSP_Solution_MMA.py [in_file] [configuration]
    Example:
        $ python TSP_Solution_MMA.py dataset/inst-0.tsp 1

    -Where:
        -[in_file] is the dataset from which to work from
        -[configuration] is the GA configuration to implement

Configurations
    -The configurations implemented in the TSP_Solution_MMA.py file mirror
     exactly the configurations in the assignment specification
    -There is a seventh configuration implemented which is a mix of all of the
     best functionality from all the previous configurations. It is recommended
     to run this with elite survival enabled
    -To enable elite survival, set the variable elite to 'True' on line 902
     of the TSP_Solution_MMA.py file
    -It is also possible to edit the run count, iterations, mutation rate,
     and population size from the TSP_Solution_MMA.py file by editing the
     variables on lines 898-901

Known Issues
    -There are no known issues, the GA implementation should work without
     error across all configurations and with changes to the variables
     discussed in the 'Configurations' section