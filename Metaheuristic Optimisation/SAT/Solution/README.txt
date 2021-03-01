#-------------------------------#
| Assignment 2 - GSAT & TSP ILS |
| Michael McAleer (R00143621)   |
#-------------------------------#

-Included Files/Folders
    - Data (The raw data files used during runs)
    - Results (The results the report are based off)
    - Assignment2_Report-MichaelMcAleer.pdf (The report)
    - part-1.py (Part 1 of the assignment)
    - part-2.py (Part 2 of the assignment)
    - part-3.py (Part 3 of the assignment)
    - README.txt (This file, details on submission)

- Run-Time details
    - All three parts of the assignment run successfully with no expected errors
    - Error handling has been implemented across all parts so any potential
      failures will be dealt with gracefully and resolved with no user input
    - Python 3.7 was used during development
    - There are no dependencies required to run any of the three parts, all
      libraries imported are standard libraries
    - All parts were built and tested on Windows 10, the parts are all idempotent
      so should have no issues running on any platform

- Part-1.py Run Configuration
    - To part 1 use the following command:
      - Format:
        - python part-1.py [file_name] [write_to_file] [write_to_csv]
            - file_name: (string) The path to the CNF sat problem - REQUIRED
            - write_to_file: (boolean) If the solution should be written to
              text file, this defaults to True - NOT REQUIRED
            - write_to_csv: (boolean) If the run results should be written to
              CSV file, this defaults to True - NOT REQUIRED
      - Example:
        - python part-1.py Data/uf20-020.cnf
            - This will run part-1.py against uf20-020.cnf and output the solution
             and run results to file.
    - When outputting files, this part will put the results files in the same
      directory

- Part-2.py Run Configuration
   - To part 1 use the following command:
     - Format:
       - python part-2.py [file_name] [write_to_file] [write_to_csv]
           - file_name: (string) The path to the CNF sat problem - REQUIRED
           - write_to_file: (boolean) If the solution should be written to
             text file, this defaults to True - NOT REQUIRED
           - write_to_csv: (boolean) If the run results should be written to
             CSV file, this defaults to True - NOT REQUIRED
     - Example:
       - python part-2.py Data/uf20-020.cnf
           - This will run part-2.py against uf20-020.cnf and output the solution
            and run results to file.
   - When outputting files, this part will put the results files in the same
      directory

- Part-3.py Run Configuration
   - To part 3 use the following command:
     - Format:
       - python part-3py [file_name] [initial_sol] [threads]
           - file_name: (string) The path to the CNF sat problem - REQUIRED
           - intitial_sol: (string) What initial solution is required, valid values
             are 'random' and 'nearest_neighbour', this defaults to
             'nearest_neighbour' - NOT REQUIRED
           - threads: (int) The amount of threads required for the local search
             algorithm, this defaults to 3000 - NOT REQUIRED
     - Example:
       - python part-3.py Data/inst-0.tsp random 5000
           - This will run part-3.py against inst-0.tsp using random initial tour
             selection and 5000 threads allocated for the local search algorithm.
   - When outputting files, this part will put the results files in the same
      directory. The TSP solution will be output to the screen and not in a file.
