# Assignment 2 - Planning & Scheduling
# Problem 2 - Tabu Search
# Michael McAleer (R00143621)
import csv
import heapq


class Job(object):
    """
    The job object containing all info about the job including ID, duration,
    due date, and weight.
    """

    def __init__(self, job_id, duration, due, weight):
        self.id = job_id
        self.duration = duration
        self.due = due
        self.weight = weight


class Sequence(object):
    """
    The sequence object holding all jobs, contains all sequence related info
    including order of jobs, cost of order, and job IDs exchanged from parent
    sequence. The cost of the sequence of jobs is determined when the
    sequence is initialised.
    """

    def __init__(self, order):
        self.order = order
        self.cost = 0
        self.exchanged = None
        self.calculate_cost()

    def __str__(self):
        """
        Return printable string representation of sequence as job order.

        :return: (str) The order of job IDs
        """
        return "{}{}{}{}".format(self.order[0].id, self.order[1].id,
                                 self.order[2].id, self.order[3].id)

    def calculate_cost(self):
        """
        Calculate the cost of the job order for current sequence.
        """
        # Set time and cost counters
        current_time = 0
        cost_total = 0
        # For each job in the sequence of jobs
        for job in self.order:
            # Add the job duration to the total time
            current_time += job.duration
            # If the current elapsed time is greater than the job due time...
            if current_time > job.due:
                # Calculate the time penalty
                penalty = (current_time - job.due)
                # Add the penalty times weight to the cumulative cost total
                cost_total += (penalty * job.weight)
        # Set the cumulative cost as the sequence cost
        self.cost = cost_total


class TabuSearchSolver:
    """
    The Tabu Search class which controls all tabu search related functions.
    Attributes include the current search sequence, best known sequence, tabu
    list, and tabu list size.
    """

    def __init__(self):
        self.curr_sol = None
        self.best_sol = None
        self.tabu_list = list()
        self.tabu_size = 0

    @staticmethod
    def process_jobs(file_name):
        """
        Read contents of CSV file to get job details and create job objects.

        :param file_name: (str) Path to the CSV file containing job info
        :return: response (list) The list of job objects
        """
        # Open the file
        with open(file_name) as csv_file:
            # Initialise response list to hold jobs
            response = list()
            # Read the CSV file contents and set delimiter, start counter
            csv_reader = csv.reader(csv_file, delimiter=',')
            counter = 0
            # For each row in the imported file contents
            for row in csv_reader:
                # First row is column headers so can be skipped
                if counter > 0:
                    # Create job object defining properties extracted from CSV
                    job = Job(job_id=int(row[0]),
                              duration=int(row[1]),
                              due=int(row[2]),
                              weight=int(row[3]))
                    # Add the job to the response list
                    response.append(job)
                # Increment counter
                counter += 1
            return response

    @staticmethod
    def get_sequence(job_data, order):
        """
        Update the sequence of job IDs to reflect list index starting at 0 and
        return list of jobs ordered by user input.

        :param job_data: (list) The job objects
        :param order: (str) The order of jobs as defined by user input
        :return: (list) The ordered list of jobs
        """
        # Initialise response list to hold ordered list of jobs
        updated_order = list()
        # Convert the order of jobs from string to list of integers
        order = list(map(int, order))
        # For the range of total jobs...
        for i in range(0, len(order)):
            # Get the index of the user specified order, 1=0, 2=1 etc.
            index = order[i] - 1
            # Add the job to the ordered list by index
            updated_order.append(job_data[index])
        return updated_order

    def get_sequence_variants(self, sequence):
        """
        For the current sequence, get all possible adjacent pair-wise
        combinations, if the pair-wise exchange is not in the Tabu list, add
        the combination to the heap queue.

        :param sequence: (obj) The current sequence
        :return: (list) All possible valid sequence variation objects
        """
        # Initialise response list to hold sequence variations
        variants = list()
        for i in range(0, len(sequence.order) - 1):
            # Make a copy of the source sequence order
            variant = sequence.order.copy()
            # Apply adjacent pair-wise exchange to sequence order
            variant[i], variant[i + 1] = variant[i + 1], variant[i]
            # Make new sequence with variant order
            seq_var = Sequence(variant)
            # Set exchanged jobs value for addition to tabu list
            exc = [variant[i].id, variant[i + 1].id]
            seq_var.exchanged = '{}{}'.format(sorted(exc)[0],
                                              sorted(exc)[1])
            # If the pair-wise exchange is not in the Tabu list...
            if seq_var.exchanged not in self.tabu_list:
                # Add sequence variant to heap with cost dictating order from
                # lowest to highest
                heapq.heappush(variants, (seq_var.cost, seq_var))
            # Else the exchange is in the Tabu list, do not add to heap...
            else:
                print("\t-Variant {} job IDs {} in tabu list, skipping".format(
                    seq_var, seq_var.exchanged))
        return variants

    def update_tabu_list(self, exchange):
        """
        Update Tabu list with pair-wise exchanged index IDs.

        :param exchange: (str) The ordered exchange IDs
        """
        # If the Tabu list is populated...
        if self.tabu_list:
            # Append the exchanged job IDs to the start of the Tabu list
            self.tabu_list = [exchange] + self.tabu_list
            # If the length of the Tabu list is longer than the specified Tabu
            # list max size, remove the last (oldest) item in the list
            if len(self.tabu_list) > self.tabu_size:
                self.tabu_list.pop(self.tabu_size)
        # Else the list is empty, append exchanged job IDs to the Tabu list
        else:
            self.tabu_list.append(exchange)

    def run_search(self, file_name, order, tabu_size=2, iterations=100):
        """
        Run the local search algorithm with Tabu list.

        :param file_name: (str) The path to CSV file containing job info
        :param order: (str) The order of the jobs by job ID
        :param tabu_size: (int) The size of the Tabu list, default is 2
        :param iterations: (int) The amount of iterations to complete during
        search process, default is 100
        """
        # Extract the job data from the CSV file
        jobs = self.process_jobs(file_name)
        # Set the order of the jobs as the supplied starting order
        order = self.get_sequence(jobs, order)
        # Set the current solution as sorted sequence of jobs
        self.curr_sol = Sequence(order=order)
        # Set current sequence as the best known sequence
        self.best_sol = self.curr_sol
        # Set the Tabu list size as the user supplied value, default value is 2
        self.tabu_size = tabu_size
        # Reset the tabu list before starting search
        self.tabu_list = list()

        # Start iteration for loop equal to length of user specified input
        for iteration in range(0, iterations):
            # Output current sequence details
            print("\n>> Iteration {} Sequence: {} \nSequence Cost: {}".format(
                iteration + 1, self.curr_sol, self.curr_sol.cost))
            print("Tabu List: {}".format(self.tabu_list))

            # Get variations heap
            variants_heap = self.get_sequence_variants(self.curr_sol)

            # Output sequence variations
            for cost, variant in variants_heap:
                print("\t-Variant {} cost: {}".format(variant, cost))

            # Take the sequence with lowest cost from heap
            best_cost, best_variant = heapq.heappop(variants_heap)
            print("Best variant {}: {}".format(best_variant, best_cost))

            # Set the best sequence as current sequence
            self.curr_sol = best_variant
            # Update the Tabu list with the new sequence
            self.update_tabu_list(best_variant.exchanged)

            # If the sequence cost is better than the current best sequence
            # cost, set sequence as best
            if best_cost < self.best_sol.cost:
                self.best_sol = best_variant
                print("\t>>> Better sequence found: {} Cost: {}".format(
                    best_variant, best_cost))

        # After all iterations are complete, output best solution found
        print("\nBest solution found: {} with a cost of {}".format(
            self.best_sol, self.best_sol.cost))


if __name__ == '__main__':
    # Initialise Tabu Search solver
    ts = TabuSearchSolver()
    job_file_name = 'pt2_jobs.csv'
    # Run search
    ts.run_search(file_name=job_file_name, order='3142', tabu_size=2,
                  iterations=100)
