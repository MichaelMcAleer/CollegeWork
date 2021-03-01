# Assignment 2 - Planning & Scheduling
# Problem 1 - Johnson's Rule
# Michael McAleer (R00143621)
import csv
from heapq import heappush, heappop


class Job(object):
    """
    The job object containing all info about the job including ID, p1 job
    duration and p2 job duration.
    """

    def __init__(self, job_id, p1, p2):
        # Set job attributes
        self.id = job_id
        self.p1 = p1
        self.p2 = p2


class JohnsonsRuleSolver:
    """
    The Johnson's Rule solver logic and CSV processing mechanism.
    """

    @staticmethod
    def process_jobs(file_name):
        """
        Read contents of CSV file to get job details, create job objects and
        define P1 & P2 job time heaps.

        :param file_name: (str) Path to the CSV file containing job info
        :return: - response (list) The list of job objects
                 - p1_heap (heap) Heap of P1 job times
                 - p2_heap (heap) Heap of P2 job times
        """
        # Open the CSV file
        with open(file_name) as csv_file:

            # Initialise job list and priority heaps for sorting M1 M2 jobs
            response = list()
            p1_heap = list()
            p2_heap = list()

            # Read the CSV file contents and set delimiter, start counter
            csv_reader = csv.reader(csv_file, delimiter=',')
            counter = 0

            # For each row in the imported file contents
            for row in csv_reader:
                # First row is column headers so can be skipped
                if counter >= 1:
                    # Create job object, add job ID and both P1 & P2 times
                    job = Job(job_id=int(row[0]),
                              p1=int(row[1]),
                              p2=int(row[2]))
                    # Add times to respective heaps, the counter is added to
                    # resolve ties between jobs on the same machine, the job
                    # with the lowest counter (index) will be chosen first
                    heappush(p1_heap, (job.p1, counter, job))
                    heappush(p2_heap, (job.p2, counter, job))
                    # Add job to response list
                    response.append(job)
                counter += 1

            return response, p1_heap, p2_heap

    def run_johnsons_rule(self, file_name):
        """
        Run the Johnson's rule solver.

        :param file_name: (str) Path to the CSV file containing job info:
        :return:
        """
        # Get jobs and M1/M2 job heaps
        jobs, m1_heap, m2_heap = self.process_jobs(file_name)
        # Set scheduled job counter to 0
        scheduled_jobs = 0
        # Initialise a set to contain all scheduled jobs (unordered) solely for
        # the purpose of considerably faster list searching when checking if a
        # job is scheduled already
        schedule = set()
        # Initialise a list the same length as the total amount of jobs and set
        # each value to None, this will be the ordered schedule to be returned
        schedule_order = [None] * len(jobs)
        # Set the index of M1 to 0 to start placement of M1 jobs at beginning
        # of schedule working forwards
        m1_index = 0
        # Set the index of M2 to length of the total amount of jobs minus 1,
        # this will control the placement of M2 jobs at the end of the schedule
        # working backwards
        m2_index = len(jobs) - 1

        # While there are still jobs remaining to be scheduled
        while scheduled_jobs != len(jobs):
            # Get the shortest jobs from M1 & M2 priority heaps
            (j1, _, j1_job) = m1_heap[0]
            (j2, _, j2_job) = m2_heap[0]

            # If the J1 job has already been scheduled, remove from heap and
            # pass iteration to start loop again
            if j1_job in schedule:
                heappop(m1_heap)
                pass
            # If the J2 job has already been scheduled, remove from heap and
            # pass iteration to start loop again
            elif j2_job in schedule:
                heappop(m2_heap)
                pass
            # Both jobs are unscheduled, determine which to schedule next
            else:
                # If the j1 time is lower than or equal to the j2 time...
                if j1 < j2 or j1 == j2:
                    # Add the associated job to the current M1 index position
                    schedule_order[m1_index] = j1_job
                    # Add the job to the schedule set
                    schedule.add(j1_job)
                    # Increment the m1 index counter by 1 for next job
                    # placement
                    m1_index += 1
                    # Remove the job from the heap so it will not be scheduled
                    # again
                    heappop(m1_heap)
                # Else the j2 time is shorter than j1 time...
                else:
                    # Add the associated job to the current M2 index position,
                    # ties in M2 jobs are broken by the index of the job in the
                    # heap
                    schedule_order[m2_index] = j2_job
                    # Add the job to the schedule set
                    schedule.add(j2_job)
                    # Decrement the m1 index counter by 1 for next job
                    # placement
                    m2_index -= 1
                    # Remove the job from the heap so it will not be scheduled
                    # again
                    heappop(m2_heap)
                # Increment scheduled jobs count
                scheduled_jobs += 1

        return schedule_order


if __name__ == '__main__':
    # File name - to change the solver to run on a different file, change the
    # value of the string job_file_name
    job_file_name = 'part1_jobs.csv'

    # Create solver
    jr_solver = JohnsonsRuleSolver()
    # Run solver
    solution = jr_solver.run_johnsons_rule(job_file_name)
    # Output solution to screen
    cnt = 1
    for scheduled_job in solution:
        print("Job {}: {}".format(cnt, scheduled_job.id))
        cnt += 1
