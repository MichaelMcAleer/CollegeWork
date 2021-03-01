"""
Author: Michael McAleer
file: Genetic Algorithms & TSP Solution
"""

import csv
import math
import random
import statistics as st
import sys
import time
from heapq import nsmallest


class Individual:
    def __init__(self, _size, _data):
        """
        Parameters and general variables
        """
        self.fitness = 0
        self.genes = []
        self.genSize = _size
        self.data = _data

        self.genes = list(self.data.keys())

        for i in range(0, self.genSize):
            n1 = random.randint(0, self.genSize - 1)
            n2 = random.randint(0, self.genSize - 1)
            tmp = self.genes[n2]
            self.genes[n2] = self.genes[n1]
            self.genes[n1] = tmp

    def setGene(self, genes):
        """
        Updating current chromosome
        """
        self.genes = []
        for gene_i in genes:
            self.genes.append(gene_i)

    def copy(self):
        """
        Creating a new individual
        """
        ind = Individual(self.genSize, self.data)
        for i in range(0, self.genSize):
            ind.genes[i] = self.genes[i]
        ind.fitness = self.getFitness()
        return ind

    def euclideanDistance(self, c1, c2):
        """
        Distance between two cities
        """
        d1 = self.data[c1]
        d2 = self.data[c2]
        return math.sqrt((d1[0] - d2[0]) ** 2 + (d1[1] - d2[1]) ** 2)

    def getFitness(self):
        return self.fitness

    def computeFitness(self):
        """
        Computing the cost or fitness of the individual
        """
        self.fitness = self.euclideanDistance(self.genes[0],
                                              self.genes[len(self.genes) - 1])
        for i in range(0, self.genSize - 1):
            self.fitness += self.euclideanDistance(self.genes[i],
                                                   self.genes[i + 1])


class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations,
                 _configuration, _run_count, _elite):
        """
        Parameters and general variables
        """
        # Population contains random variations of TSP solutions
        # Each individual in the population has a dict of all positions,
        # chromosome size (genSize), genes (chromosome) list, and fitness
        self.population = list()
        self.matingPool = list()

        # File Handling
        self.fName = _fName
        self.data = dict()
        self.csv_files = list()

        # Run-Time Variables
        self.best = None
        self.second_best = None
        self.worst = None
        self.genSize = None
        self.elite = elite
        self.elite_pool = list()
        self.iteration = 0
        self.popSize = _popSize
        self.mutationRate = _mutationRate
        self.maxIterations = _maxIterations
        self.configuration = _configuration
        self.run_count = _run_count
        self.current_run = 1

        # Run Performance Monitors
        self.improvement_cnt = 0
        self.run_mean = list()
        self.run_median = list()
        self.run_max = list()
        self.run_min = list()
        self.run_range = list()

        # Initialise GA, read data in and build population
        print("================================================"
              "\nGenetic Algorithms - Travelling Salesman Problem"
              "\nSolution Implemented by Michael McAleer"
              "\nRunning Configuration {} - Run Count is {}"
              "\n================================================".format(
            self.configuration, self.run_count))
        self.readInstance()
        self.initPopulation()

    # DATA INPUT CONTROL

    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data.clear()
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

    # POPULATION CONTROL

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        # Clear population and mating pool when initialise population
        # is called
        self.population.clear()
        self.matingPool.clear()

        # If this is not the first run, elite_pool may be populated
        # with best solutions from previous run, this if/else block handles
        # the addition of elite survivors if enabled
        if len(self.elite_pool) > 0:
            self.population += self.elite_pool
            for i in range(len(self.elite_pool), self.popSize):
                individual = Individual(self.genSize, self.data)
                individual.computeFitness()
                self.population.append(individual)
            self.elite_pool.clear()
        else:
            for i in range(0, self.popSize):
                individual = Individual(self.genSize, self.data)
                individual.computeFitness()
                self.population.append(individual)

        # Determine best/second best/worst individuals in population
        self.best = self.population[0].copy()
        self.worst = self.population[0].copy()
        for ind_i in self.population:
            ind_i.computeFitness()
            # Calculate Best Fitness
            if self.best.getFitness() > ind_i.getFitness():
                self.second_best = self.best.copy()
                self.best = ind_i.copy()
            # Calculate Worst Fitness
            if self.worst.getFitness() < ind_i.getFitness():
                self.worst = ind_i.copy()

        print("Best initial sol: ", self.best.getFitness())

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = list(self.population)

    def updateBest(self, candidate):
        if not self.best or candidate.getFitness() < (
                self.best.getFitness()):
            self.second_best = self.best.copy()
            self.best = candidate.copy()
            self.improvement_cnt += 1
            print("iteration: ", self.iteration, "best: ",
                  self.best.getFitness())

    # SELECTION FUNCTIONS

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[random.randint(0, self.popSize - 1)]
        indB = self.matingPool[random.randint(0, self.popSize - 1)]
        return indA, indB

    def rouletteWheel(self):
        """
        Roulette Wheel Selection with minimisaton focus.

        :return: The first individual met whose selection probability added to
        the cumulative total of all previous individuals is greater than the
        random number selected between 0 and 1.
        """
        # Get the maximum fitness of all individuals in population
        max_fitness = self.worst.getFitness()
        # Determine the sum of all the fitness values of all individuals
        # in population
        fitness_sum = sum([((max_fitness + 1) - ind.fitness) for ind in
                           self.matingPool])

        # Initialise the selection probability to 0
        probability_offset = 0
        # Select the random roulette number in range of 0 to 1
        roulette_pick = random.uniform(0, 1)

        # For each individual in the matin pool...
        for ind in self.matingPool:
            # Determine the selection probability of the individual, the
            # probability offset here will be the cumulative total of all
            # probabilities of individuals encountered before the current
            # individual. This method will return the roulette selection when
            # it is met instead of calculating entire population fitness values
            ind.probability = (
                    probability_offset + (
                    (max_fitness + 1) - ind.fitness) / fitness_sum)
            # If the individual probability selection is greater than the
            # roulette selection, then the value has been met i.e. is the first
            # fitness selection greater than the random number
            if ind.probability > roulette_pick:
                return ind
            # Else the routlette value has not been met, add the current
            # individual selection probability to the probability offset and
            # continue to the next individual in the mating pool
            else:
                probability_offset += (((max_fitness + 1) - ind.fitness) /
                                       fitness_sum)

    # CROSSOVER FUNCTIONS

    def uniformCrossover(self, chromosome_a, chromosome_b):
        """
        Uniform Order-Based Crossover implementation.

        :param chromosome_a: The first parent selected in advance from mating
        pool.
        :param chromosome_b: The second parent selected in advance from mating
        pool.
        :return: Two individual children with genes constructed using uniform
        order-based crossover technique from both parents.
        """
        # Initialise the two child gene lists with all values set to None,
        # length is dependent on the length of the gene size of parent genes
        child_a, child_b = [None] * chromosome_a.genSize, \
                           [None] * chromosome_b.genSize

        # Get random selection of indexes equal to half total size of parent
        # gene size, if odd number, round up to nearest integer
        random_index_list = random.sample(range(0, chromosome_a.genSize - 1),
                                          int(round(chromosome_a.genSize) / 2))

        # Populate children with index values of parent genes, indexes
        # determined from random selection in previous step
        for i in random_index_list:
            child_a[i], child_b[i] = (chromosome_a.genes[i],
                                      chromosome_b.genes[i])

        def _ordered_crossover(child, target_chromosome):
            """
            Nested function which handles the assignment of remaining genes
            from opposite parent.
            :param child: The child genes.
            :param target_chromosome: The opposite parent genes.
            :return: New individual with genes constructed from both parents.
            """
            # Get the remaining genes left to be assigned to the current child.
            # This list is populated by determining which genes in parent have
            # already been assigned to the child, removing them, and creating
            # a new list with only unassigned genes
            genes_for_child = [gene for gene in target_chromosome.genes if
                               gene not in child]
            # For gene(k) in child genes...
            for k in range(0, len(child)):
                # If the gene at position K is None and there are still genes
                # left from parent to be assigned to the child...
                if child[k] is None and genes_for_child:
                    # Assign the gene to the current index position of the
                    # child
                    child[k] = genes_for_child[0]
                    # Remove the gene from the list of genes to be assigned,
                    # decrementing all remaining genes index by 1 (next gene to
                    # be assigned moves to position 0)
                    genes_for_child.pop(0)

            # Make a copy of the current parent chromosome and assign the new
            # genes so it can be added to the population later and the mating
            # pool remains unaffected
            new_individual = target_chromosome.copy()
            new_individual.genes = child
            return new_individual

        return _ordered_crossover(child_a, chromosome_b), _ordered_crossover(
            child_b, chromosome_a)

    def cycleCrossover(self, chromosome_a, chromosome_b):
        """
        Cycle Crossover Implementation.

        :param chromosome_a: The first parent selected in advance from mating
        pool.
        :param chromosome_b: The second parent selected in advance from mating
        pool.
        :return: Two individual children with genes constructed using cycle
        crossover technique from both parents.
        """
        # Initialise the two child gene lists with all values set to None,
        # length is dependent on the length of the gene size of parent genes
        child_a, child_b = ([None] * chromosome_a.genSize,
                            [None] * chromosome_b.genSize)

        def _run_cycle(chrom_a, chrom_b, alt_cycle):
            """
            Nested funtion to control the cycle selection and assignment
            process.

            :param chrom_a: The first parent selected in advance from mating
            pool.
            :param chrom_b: The second parent selected in advance from mating
            pool.
            :param alt_cycle: If the cycle is to be carried out on the opposite
            parent (alternate cycle)
            """
            # Initialise starting index value, this will remain throughout
            # cycles so the next index to be assigned is already known without
            # another For loop
            index = 0
            # Set the gene which the current cycle will start with
            start_value = chrom_a[0]
            # Initialise the variable for the next gene to populate when the
            # cycle commences
            next_value = None

            # Continue the loop until the next gene selected is equal to the
            # gene started from indicating the end of the current cycle
            while start_value != next_value:
                # If there are None values still in the child then determine
                # next gene to be assigned
                if None in child_a:
                    # If there is a gene at current index position, increment
                    # index variable by 1 and skip to next index position
                    if child_a[index]:
                        index += 1
                    # If there is no gene at current index position
                    if not child_a[index]:
                        # If the current cycle is alternate cycle, assign gene
                        # from alterante parent genes (A-B, B-A)
                        if alt_cycle:
                            child_a[index] = chrom_b[index]
                            child_b[index] = chrom_a[index]
                        # Else the current cycle is not alternate, assign gene
                        # from matching parent (A-A, B-B)
                        else:
                            child_a[index] = chrom_a[index]
                            child_b[index] = chrom_b[index]
                        # If the gene at current index value in parent B
                        # matches the gene the cycle started with, set next
                        # gene value to start value, end current cycle, restart
                        # index counter
                        if chrom_b[index] == start_value:
                            index = 0
                            next_value = start_value
                        # Else get the gene at the index value of parent B
                        # genes for the next loop in current cycle
                        else:
                            index = chrom_a.index(chrom_b[index])
                # Else all genes have been assigned to child, break out of loop
                else:
                    break

        # Start with regular cycle
        alternate_cycle = False
        # While there are None values remaining in genes of child, run cycle
        # crossover function
        while None in child_a:
            _run_cycle(chromosome_a.genes, chromosome_b.genes,
                       alternate_cycle)
            # For each run of the cycle, alternate the cycle type
            alternate_cycle = True if not alternate_cycle else False

        # Initialise new individuals by copying parent indivudals, leaving
        # parents untouched in the mating pool
        new_individual_a = chromosome_a.copy()
        new_individual_b = chromosome_a.copy()
        # Assign the new child genes to the new individuals
        new_individual_a.genes = child_a
        new_individual_b.genes = child_b

        return new_individual_a, new_individual_b

    # MUTATION FUNCTIONS

    def reciprocalExchangeMutation(self, individual):
        """
        Reciprocal Exchange Mutation implementation
        :param individual: The individual to be mutated
        """
        # Select a random mutation rate value, if the value is greater than
        # the pre-determined mutation rate, skip mutation
        if random.random() > self.mutationRate:
            return
        # Get two random index values in the range of 0 and size of the genes
        # list minus 1
        mutate_index = random.sample(range(0, len(individual.genes) - 1), 2)
        # Get the the genes at the two random indexes selected
        gene_a, gene_b = (individual.genes[mutate_index[0]],
                          individual.genes[mutate_index[1]])
        # Swap the two genes at the random indexes selected
        individual.genes[mutate_index[0]] = gene_b
        individual.genes[mutate_index[1]] = gene_a

    def scrambleMutation(self, individual):
        """
        Scramble Mutation implementation
        :param individual: The individual to be mutated
        """
        # Select a random mutation rate value, if the value is greater than
        # the pre-determined mutation rate, skip mutation
        if random.random() > self.mutationRate:
            return
        # Get two indexes within the range of the size of the individual's
        # genes, sort the two indexes so they are in ascending order
        r_list = random.sample(range(0, individual.genSize), 2)
        r_list.sort()

        # Get the individual's genes
        chromosome_genes = individual.genes
        # Split the individuals gene list into three parts, the first being
        # from index 0 to the first randomly selected index
        list_pt1 = chromosome_genes[:r_list[0]]
        # The second part will be the range to be scrambled, starting with the
        # lowest random index solution and ending with the higher random index
        scramble_section = chromosome_genes[r_list[0]:r_list[1] + 1]
        # The third and last part will be the remaining genes after the higher
        # random index selection
        list_pt2 = chromosome_genes[r_list[1] + 1:]

        # Shuffle the genes between the lower and higher index selection
        random.shuffle(scramble_section)
        # Put all three parts of the individuals genes back together again
        list_pt1 += scramble_section + list_pt2
        # Assign the new genes to the individual
        individual.genes = list_pt1

    # Fitness Calculations

    def iteration_fitness_calculations(self, run_fitness_info):
        """
        Functions to determine fitness of current iteration.

        :param run_fitness_info: List of all the fitness values of the current
        iteration children genes
        """

        # Calculate iteration mean fitness
        run_mean = st.mean(run_fitness_info)
        self.run_mean.append(run_mean)

        # Calculate iteration median fitness
        run_median = st.median(run_fitness_info)
        self.run_median.append(run_median)

        # Calculate iteration smallest fitness (best)
        run_min = min(run_fitness_info)
        self.run_min.append(run_min)

        # Calculate iteration largest fitness (worst)
        run_max = max(run_fitness_info)
        self.run_max.append(run_max)

        # Send results of current iteration to file output function
        self.low_level_results_output_to_file(False, run_min, run_max,
                                              run_mean, run_median)

    # Output Control - High & Low level

    def low_level_results_output_to_file(self, new_file, *args):
        """
        Output low level results to CSV file
        :param new_file: If the file is to be created instead of added to
        :param args: The performance metrics to be written to file
        """
        complete_loc = self.fName.split('/')
        loc_size = len(complete_loc)
        f_name = complete_loc[loc_size - 1]

        file_name = ('TSP_GA_Config{}-{}-Run_{}-Low_Level_Results.csv'.format(
            self.configuration, f_name, self.current_run))

        if file_name not in self.csv_files:
            self.csv_files.append(file_name)

        if new_file:
            my_file = open(file_name, 'w', newline='')
            with my_file:
                writer = csv.writer(my_file,
                                    delimiter=',',
                                    quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Fitness Minimum', 'Fitness Max',
                                 'Fitness Mean', 'Fitness Median'])
        else:
            my_file = open(file_name, 'a', newline='')
            with my_file:
                writer = csv.writer(my_file,
                                    delimiter=',',
                                    quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerow(args)

    def high_level_results_output_to_file(self, new_file, *args):
        """
        Output high level results to CSV file
        :param new_file: If the file is to be created instead of added to
        :param args: The performance metrics to be written to file
        """
        complete_loc = self.fName.split('/')
        loc_size = len(complete_loc)
        f_name = complete_loc[loc_size - 1]

        file_name = ('TSP_GA_Config{}-{}-High_Level_Results.csv'.format(
            self.configuration, f_name))

        if file_name not in self.csv_files:
            self.csv_files.append(file_name)

        if new_file:
            my_file = open(file_name, 'w', newline='')
            with my_file:
                writer = csv.writer(my_file,
                                    delimiter=',',
                                    quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerow(
                    ['Run #', 'Total Run Time (s)',
                     'Avg. run time per iteration (s)', 'Fitness Minimum',
                     'Fitness Max', 'Fitness Range', 'Fitness Mean',
                     'Fitness Median', 'Fitness Improvement Likelihood %'])
        else:
            my_file = open(file_name, 'a', newline='')
            with my_file:
                writer = csv.writer(my_file,
                                    delimiter=',',
                                    quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
                writer.writerow(args)

    # Configuration Control

    def newGeneration_config_1(self):
        """
        Creating a new generation, Depending of your experiment you need to use
        the most suitable algorithms for:
        1. Select two candidates - Random
        2. Apply Crossover - Uniform
        3. Apply Mutation - Reciprocal
        """
        run_fitness = list()
        new_pop = list()
        for i in range(0, len(self.population)):
            # Get random parents from mating pool
            ind_a, ind_b = self.randomSelection()
            # Perform crossover
            child_a, child_b = self.uniformCrossover(ind_a, ind_b)
            # Perform mutation
            self.reciprocalExchangeMutation(child_a)
            self.reciprocalExchangeMutation(child_b)
            # Compute fitness of children
            child_a.computeFitness()
            child_b.computeFitness()
            # Keep the best of the two children
            if child_a.getFitness() > child_b.getFitness():
                new_pop.append(child_a)
                self.updateBest(child_a)
                run_fitness.append(child_a.getFitness())
            else:
                new_pop.append(child_b)
                self.updateBest(child_b)
                run_fitness.append(child_b.getFitness())

        self.population = list(new_pop)
        self.iteration_fitness_calculations(run_fitness)

    def newGeneration_config_2(self):
        """
        Creating a new generation, Depending of your experiment you need to use
        the most suitable algorithms for:
        1. Select two candidates - Random
        2. Apply Crossover - Cycle
        3. Apply Mutation - Scramble
        """
        run_fitness = list()
        new_pop = list()
        for i in range(0, len(self.population)):
            # Get random parents from mating pool
            ind_a, ind_b = self.randomSelection()
            # Perform crossover
            child_a, child_b = self.cycleCrossover(ind_a, ind_b)
            # Perform mutation and add to new population
            self.scrambleMutation(child_a)
            self.scrambleMutation(child_b)

            # Compute fitness of children
            child_a.computeFitness()
            child_b.computeFitness()
            # Keep the best of the two children
            if child_a.getFitness() > child_b.getFitness():
                new_pop.append(child_a)
                self.updateBest(child_a)
                run_fitness.append(child_a.getFitness())
            else:
                new_pop.append(child_b)
                self.updateBest(child_b)
                run_fitness.append(child_b.getFitness())

        self.population = list(new_pop)
        self.iteration_fitness_calculations(run_fitness)

    def newGeneration_config_3(self):
        """
        Creating a new generation, Depending of your experiment you need to use
        the most suitable algorithms for:
        1. Select two candidates - Roulette
        2. Apply Crossover - Uniform
        3. Apply Mutation - Reciprocal
        """
        run_fitness = list()
        new_pop = list()
        for i in range(0, len(self.population)):
            # Get random parents from mating pool
            ind_a = self.rouletteWheel()
            ind_b = self.rouletteWheel()
            # Perform crossover
            child_a, child_b = self.uniformCrossover(ind_a, ind_b)
            # Perform mutation
            self.reciprocalExchangeMutation(child_a)
            self.reciprocalExchangeMutation(child_b)

            # Compute fitness of children
            child_a.computeFitness()
            child_b.computeFitness()
            # Keep the best of the two children
            if child_a.getFitness() > child_b.getFitness():
                new_pop.append(child_a)
                self.updateBest(child_a)
                run_fitness.append(child_a.getFitness())
            else:
                new_pop.append(child_b)
                self.updateBest(child_b)
                run_fitness.append(child_b.getFitness())

        self.population = list(new_pop)
        self.iteration_fitness_calculations(run_fitness)

    def newGeneration_config_4(self):
        """
        Creating a new generation, Depending of your experiment you need to use
        the most suitable algorithms for:
        1. Select two candidates - Roulette
        2. Apply Crossover - Cycle
        3. Apply Mutation - Reciprocal
        """
        run_fitness = list()
        new_pop = list()
        for i in range(0, len(self.population)):
            # Get random parents from mating pool
            ind_a = self.rouletteWheel()
            ind_b = self.rouletteWheel()
            # Perform crossover
            child_a, child_b = self.cycleCrossover(ind_a, ind_b)
            # Perform mutation
            self.reciprocalExchangeMutation(child_a)
            self.reciprocalExchangeMutation(child_b)

            # Compute fitness of children
            child_a.computeFitness()
            child_b.computeFitness()
            # Keep the best of the two children
            if child_a.getFitness() > child_b.getFitness():
                new_pop.append(child_a)
                self.updateBest(child_a)
                run_fitness.append(child_a.getFitness())
            else:
                new_pop.append(child_b)
                self.updateBest(child_b)
                run_fitness.append(child_b.getFitness())

        self.population = list(new_pop)
        self.iteration_fitness_calculations(run_fitness)

    def newGeneration_config_5(self):
        """
        Creating a new generation, Depending of your experiment you need to use
        the most suitable algorithms for:
        1. Select two candidates - Roulette
        2. Apply Crossover - Cycle
        3. Apply Mutation - Scramble
        """
        run_fitness = list()
        new_pop = list()
        for i in range(0, len(self.population)):
            # Get random parents from mating pool
            ind_a = self.rouletteWheel()
            ind_b = self.rouletteWheel()
            # Perform crossover
            child_a, child_b = self.cycleCrossover(ind_a, ind_b)
            # Perform mutation
            self.scrambleMutation(child_a)
            self.scrambleMutation(child_b)

            # Compute fitness of children
            child_a.computeFitness()
            child_b.computeFitness()
            # Keep the best of the two children
            if child_a.getFitness() > child_b.getFitness():
                new_pop.append(child_a)
                self.updateBest(child_a)
                run_fitness.append(child_a.getFitness())
            else:
                new_pop.append(child_b)
                self.updateBest(child_b)
                run_fitness.append(child_b.getFitness())

        self.population = list(new_pop)
        self.iteration_fitness_calculations(run_fitness)

    def newGeneration_config_6(self):
        """
        Creating a new generation, Depending of your experiment you need to use
        the most suitable algorithms for:
        1. Select two candidates - Best/2nd Best
        2. Apply Crossover - Uniform
        3. Apply Mutation - Scramble
        """
        run_fitness = list()
        new_pop = list()
        ind_a = self.best.copy()
        ind_b = self.second_best.copy()
        for i in range(0, len(self.population)):
            child_a, child_b = self.uniformCrossover(ind_a, ind_b)
            # Perform mutation
            self.scrambleMutation(child_a)
            self.scrambleMutation(child_b)

            # Compute fitness of children
            child_a.computeFitness()
            child_b.computeFitness()
            # Keep the best of the two children
            if child_a.getFitness() > child_b.getFitness():
                new_pop.append(child_a)
                self.updateBest(child_a)
                run_fitness.append(child_a.getFitness())
            else:
                new_pop.append(child_b)
                self.updateBest(child_b)
                run_fitness.append(child_b.getFitness())

        self.population = list(new_pop)
        self.iteration_fitness_calculations(run_fitness)

    def newGeneration_config_7(self):
        """
        Creating a new generation, Depending of your experiment you need to use
        the most suitable algorithms for:
        1. Select two candidates - Best/Roulette
        2. Apply Crossover - Uniform
        3. Apply Mutation - Scramble
        4. Elite Survival - 20%
        """
        run_fitness = list()
        new_pop = list()
        for i in range(0, len(self.population)):
            # Get best and 2nd best from population, picking best and 2nd best
            # as they are found
            ind_a = self.best.copy()
            ind_b = self.rouletteWheel()

            child_a, child_b = self.uniformCrossover(ind_a, ind_b)
            # Perform mutation
            self.scrambleMutation(child_a)
            self.scrambleMutation(child_b)

            # Compute fitness of children
            child_a.computeFitness()
            child_b.computeFitness()
            # Keep the best of the two children
            if child_a.getFitness() > child_b.getFitness():
                new_pop.append(child_a)
                self.updateBest(child_a)
                run_fitness.append(child_a.getFitness())
            else:
                new_pop.append(child_b)
                self.updateBest(child_b)
                run_fitness.append(child_b.getFitness())

        self.population = list(new_pop)
        self.iteration_fitness_calculations(run_fitness)

    # GA RUNTIME FUNCTIONALITY

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """
        self.updateMatingPool()
        if self.configuration == 1:
            self.newGeneration_config_1()
        if self.configuration == 2:
            self.newGeneration_config_2()
        if self.configuration == 3:
            self.newGeneration_config_3()
        if self.configuration == 4:
            self.newGeneration_config_4()
        if self.configuration == 5:
            self.newGeneration_config_5()
        if self.configuration == 6:
            self.newGeneration_config_6()
        if self.configuration == 7:
            self.newGeneration_config_7()

    def reset_for_next_run(self):
        self.run_mean.clear()
        self.run_median.clear()
        self.run_max.clear()
        self.run_min.clear()
        self.run_max.clear()
        self.initPopulation()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        self.improvement_cnt = 0
        self.high_level_results_output_to_file(new_file=True)
        for i in range(0, self.run_count):
            print("Starting run #{}".format(self.current_run))
            self.iteration, self.improvement_cnt = 0, 0
            self.low_level_results_output_to_file(new_file=True)

            start = time.time()
            while self.iteration < self.maxIterations:
                self.GAStep()
                self.iteration += 1
            end = time.time()

            self.high_level_results_output_to_file(
                False, self.current_run, (end - start),
                ((end - start) / self.maxIterations), min(self.run_min),
                max(self.run_max), (max(self.run_max) - min(self.run_min)),
                st.mean(self.run_mean), st.median(self.run_median),
                (self.improvement_cnt / self.maxIterations) * 100)

            if self.elite:
                fitness_values = list()
                self.population.append(self.best)
                self.population.append(self.second_best)
                keep_cnt = int(self.popSize / 5)
                for ind in self.population:
                    fitness_values.append(ind.fitness)
                elite_keep = nsmallest(keep_cnt, fitness_values)
                for ind in self.population:
                    if ind.fitness <= max(elite_keep):
                        self.elite_pool.append(ind.copy())

            print("Total iterations: ", self.iteration)
            print("Best Solution: ", self.best.getFitness())
            print("================================================")

            if self.current_run < self.run_count:
                self.reset_for_next_run()
            self.current_run += 1

        print("{} Runs completed, please see high and low level output .csv "
              "files for performance breakdown".format(self.run_count))


run_count = 3
iterations = 300
mutation_rate = 0.1
pop_size = 100
elite = False

if len(sys.argv) < 2:
    print("Error - Incorrect input")
    print("Expecting $ python TSP_Solution_MMA.py [in_file] [configuration]")
    sys.exit(0)

in_file = sys.argv[1]
configuration = int(sys.argv[2])

ga = BasicTSP(_fName=in_file, _popSize=pop_size, _mutationRate=mutation_rate,
              _maxIterations=iterations, _configuration=configuration,
              _run_count=run_count, _elite=elite)
ga.search()
