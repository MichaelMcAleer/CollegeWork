import csv
import numpy as np
import time
import warnings

# Ignore RuntimeWarning: divide by zero encountered in double_scalars when
# determining distance weighted nearest neighbour classification, warning can
# be safely ignored
warnings.filterwarnings("ignore")


def read_csv_normalise(file_in, normalise=True):
    """
    Read a CSV file to extract contents, delimiter set to comma ','.

    :param file_in: (string) The CSV filename to be read.
    :param normalise: (boolean) If the data needs to be normalised.
    :return: (np-array) The data from the CSV file.
    """
    # There are a lot of values in the original data set which are floating
    # point decimal numbers where they should be discrete integers in specific
    # range, np.ring() will round to the nearest integer to remove these
    # incorrect discrete feature values
    data = np.genfromtxt(file_in, delimiter=',')

    def _normalise(np_array):
        """
        Normalise array values using  calculation:
            new_value = (old_value - min_value) / (max_value - min_value)

        :param np_array: (np-array) The data from the CSV file.
        :return: (np-array) The normalised data from the CSV file.
        """
        # Get the min and max values of each column in the data set
        min_val = np.min(np_array, axis=0)
        max_val = np.max(np_array, axis=0)
        return (np_array - min_val) / (max_val - min_val)

    if normalise:
        return _normalise(data)
    else:
        return data


def output_to_screen(r2, run_time):
    """
    Output results to screen.

    :param r2: (Float) The r2 value
    :param run_time: (Float) The total run-time of the ML-algorithm
    """
    print("============================")
    print("RUN STATISTICS")
    print("\tRun time: {}s".format(run_time))
    print("\tR2 Value: {}".format(r2))
    print("============================")


def output_to_file(new_file, measurement, k_neighbours=None, r2=None,
                   run_time=None, p_value=None):
    """
    Output results to CSV file.

    :param new_file: (boolean) If the file is to be created is a new file.
    :param measurement: (string) The distance measurement type, valid values
    are 'euclidean', 'manhattan', 'minkowski', 'hamming', and 'heterogeneous'.
    :param k_neighbours: (int) The amount of neighbours to return.
    :param r2: (float) The calculated r2 value.
    """
    file_name = ('PML-kNN-Regression-{}-{}-weighted-standard.csv'.format(
        measurement, p_value))

    if new_file:
        # File is new, create the file and add the relevant headings
        my_file = open(file_name, 'w', newline='')
        with my_file:
            writer = csv.writer(my_file,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['k Value', 'r2 Value', 'Run Time'])
    else:

        my_file = open(file_name, 'a', newline='')
        with my_file:
            writer = csv.writer(my_file,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow([k_neighbours, r2, run_time])


def calculate_distance(measurement, f_query, t_data, p_value=None):
    """
    Calculate the distance between a feature and all features in a training
    set using the specified measurement.

    :param measurement: (string) The distance measurement type, valid values
    are 'euclidean', 'manhattan', 'minkowski', 'hamming' and 'heterogeneous'.
    :param f_query: (np-array) The feature query to compare.
    :param t_data: (np-array) The training data set with all known features.
    :param p_value: (int) If Minkowski distance, this is the power value.
    :return:
    distances (np-array): The distances between the feature and all features in
        the training set.
    distances_results (np-array): The indices of the distances ordered by numpy
        argsort of kind 'quicksort'.
    """
    # Specify features to use in measurement calculations, drop severity as it
    # is not part of the distance calculation and will be used later for
    # feature prediction
    arr_a = f_query[:11]
    arr_b = t_data[:, :11]

    if measurement == 'euclidean':
        # Get the sum of all squared subtractions
        values = np.sum((arr_a - arr_b) ** 2, axis=1)
        # Get the square root
        distances = np.sqrt(values)
        # Get the results indices by argsort
        distances_indices = np.argsort(distances)

    elif measurement == 'manhattan':
        # Get the sum of all squared subtractions
        distances = np.sum(np.abs(arr_a - arr_b), axis=1)
        # Get the results indices by argsort
        distances_indices = np.argsort(distances)

    elif measurement == 'minkowski':
        # Get the sum of all squared to the power of p_value subtractions.
        # Absolute is used here to prevent NumPy runtime warning for invalid
        # value encountered in power for distances calculation
        value = np.sum((abs(arr_a - arr_b) ** p_value), axis=1)
        # Calculate distances by multiplying values from previous equation
        # by 1 over the p_value
        distances = value ** (1 / p_value)
        # Get the results indices by argsort
        distances_indices = np.argsort(distances)

    else:
        raise Exception("An unknown distance calculation type has been "
                        "specified, exiting application.")

    if distances.size == 0 or distances_indices.size == 0:
        raise Exception("There has been a problem calculating the distances "
                        "or sorting the distances via argsort, exiting "
                        "application.")

    return distances, distances_indices


def k_nearest_neighbour(measurement, f_query, t_data, k_neighbours,
                        p_value=None):
    """
    Calculate the k nearest neighbours, where k is the amount of neighbours
    to return.

    :param measurement: (string) The measurement type.
    :param f_query: (np-array) The feature query to compare.
    :param t_data: (np-array) The training data set with all known features.
    :param k_neighbours: (int) The amount of neighbours to return.
    :param p_value: (int) If Minkowski distance is used, this is the power of
    value.
    :return:
    distances: (list) The distances from the feature query instance to each
        of the k nearest neighbours
    neighbours: (list) The k nearest neighbours from the training data set
    """
    # Calculate the distances between the test instance and the training data
    distances, distances_indices = calculate_distance(
        measurement, f_query, t_data, p_value)

    if k_neighbours <= 0:
        return distances, t_data[distances_indices]
    else:
        return distances[distances_indices[:k_neighbours]], t_data[
            distances_indices[:k_neighbours]]


def majority_vote_determination(neighbours):
    """
    Given the closest neighbours, determine the most likely severity of the
    feature query by calculating average of k-NN r values.

    :param neighbours: (np-array) The k closes neighbours to the test query
    :return: prediction: (float) The predicted r value of the test query
    """
    # Get the r values of the k closest neighbours
    neighbour_results = neighbours[:, 12]

    # Predict the r-value of the test query by summing all values of k nearest
    # neighbours and dividing by the amount of neighbours
    prediction = sum(neighbour_results) / len(neighbours)

    return prediction


def weighted_neighbour_calculation(distances, neighbours):
    """
    Given the closest neighbours, determine the most likely severity of the
    feature query by calculating the weighted r values of k nearest neighbours.

    :param distances: (np-array) The distances to the kNN
    :param neighbours: (np-array) The nearest neighbours from the training data
    :return: prediction: (float) The predicted r value of the test query
    """
    # Initialise the lists used to store neighbour weights, and the fractions
    # to be used in calculating the test feature r value
    neighbour_weights = list()
    neighbour_distance_fractions = list()

    # For each of the neighbours
    for i in range(0, len(neighbours)):
        # Get the neighbour
        neighbour = neighbours[i]
        # Calculate the weight of the neighbour and append to weight list
        neighbour_weights.append((1 / distances[i]) * neighbour[12])
        # Append the neighbour distance fraction to the fraction list
        neighbour_distance_fractions.append(1 / distances[i])

    # Calculate the predicted r value of the test query by summing up all
    # values in both lists and dividing the weights by the distance fractions
    prediction = np.sum(np.array(neighbour_weights)) / np.sum(
        np.array(neighbour_distance_fractions))

    return prediction


def calculate_r2(residual_differences, t_data):
    """
    Calculate the r2 value.

    :param residual_differences: (np-array) The residual differences of all
    predicted values
    :param t_data: (np-array) The training data set
    :return:
    """
    # >> Calculate the sum of the squared residuals
    # First get the squared values of all residual differences
    # Sum up all the squared residual differences to get the final value
    ssr = np.sum(np.square(np.array(residual_differences)))

    # >> Calculate the total sum of squares
    # First calculate the mean value for the regression variables
    tss_average = np.mean(t_data[:, 12])
    # Subtract the mean value from each of the regression variables
    tss = np.subtract(t_data[:, 12], tss_average)
    # Get the sum of all squared regression variable differences to get total
    # sum of squares
    tss = np.sum(np.square(tss))

    # >> Get the r2 value using 1 minus the sum of the squared residuals
    # divided by the total sum of squares
    r2 = 1 - (ssr / tss)

    return r2


def run_knn(f_data, t_data, measurement, k_neighbours, weighted=False,
            p_value=None, to_file=False):
    """
    Run the k nearest neighbour algorithm on a test data set and training data
    set. The type of distances used and majority/weighted voting is specified
    by input params.

    :param f_data: (np-array) The feature data to be compared
    :param t_data: (np-array) The training data set with all known features
    :param measurement: (string) The measurement type
    :param k_neighbours: (int) The amount of neighbours to return
    :param weighted: (boolean) If distance weighted kNN is required
    :param p_value: (int) If Minkowski distance is used, this is the power of
    value
    :param to_file: (boolean) If the results are written to file
    """
    # Initialise counters and start timer
    actual_val, predicted_val = None, None
    residual_differences = list()
    start = time.time()

    # For each feature in the test data set
    for feature in f_data:
        # Get the nearest k nearest neighbours
        distances, neighbours = k_nearest_neighbour(
            measurement, feature, t_data, k_neighbours, p_value)
        # Get the target r value
        actual_val = feature[12]

        # If the kNN is majority vote based...
        if not weighted:
            # Given the k nearest neighbours, determine the predicted r value
            # by the average of neighbour r values
            predicted_val = majority_vote_determination(neighbours)

        if weighted:
            # Given the k nearest neighbours, determine the predicted r value
            # by neighbour weights
            predicted_val = weighted_neighbour_calculation(distances,
                                                           neighbours)

        residual_differences.append(actual_val - predicted_val)

    r2 = calculate_r2(residual_differences, t_data)

    # End timer
    end = time.time()
    run_time = end - start
    # Output the results to screen
    output_to_screen(r2, (end - start))

    if to_file:
        output_to_file(False, measurement, k_neighbours, r2, run_time, p_value)


test_data = read_csv_normalise('RegressionTestData.csv',
                               normalise=False)
training_data = read_csv_normalise('RegressionTrainingData.csv',
                                   normalise=False)

run_knn(f_data=test_data,
        t_data=training_data,
        measurement='euclidean',
        k_neighbours=3,
        weighted=True,
        p_value=1.5,
        to_file=True)
