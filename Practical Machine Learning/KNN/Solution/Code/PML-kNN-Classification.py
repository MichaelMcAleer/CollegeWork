"""
Practical Machine Learning
Assignment 1 - K-Nearest Neighbour Algorithms
Solution by Michael McAleer (R00143621)
"""
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
    data = np.rint(np.genfromtxt(file_in, delimiter=','))

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


def output_to_file(new_file, measurement=None, k_neighbours=None, *args):
    """
    Output results to CSV file.

    :param new_file: (boolean) If the file is to be created is a new file.
    :param measurement: (string) The distance measurement type, valid values
    are 'euclidean', 'manhattan', 'minkowski', 'hamming', and 'heterogeneous'.
    :param k_neighbours: (int) The amount of neighbours to return.
    :param args: (list) The performance metrics to be written to file.
    """
    file_name = ('PML-kNN-{}.csv'.format(measurement))

    if new_file:
        # File is new, create the file and add the relevant headings
        my_file = open(file_name, 'w', newline='')
        with my_file:
            writer = csv.writer(my_file,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['k Value', '% Correct',
                             '% Incorrect'])
    else:
        # File already exists, append performance metrics to the next row
        p_correct = (args[0] / args[2]) * 100
        p_incorrect = (args[1] / args[2]) * 100

        my_file = open(file_name, 'a', newline='')
        with my_file:
            writer = csv.writer(my_file,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow([k_neighbours, p_correct, p_incorrect])


def output_to_screen(correct, incorrect, count, run_time):
    """
    Output results to screen.

    :param correct: (float) Percentage of correct predictions.
    :param incorrect: (float) Percentage of incorrect predictions.
    :param count: (int) The amount of instances for comparison in the test set.
    :param run_time: (float) The time in seconds it took to complete predict
    severity of all instances in the test set.
    """
    print("============================")
    print("RUN STATISTICS")
    print("\tTotal instances queried: {}".format(count))
    print("\tRun time: {}s".format(run_time))
    print("\tCorrect: {}/{} => {}%".format(
        correct, count, ((correct / count) * 100)))
    print("\tIncorrect: {}/{} => {}%".format(
        incorrect, count, ((incorrect / count) * 100)))
    print("============================")


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
    arr_a = f_query[:5]
    arr_b = t_data[:, :5]

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

    elif measurement == 'hamming':
        # Return True/False numpy array for each feature/training set feature
        hamming_values = (arr_a != arr_b)
        # Count all True (1) values in hamming values array
        distances = np.count_nonzero(hamming_values, axis=1)
        # Get the results indices by argsort
        distances_indices = np.argsort(distances)

    elif measurement == 'heterogeneous':
        # Make clones of each array. Both arrays have age column removed for
        # hamming distance calculation on remaining discrete values
        hamming_arr1 = np.array(np.delete(arr_a, 1))
        hamming_arr2 = np.array(np.delete(arr_b, 1, 1))

        # Make additional clones of each array. Only the age columns have been,
        # retained for continuous data distance calculation
        euclidean_arr1 = np.array(arr_a[1])
        euclidean_arr2 = np.array(arr_b[:, [1]])

        # Calculate Hamming distances of discrete data
        h_values = (hamming_arr1 != hamming_arr2)
        h_distances = np.count_nonzero(h_values, axis=1)
        # Calculate Euclidean distances of continuous data
        e_values = np.sum((euclidean_arr1 - euclidean_arr2) ** 2, axis=1)
        e_distances = np.sqrt(e_values)

        # Add the hamming distances to the euclidean distances to get the total
        # distances between the initial feature and the training set features
        distances = h_distances + e_distances
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


def k_nearest_neighbour(measurement, f_query, t_data, k_neighbours, p_value):
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

    return distances[distances_indices[:k_neighbours]], t_data[
        distances_indices[:k_neighbours]]


def majority_vote_determination(feature_query, neighbours, t_data):
    """
    Given the closest neighbours, determine the most likely severity of the
    feature query.

    :param feature_query: (np-array)
    :param neighbours: (np-array)
    :param t_data: (np-array)
    :return:
    Actual severity: (int) The actual severity of the feature query being
        tested
    Predicted severity: (int) The predicted severity of the feature query being
        tested
    """
    # Determine the severity of feature query
    actual_severity = feature_query[5]

    # Calculate neighbour severity by count, returns a dict with counts of
    # 0 & 1 values
    neighbour_results = dict(
        zip(*np.unique(neighbours[:, 5], return_counts=True)))

    # Determine the most prominent severity by calculating most occurring
    # neighbour severity, if neighbour severity is evenly distributed, return
    # both severities
    prediction = ([k for k, v in neighbour_results.items() if
                   v == max(neighbour_results.values())])

    # If there is an outright prediction, return this value
    if len(prediction) == 1:
        prediction = prediction[0]
    # Else there is a tie between one or more classes, count the most
    # class and return that as prediction
    else:
        severities, count = np.unique(t_data[:, 5], return_counts=True)
        most_occurrences = np.argsort(-count)
        prediction = severities[most_occurrences[0]]

    return actual_severity, prediction


def distance_weighted_determination(measurement, k_neighbours, f_query,
                                    t_data, p_value):
    """
    Determine distance weighted vote between feature and k nearest neighbours

    :param measurement: (string) The measurement type
    :param k_neighbours: (int) The amount of neightbours to return
    :param f_query: (np-array) The feature query to compare
    :param t_data: (np-array) The training dataset with all known features
    :param p_value: (int) If Minkowski distance is used, this is the power of
    value
    :return:
    Actual severity: (int) The actual severity of the feature query being
        tested
    Predicted severity: (int) The predicted severity of the feature query being
        tested
    """
    # Initialise severity lists to store neighbours weights
    severity_0 = list()
    severity_1 = list()

    # No Nearest Neighbours specified, run weights against all features in
    # training set
    if k_neighbours <= 0:
        # Get the distances and distances sorted by numpy argsort
        distances, distances_indices = calculate_distance(
            measurement, f_query, t_data, p_value)

        # For each index in the returned indices
        for index in distances_indices:
            # Get the corresponding instance from the training data
            training_feature = training_data[index]
            # Get the classification of the instance from the training data
            training_classification = training_feature[5]
            # Get the distance between the feature query and the instance from
            # the training data
            training_distance = distances[index]

            # If the training data instance is classified as 0, add the
            # distance of the instance to the severity_0 list
            if training_classification == 0:
                severity_0.append(1 / training_distance)
            # Else the training data instance is classified as 0, add the
            # distance to the severity_1 list
            else:
                severity_1.append(1 / training_distance)

    # Neighbour count specified, run weights only against k nearest neighbours
    else:
        # Get the distances and distances sorted by numpy argsort
        distances, closest_neighbours = k_nearest_neighbour(
            measurement, f_query, t_data, k_neighbours, p_value)
        # For each of the k nearest neighbours returned
        for i in range(0, k_neighbours):
            c_neighbour = closest_neighbours[i]
            # If the neighbour is classified as 0, add the distance of the
            # instance to the severity_0 list
            if c_neighbour[5] == 0:
                severity_0.append(1 / distances[i])
            # Else the neighbour is classified as 1, add the distance of the
            # instance to the severity_1 list
            elif c_neighbour[5] == 1:
                severity_1.append(1 / distances[i])

    # Get the severity classification of the feature from the test set being
    # queried
    actual_severity = f_query[5]
    prediction = None

    # If the weighted distances of severity 0 neighbours is greater than the
    # weight of the severity 1 neighbours, the predicted classification of the
    # test set feature is 0
    if sum(severity_0) > sum(severity_1):
        prediction = 0
    # If the weighted distances of severity 1 neighbours is greater than the
    # weight of the severity 0 neighbours, the predicted classification of the
    # test set feature is 1
    elif sum(severity_0) < sum(severity_1):
        prediction = 1
    # If the weighted distances between severity 0 and severity 1 are equal,
    # determine the classification which occurs most frequently in the training
    # data set and return that as the prediction.
    elif sum(severity_0) == sum(severity_1):
        severities, count = np.unique(t_data[:, 5], return_counts=True)
        most_occurrences = np.argsort(-count)
        prediction = severities[most_occurrences[0]]

    return actual_severity, prediction


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
    correct, incorrect, count = 0, 0, 0
    actual_severity, predicted_severity = None, None
    start = time.time()

    # For each feature in the test data set
    for feature in f_data:
        # If the kNN is majority vote based...
        if not weighted:
            # Get the nearest k nearest neighbours
            __, neighbours = k_nearest_neighbour(
                measurement, feature, t_data, k_neighbours, p_value)
            # Given the k nearest neighbours, determine the predicted class
            # by the most frequently occurring neighbour class
            actual_severity, predicted_severity = (
                majority_vote_determination(feature, neighbours, t_data))

        # If the kNN is weighted distance vote based...
        elif weighted:
            # Run the distance weighted algorithm for determining the test
            # instance classification
            actual_severity, predicted_severity = (
                distance_weighted_determination(measurement, k_neighbours,
                                                feature, t_data, p_value))

        # If the test data set instance class matches the predicted class
        # then increment the correct prediction counter by 1
        if actual_severity == predicted_severity:
            correct += 1
        # Else the test data set instance class does not match the predicted
        # class, increment the incorrect prediction counter by 1
        else:
            incorrect += 1
        # Increment run counter by 1
        count += 1
    # End timer
    end = time.time()
    # Output the results to screen
    output_to_screen(correct, incorrect, count, (end - start))
    # If the results need to be output to file
    if to_file:
        output_to_file(False, measurement, k_neighbours, correct, incorrect,
                       count)


# kNN Testing
test_data = read_csv_normalise('ClassificationTestData.csv',
                               normalise=False)
training_data = read_csv_normalise('ClassificationTrainingData.csv',
                                   normalise=False)

output_to_file(True, measurement='euclidean')
run_knn(f_data=test_data,
        t_data=training_data,
        measurement='euclidean',
        k_neighbours=3,
        weighted=True,
        p_value=0,
        to_file=True)
