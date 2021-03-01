>>Practical Machine Learning
>Assignment 1
>Michael McAleer (R00143631)

To run either the classification or regression algorithm execute the respective
.py file from the command line using Python 3.6.x or 3.7. The only required
dependency for these algorithms to run is the NumPy library, this can be
installed via PIP.

To run the script enter the command:

$ python3 <classification/regression .py file>

There are a number of input parameters which can be altered in both python files
to change the measurement calculation, k-value, p-value for minkowski, select
weighted vs majority, and output results to file.

Example:
Both algorithm files have the below function call at the bottom of the file,
each of the parameters can be changed to suit the purposes of the user.

run_knn(f_data=test_data,
        t_data=training_data,
        measurement='euclidean',
        k_neighbours=3,
        weighted=True,
        p_value=1.5,
        to_file=True)

If normalised data is required, this is specified in the function just above
the function run_knn:

test_data = read_csv_normalise('RegressionTestData.csv',
                               normalise=False)
training_data = read_csv_normalise('RegressionTrainingData.csv',
                                   normalise=False)

Change the 'normalise' parameter to 'True' if you would like to normalise the
data before running the algorithm.
