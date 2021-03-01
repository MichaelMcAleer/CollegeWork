# ------------------------------------------
# Big Data Processing
# Assignment 1 - Spark Core & Spark SQL
# Part 1 - Spark Core
# Michael McAleer R00143621
# ------------------------------------------
import pyspark


def process_line(line):
    """Process a line from the input data rdd, cleans and returns tuple of
    values.

    :param line: raw line data -- str
    :return: line values -- tuple (str * len(params))
    """
    # Remove new line character
    line = line.replace('\n', '')
    # Split line on delimiter
    params = line.split(';')
    # If the param count is 7 return tuple of line values else empty tuple
    return tuple(params) if len(params) == 7 else ()


###############################################################################

def ex1(rdd):
    """Exercise 1: Total amount of entries in the dataset.

    :param rdd: processed input rdd -- pyspark rdd
    """
    # Count the total amount of entries in the rdd and print
    print('- Total dataset entries: {c}'.format(c=rdd.count()))


###############################################################################

def ex2(rdd):
    """Exercise 2: Number of Coca-cola bikes stations in Cork.

    :param rdd: processed input rdd -- pyspark rdd
    """
    # Extract the station names from entire rdd
    extracted_rdd = rdd.map(lambda x: x[1])
    # Get all the unique values from the rdd
    distinct_rdd = extracted_rdd.distinct()
    # Output the amount of unique values
    print('- Total bike stations in Cork: {c}'.format(c=distinct_rdd.count()))


###############################################################################

def ex3(rdd):
    """Exercise 3: List of Coca-Cola bike stations.

    Note: It would be more efficient to put this in ex2() and collect from
    there so the distinct value count does not need to be computed twice. In a
    real world scenario this approach would be taken.

    :param rdd: processed input rdd -- pyspark rdd
    """
    # Extract the station names from entire rdd
    extracted_rdd = rdd.map(lambda x: x[1])
    # Get all the unique elements from the rdd
    distinct_rdd = extracted_rdd.distinct()
    # Collect all the unique elements into a list
    result = distinct_rdd.collect()
    # Print each of the elements individually
    print('Cork Bike Station List:')
    for n, val in enumerate(result):
        print('- {i}: {station}'.format(i=n + 1, station=val))


###############################################################################

def ex4(rdd):
    """Exercise 4: Sort the bike stations by their longitude (East to West).

    :param rdd: processed input rdd -- pyspark rdd
    """
    # Extract the station name and longitude from the input rdd
    extracted_rdd = rdd.map(lambda x: tuple([x[1], float(x[2])]))
    # Get the unique elements from the rdd
    distinct_rdd = extracted_rdd.distinct()
    # Sort the stations by longitude from East to West (descending)
    result = distinct_rdd.sortBy(lambda x: x[1], ascending=False).collect()
    # Print each of the elements
    print('Cork Bike Stations (East -> West):')
    for val in result:
        print('- {station}: {pos}'.format(station=val[0], pos=val[1]))


###############################################################################

def ex5(rdd):
    """Exercise 5: Average number of bikes available at Kent Station.

    :param rdd: processed input rdd -- pyspark rdd
    """
    # Filter the input rdd to extract only elements which match 'Kent Station'
    filter_rdd = rdd.filter(
        lambda x: True if x[1] == 'Kent Station' else False)
    # Transform the rdd to float bike counts
    bike_cnt_rdd = filter_rdd.map(lambda x: float(x[5]))
    # Aggregate the bike counts and keep count of elements processed
    result = bike_cnt_rdd.aggregate(
        (0, 0),
        lambda x, y: tuple([(x[0] + y), (x[1] + 1)]),
        lambda x, y: tuple([(x[0] + y[0]), x[1] + y[1]]))
    # Print the results
    print('- Total sum of bikes: {s}'.format(s=int(result[0])))
    print('- Total intervals: {i}'.format(i=result[1]))
    print('- Average bike count: {a}'.format(a=result[0] / result[1]))


###############################################################################

if __name__ == '__main__':
    # Set the location of the dataset directory
    FILE_STORE = '/FileStore/tables/cork_bike_data'
    # Configure the Spark Context
    sc = pyspark.SparkContext.getOrCreate()
    # Set log level
    sc.setLogLevel('WARN')
    # Load the dataset into a RDD
    raw_rdd = sc.textFile('{data}/*.csv'.format(data=FILE_STORE))
    # Process each line to get the relevant info as a tuple of values
    input_rdd = raw_rdd.map(process_line)
    # Persist the RDD to memory for re-use
    input_rdd.cache()
    # Call the functions
    for i, ex in enumerate([ex1, ex2, ex3, ex4, ex5]):
        print('\n#-------------#\n'
              '| Exercise: {e} |\n'
              '#-------------#'.format(e=i + 1))
        ex(input_rdd)
