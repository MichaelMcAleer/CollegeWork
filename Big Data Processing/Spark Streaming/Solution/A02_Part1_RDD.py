# -----------------------------------------------------
# Big Data Processing
# Assignment 2 - Spark Streaming & Structured Streaming
# Part 1 - Spark Core
# Michael McAleer R00143621
# -----------------------------------------------------
import pyspark
import time

SEP = '-------------------------------------------'


# ------------------------------------------
# Exercise 1
# ------------------------------------------
def ex1(rdd):
    """Number of measurements (lines).

    :param rdd: processed input rdd -- pyspark rdd
    """
    # Count the total amount of entries in the rdd and print
    print('Total dataset entries: {c}'.format(c=rdd.count()))


# ------------------------------------------
# Exercise 2
# ------------------------------------------
def ex2(rdd, station_id):
    """Amount of days in the calendar year (01/09/2016 - 31/08/2017) for which
    data is collected (240101 - UCC WGB – Lotabeg).

    :param rdd: processed input rdd -- pyspark rdd
    :param station_id: station number -- int
    """
    # Filter the input RDD to return entries matching station_id
    filter_rdd = rdd.filter(lambda x: True if x[0] == station_id else False)
    # Map the RDD values so date values are isolated
    mapped_rdd = filter_rdd.map(lambda x: x[4])
    # Get the distinct dates and count total
    distint_count = mapped_rdd.distinct().count()
    # Output results
    print(distint_count)


# ------------------------------------------
# Exercise 3
# ------------------------------------------
def ex3(rdd, station_id):
    """For given station (240561 - UCC WGB – Curraheen) get the amount
    of measurements where:
        - scheduled_time >= expected_arrival_time (Behind)
        - scheduled_time < expected_arrival_time (Ahead)

    :param rdd: processed input rdd -- pyspark rdd
    :param station_id: station number -- int
    """
    # Filter the input RDD to return entries matching station_id
    filter_rdd = rdd.filter(lambda x: True if x[0] == station_id else False)
    # Map the RDD values to indicate if bus queried is ahead or behind its
    # scheduled time
    mapped_rdd = filter_rdd.map(
        lambda x: (
            tuple(['Ahead', 1]) if x[6] >= x[7] else tuple(['Behind', 1])))
    # Reduce the RDD by key so instances of each key are counted, giving the
    # amount of measurements for both behind and ahead keys
    count_rdd = mapped_rdd.reduceByKey(lambda x, y: x + y)
    # Collect the RDD values for output
    result = count_rdd.collect()
    # Output results
    print(result)


# ------------------------------------------
# Exercise 4
# ------------------------------------------
def ex4(rdd, station_id):
    """Get a list with the scheduled_time values found for each day of the
    week (241111 - CIT Tech. Park – Lotabeg).

    :param rdd: processed input rdd -- pyspark rdd
    :param station_id: station number -- int
    """
    # Filter the input RDD to return entries matching station_id
    filter_rdd = rdd.filter(lambda x: True if x[0] == station_id else False)
    # Map the RDD values so day_of_week and scheduled_time are isolated
    mapped_rdd = filter_rdd.map(lambda x: tuple([x[3], x[6]]))
    # Combine the mapped RDD values by key so a list of unique times for each
    # day is returned after the final step in the process
    #   Pt1 - Convert key value to a list
    #   Pt2 - Merge any new values to the existing list for a key if the key
    #         exists
    #   Pt3 - Convert lists to sets and get union of both, convert resulting
    #         set to list for output
    combined_rdd = mapped_rdd.combineByKey(
        lambda x: [x],
        lambda accum, y: accum + [y],
        lambda fin_a, fin_b: list(set(fin_a).union(set(fin_b))))
    # Collect the RDD values for output
    result = combined_rdd.collect()
    # Output results
    for record in result:
        print(record)


# ------------------------------------------
# Exercise 5
# ------------------------------------------
def ex5(rdd, station_id, months):
    """Consider only the months of Semester 1 (i.e., months 09, 10 and 11) and
    aggregate the measurements by month and day of the week to compute the
    average waiting time (i.e., expected_arrival_time - query_time). Sort the
    entries by decreasing average waiting time (240491 - Patrick Street – 
    Curraheen).

    :param rdd: processed input rdd -- pyspark rdd
    :param station_id: station number -- int
    :param months: months to filter on -- list
    """
    # Convert list of months to a set for more efficient hash lookups
    month_set = set(months)
    # Filter input RDD so only selected station and months are returned
    filter_rdd = rdd.filter(
        lambda x: True if (
                x[0] == station_id and x[4][3:5] in month_set) else False)
    # Map the filtered RDD so a new key with 'Day/Month' and time differences
    # are returned
    mapped_rdd = filter_rdd.map(month_day_wait_map)
    # Combine the mapped RDD values by key so the total time difference and a
    # count for each 'Day/Month' combination is returned
    #   Pt1 - Convert key values to a tuple of (value, counter)
    #   Pt2 - Increment existing accumulator with time difference value of
    #         new key/value and increment counter by 1
    #   Pt3 - Sum each accumulator and counter once all
    combined_rdd = mapped_rdd.combineByKey(
        lambda x: tuple([x, 1]),
        lambda accum, y: tuple([accum[0] + y, accum[1] + 1]),
        lambda fin_a, fin_b: tuple([fin_a[0] + fin_b[0],
                                    fin_a[1] + fin_b[1]]))
    # Map each value so the time difference is divided by the count to give the
    # average time difference per 'Day/Month'
    avg_rdd = combined_rdd.mapValues(lambda x: x[0] / x[1])
    # Sort the average values and collect results for output
    result = avg_rdd.sortBy(lambda x: x[1]).collect()
    # Output the results
    for record in result:
        print(record)


# ------------------------------------------
# Assistive Functions
# ------------------------------------------
def convert_time_to_epoch(str_time_in):
    """Convert a date time string to milliseconds since epoch.

    :param str_time_in: date time -- str
    :return: epoch time -- int
    """
    return int(
        time.mktime(time.strptime(str_time_in, '%d/%m/%Y %H:%M:%S')))


def month_day_wait_map(x):
    """Transform value data so a new key is returned of 'Day/Month' and the
    time difference between expected time and query time is the corresponding
    value.

    :param x: element from dataset -- RDD element
    :return: key, time difference -- tuple(str, int)
    """
    # Extract day and month (number) from the RDD element
    day, month = x[3], x[4][3:5]
    # Form the return key of 'Day/Month'
    new_key = '{d} {m}'.format(d=day, m=month)
    # Convert the query time to seconds since epoch
    query_time = convert_time_to_epoch('{d} {t}'.format(d=x[4], t=x[5]))
    # Conver the scheduled time to seconds since epoch
    exp_time = convert_time_to_epoch('{d} {t}'.format(d=x[4], t=x[7]))
    # Calculate the time difference between the expected time and the scheduled
    # time in seconds
    time_diff = exp_time - query_time
    # Return the new key and time difference
    return tuple([new_key, time_diff])


def process_line(line):
    res = ()
    line = line.replace('\n', '')
    params = line.split(';')
    if len(params) == 8:
        res = (int(params[0]), str(params[1]), str(params[2]), str(params[3]),
               str(params[4]), str(params[5]), str(params[6]), str(params[7]))
    return res


# ---------------------------------------------------------------
# Execute Program
# ---------------------------------------------------------------
def my_main(processed_rdd, ex_option):
    if ex_option == 1:
        ex1(processed_rdd)

    if ex_option == 2:
        ex2(processed_rdd, 240101)

    if ex_option == 3:
        ex3(processed_rdd, 240561)

    if ex_option == 4:
        ex4(processed_rdd, 241111)

    if ex_option == 5:
        ex5(processed_rdd, 240491, ['09', '10', '11'])


if __name__ == '__main__':
    # 1. Exercise option, set as 0 to run all exercises (default)
    option = 0

    # 2. Local Spark or Databricks
    is_local_spark = True

    # 3. We set the path to my_dataset and my_result
    LOCAL_PATH = '/home/michael/FileStore/CorkBusData'
    DATABRICKS_PATH = '/FileStore/tables/cork_bus_data'
    DATASET_DIR = '/my_dataset_single_file'

    if is_local_spark:
        FILE_STORE = LOCAL_PATH + DATASET_DIR
    else:
        FILE_STORE = DATABRICKS_PATH + DATASET_DIR

    # 4. We configure the Spark Context
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel('WARN')

    # 5. Load the dataset into a RDD
    raw_rdd = sc.textFile('{data}/*.csv'.format(data=FILE_STORE))

    # 6. Process each line to get the relevant info as a tuple of values
    input_rdd = raw_rdd.map(process_line)
    # Persist the RDD to memory for re-use
    input_rdd.cache()

    # 7. We call to our main function
    if option:
        print('\n\n{}\nExercise {}\n{}'.format(SEP, option, SEP))
        my_main(input_rdd, option)
    else:
        for i in range(1, 6):
            print('\n\n{}\nExercise {}\n{}'.format(SEP, i, SEP))
            my_main(input_rdd, i)
    print()
