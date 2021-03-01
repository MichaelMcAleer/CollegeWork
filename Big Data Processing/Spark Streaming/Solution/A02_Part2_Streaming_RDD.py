# -----------------------------------------------------
# Big Data Processing
# Assignment 2 - Spark Streaming & Structured Streaming
# Part 2 - Spark Streaming
# Michael McAleer R00143621
# -----------------------------------------------------
import pyspark
import pyspark.streaming
import os
import shutil
import time

from datetime import datetime

SEP = '-------------------------------------------'


# ------------------------------------------
# Exercise 1
# ------------------------------------------
def ex1(ssc, monitoring_directory):
    """Number of measurements (lines).

    :param ssc: spark streaming context -- spark context
    :param monitoring_directory: path to monitoring directory -- str
    """
    # Set up text file stream monitoring
    input_dstream = ssc.textFileStream(monitoring_directory)
    # Count the amount of values in the input datastream
    count_dstream = input_dstream.count()
    # Persist the results to memory
    count_dstream.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    # Output the results
    count_dstream.pprint()


# ------------------------------------------
# Exercise 2
# ------------------------------------------
def ex2(ssc, monitoring_directory, station_id):
    """Amount of days in the calendar year (01/09/2016 - 31/08/2017) for which
    data is collected (240101 - UCC WGB – Lotabeg).

    :param ssc: spark streaming context -- spark context
    :param monitoring_directory: path to monitoring directory -- str
    :param station_id: station number -- int
    """
    # Set up text file stream monitoring
    input_dstream = ssc.textFileStream(monitoring_directory)
    # Process the input data to convert each line to a tuple of variables
    processed_dstream = input_dstream.map(process_line)

    # Filter the dstream to return entries matching station_id
    filter_dstream = processed_dstream.filter(
        lambda x: True if x[0] == station_id else False)
    # Map the dstream values so date values are isolated
    mapped_dstream = filter_dstream.map(lambda x: x[4])
    # Transform the mapped datastream to return only distinct values
    distinct_dstream = mapped_dstream.transform(lambda rdd: rdd.distinct())
    # Count the amount of distinct values
    solution_dstream = distinct_dstream.count()

    # Persist the results to memory
    solution_dstream.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    # Output the results
    solution_dstream.pprint()


# ------------------------------------------
# Exercise 3
# ------------------------------------------
def ex3(ssc, monitoring_directory, station_id):
    """For given station (240561 - UCC WGB – Curraheen) get the amount
    of measurements where:
        - scheduled_time >= expected_arrival_time (Behind)
        - scheduled_time < expected_arrival_time (Ahead)

    :param ssc: spark streaming context -- spark context
    :param monitoring_directory: path to monitoring directory -- str
    :param station_id: station number -- int
    """
    # Set up text file stream monitoring
    input_dstream = ssc.textFileStream(monitoring_directory)
    # Process the input data to convert each line to a tuple of variables
    processed_dstream = input_dstream.map(process_line)

    # Filter the dstream to return entries matching station_id
    filter_rdd = processed_dstream.filter(
        lambda x: True if x[0] == station_id else False)
    # Map the dstream values to indicate if bus queried is ahead or behind
    # its scheduled time
    mapped_rdd = filter_rdd.map(
        lambda x: tuple(
            ['Ahead', 1]) if x[6] >= x[7] else tuple(['Behind', 1]))
    # Reduce the dstream by key so instances of each key are counted,
    # giving the amount of measurements for both behind and ahead keys
    count_rdd = mapped_rdd.reduceByKey(lambda x, y: x + y)

    # Persist the results to memory
    count_rdd.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    # Output the results
    count_rdd.pprint()


# ------------------------------------------
# Exercise 4
# ------------------------------------------
def ex4(ssc, monitoring_directory, station_id):
    """Get a list with the scheduled_time values found for each day of the
    week (241111 - CIT Tech. Park – Lotabeg).

    :param ssc: spark streaming context -- spark context
    :param monitoring_directory: path to monitoring directory -- str
    :param station_id: station number -- int
    """
    # Set up text file stream monitoring
    input_dstream = ssc.textFileStream(monitoring_directory)
    # Process the input data to convert each line to a tuple of variables
    processed_dstream = input_dstream.map(process_line)

    # Filter the dstream to return entries matching station_id
    filter_dstream = processed_dstream.filter(
        lambda x: True if x[0] == station_id else False)
    # Map the dstream values so day_of_week and scheduled_time are isolated
    mapped_dstream = filter_dstream.map(lambda x: tuple([x[3], x[6]]))
    # Combine the mapped dstream values by key so a list of times for each
    # day is returned after the final step in the process
    #   Pt1 - Convert key value to a list
    #   Pt2 - Merge any new values to the existing list for a key if the key
    #         exists
    #   Pt3 - Merge lists as-is in final step
    combined_dstream = mapped_dstream.combineByKey(
        lambda x: [x],
        lambda accum, y: accum + [y],
        lambda f_accum_a, f_accum_b: f_accum_a + f_accum_b)
    # Map the combined key values so lists of times are turned into sets of
    # unique times only
    solution_dstream = combined_dstream.mapValues(lambda x: set(x))

    # Persist the results to memory
    solution_dstream.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    # Output the results
    solution_dstream.pprint()


# ------------------------------------------
# Exercise 5
# ------------------------------------------
def ex5(ssc, monitoring_directory, station_id, months):
    """Consider only the months of Semester 1 (i.e., months 09, 10 and 11) and
    aggregate the measurements by month and day of the week to compute the
    average waiting time (i.e., expected_arrival_time - query_time). Sort the
    entries by decreasing average waiting time (240491 - Patrick Street –
    Curraheen).

    :param ssc: spark streaming context -- spark context
    :param monitoring_directory: path to monitoring directory -- str
    :param station_id: station number -- int
    :param months: months to filter on -- list
    """
    # Set up text file stream monitoring
    input_dstream = ssc.textFileStream(monitoring_directory)
    # Process the input data to convert each line to a tuple of variables
    processed_dstream = input_dstream.map(process_line)
    # Convert list of months to a set for more efficient hash lookups
    month_set = set(months)
    # Filter input dstream so only selected station and months are returned
    filter_dstream = processed_dstream.filter(
        lambda x: True if (
                x[0] == station_id and x[4][3:5] in month_set) else False)
    # Map the filtered dstream so a new key with 'Day/Month' and time
    # differences are returned
    mapped_dstream = filter_dstream.map(month_day_wait_map)
    # Combine the mapped dstream values by key so the total time difference
    # and a count for each 'Day/Month' combination is returned
    #   Pt1 - Convert key values to a tuple of (value, counter)
    #   Pt2 - Increment existing accumulator with time difference value of
    #         new key/value and increment counter by 1
    #   Pt3 - Sum each accumulator and counter once all
    combined_dstream = mapped_dstream.combineByKey(
        lambda x: tuple([x, 1]),
        lambda accum, y: tuple([accum[0] + y, accum[1] + 1]),
        lambda fin_a, fin_b: tuple([fin_a[0] + fin_b[0],
                                    fin_a[1] + fin_b[1]]))
    # Map each value so the time difference is divided by the count to give the
    # average time difference per 'Day/Month'
    avg_dstream = combined_dstream.mapValues(lambda x: x[0] / x[1])
    # Sort the average values and collect results for output
    sorted_dstream = avg_dstream.transform(
        lambda rdd: rdd.sortBy(lambda x: x[1]))

    # Persist the results to memory
    sorted_dstream.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    # Output the results
    sorted_dstream.foreachRDD(collect_and_print)


# ------------------------------------------
# Assistive Functions
# ------------------------------------------
def collect_and_print(rdd):
    """Collect and print RDD values.

    :param rdd: processed rdd -- pyspark rdd
    """
    # Collect the RDD values for output
    result = rdd.collect()
    # Get the current time
    now = datetime.now()
    str_now = now.strftime("%m/%d/%Y, %H:%M:%S")
    # Output results
    print('{s}\nTime: {t}\n{s}'.format(s=SEP, t=str_now))
    for record in result:
        print(record)
    print('')


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


# ------------------------------------------
# Context & Streaming Simulation
# ------------------------------------------
def get_source_dir_file_names(is_local, source_directory, verbose_output):
    res = list()
    if is_local:
        file_info_objects = os.listdir(source_directory)
    else:
        file_info_objects = dbutils.fs.ls(source_directory)
    for item in file_info_objects:
        file_name = str(item)
        if not is_local:
            lb_index = file_name.index("name='")
            file_name = file_name[(lb_index + 6):]
            ub_index = file_name.index("',")
            file_name = file_name[:ub_index]
        res.append(file_name)
        if verbose_output:
            print(file_name)
    res.sort()
    return res


def streaming_simulation(
        is_local, source_directory, monitoring_directory, time_step,
        verbose_output, dataset_file_names):
    start, count = time.time(), 0
    if verbose_output:
        print("Start time = " + str(start))
    for file in dataset_file_names:
        if is_local:
            shutil.copyfile(source_directory + file,
                            monitoring_directory + file)
        else:
            dbutils.fs.cp(source_directory + file,
                          monitoring_directory + file)
        if verbose_output:
            print("File " + str(
                count) + " transferred. Time since start = " + str(
                time.time() - start))
        count += 1
        time_to_wait = (start + (count * time_step)) - time.time()
        if time_to_wait > 0:
            time.sleep(time_to_wait)


def create_ssc(spark_context, monitoring_directory, time_step, ex_option):
    ssc = pyspark.streaming.StreamingContext(spark_context, time_step)
    my_model(ssc, monitoring_directory, ex_option)
    return ssc


def my_main(spark_context, is_local, source_directory, monitoring_directory,
            checkpoint_directory, time_step, verbose_output, ex_option):
    dataset_file_names = get_source_dir_file_names(
        is_local, source_directory, verbose_output)
    ssc = pyspark.streaming.StreamingContext.getActiveOrCreate(
        checkpoint_directory,
        lambda: create_ssc(spark_context, monitoring_directory,
                           time_step, ex_option))
    ssc.start()
    ssc.awaitTerminationOrTimeout(time_step)
    streaming_simulation(
        is_local, source_directory, monitoring_directory, time_step,
        verbose_output, dataset_file_names)
    ssc.stop(is_local)
    if not spark_context._jvm.StreamingContext.getActive().isEmpty():
        spark_context._jvm.StreamingContext.getActive().get().stop(is_local)


def my_model(ssc, monitoring_directory, ex_option):
    if ex_option == 1:
        ex1(ssc, monitoring_directory)

    if ex_option == 2:
        ex2(ssc, monitoring_directory, 240101)

    if ex_option == 3:
        ex3(ssc, monitoring_directory, 240561)

    if ex_option == 4:
        ex4(ssc, monitoring_directory, 241111)

    if ex_option == 5:
        ex5(ssc, monitoring_directory, 240491, ['09', '10', '11'])


# ---------------------------------------------------------------
# Execute Program
# ---------------------------------------------------------------
if __name__ == '__main__':
    # 1. We use as many input arguments as needed
    # 1.1. We specify the time interval each of our micro-batches (files)
    # appear for its processing.
    time_step_interval = 10
    # 1.2. We configure verbosity during the program run
    verbose = False
    # 1.3. We specify the exercise we want to solve
    option = 1

    # 2. Local or Databricks
    is_local_spark = True

    # 3. We set the path to my_dataset, my_monitoring, my_checkpoint
    # and my_result
    LOCAL_PATH = '/home/michael/FileStore/CorkBusData'
    DATABRICKS_PATH = '/FileStore/tables/cork_bus_data'
    SOURCE_DIR = "/my_dataset_complete/"
    MONITORING_DIR = "/my_monitoring/"
    CHECKPOINT_DIR = "/my_checkpoint/"

    if is_local_spark:
        source_dir = LOCAL_PATH + SOURCE_DIR
        monitoring_dir = LOCAL_PATH + MONITORING_DIR
        checkpoint_dir = LOCAL_PATH + CHECKPOINT_DIR
    else:
        source_dir = DATABRICKS_PATH + SOURCE_DIR
        monitoring_dir = DATABRICKS_PATH + MONITORING_DIR
        checkpoint_dir = DATABRICKS_PATH + CHECKPOINT_DIR

    # 4. We remove the directories
    if is_local_spark:
        # 4.1. We remove the monitoring_dir
        if os.path.exists(monitoring_dir):
            shutil.rmtree(monitoring_dir)

        # 4.2. We remove the checkpoint_dir
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
    else:
        # 4.1. We remove the monitoring_dir
        dbutils.fs.rm(monitoring_dir, True)
        # 4.2. We remove the checkpoint_dir
        dbutils.fs.rm(checkpoint_dir, True)

    # 5. We re-create the directories again
    if is_local_spark:
        # 5.1. We re-create the monitoring_dir
        os.mkdir(monitoring_dir)
        # 5.2. We re-create the checkpoint_dir
        os.mkdir(checkpoint_dir)
    else:
        # 5.1. We re-create the monitoring_dir
        dbutils.fs.mkdirs(monitoring_dir)
        # 5.2. We re-create the checkpoint_dir
        dbutils.fs.mkdirs(checkpoint_dir)

    # 6. We configure the Spark Context
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel('WARN')

    # 7. We call to our main function
    print('\n\n{}\nExercise {}\n{}'.format(SEP, option, SEP))
    my_main(sc, is_local_spark, source_dir, monitoring_dir,
            checkpoint_dir, time_step_interval, verbose, option)
    print()
