# -----------------------------------------------------
# Big Data Processing
# Assignment 2 - Spark Streaming & Structured Streaming
# Part 3 - Spark SQL Structured Streaming
# Michael McAleer R00143621
# -----------------------------------------------------
import pyspark
import pyspark.sql.functions as sp_f
import os
import shutil
import time

from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import IntegerType, StringType, FloatType

SEP = '-------------------------------------------'


# ------------------------------------------
# Exercise 1
# ------------------------------------------
def ex1(spark, monitoring_dir, interval):
    """Number of measurements (lines).

    :param spark: spark session -- pyspark Session obj
    :param monitoring_dir: path to monitoring directory -- str
    :param interval: timestep interval -- int
    :return: data stream writer - pyspark DataStreamWriter obj
    """
    # Set the frequency, window frquency, and sliding frequency
    freq, win, slide = get_frequencies(interval, 1, 1)
    # Create the DataFrame from the dataset and the schema
    input_sdf = (spark.readStream.format('csv')
                 .option('delimiter', ';')
                 .option('quote', '')
                 .option('header', 'false')
                 .schema(get_schema())
                 .load(monitoring_dir))

    # Add the current timestamp in a new column
    time_input_sdf = input_sdf.withColumn('my_time',
                                          sp_f.current_timestamp())
    # Group the sdf by the timestamp column and count occurrences
    count_sdf = (time_input_sdf.withWatermark('my_time', '0 seconds')
                 .groupBy(sp_f.window('my_time', win, slide),
                          sp_f.col('my_time')).count()
                 .drop('window').drop('my_time'))

    # Return the data stream writer (output to console sink)
    return (count_sdf.writeStream
            .format('console')
            .trigger(processingTime=freq)
            .outputMode('append'))


# ------------------------------------------
# Exercise 2
# ------------------------------------------
def ex2(spark, monitoring_dir, interval, station_number):
    """Amount of days in the calendar year (01/09/2016 - 31/08/2017) for which
    data is collected (240101 - UCC WGB – Lotabeg).

    :param spark: spark session -- pyspark Session obj
    :param monitoring_dir: path to monitoring directory -- str
    :param interval: timestep interval -- int
    :param station_number: station number -- int
    :return: data stream writer - pyspark DataStreamWriter obj
    """
    # Set the frequency, window frquency, and sliding frequency
    freq, win, slide = get_frequencies(interval, 1, 1)
    # Create the DataFrame from the dataset and the schema
    input_sdf = (spark.readStream.format('csv')
                 .option('delimiter', ';')
                 .option('quote', '')
                 .option('header', 'false')
                 .schema(get_schema())
                 .load(monitoring_dir))

    # Filter the input sdf to return entries matching station_id
    filter_sdf = input_sdf.filter(
        input_sdf['station_number'] == station_number)
    # Add the current timestamp in a new column
    time_input_sdf = filter_sdf.withColumn('my_time',
                                           sp_f.current_timestamp())
    # Drop the duplicate values by timestamp and date so only unique (distinct)
    # values remain
    duplicate_sdf = time_input_sdf.dropDuplicates(subset=['my_time', 'date'])
    # Group the sdf by the timestamp column and count occurrences, this gives
    # our count of the unique dates
    date_sdf = (duplicate_sdf.withWatermark('my_time', '0 seconds')
                .groupBy(sp_f.window('my_time', win, slide))
                .count().drop('window'))

    # Return the data stream writer (output to console sink)
    return (date_sdf.writeStream
            .format('console')
            .trigger(processingTime=freq)
            .outputMode('append'))


# ------------------------------------------
# Exercise 3
# ------------------------------------------
def ex3(spark, monitoring_dir, interval, station_number):
    """For given station (240561 - UCC WGB – Curraheen) get the amount
    of measurements where:
        - scheduled_time >= expected_arrival_time (Ahead)
        - scheduled_time < expected_arrival_time (Behind)

    :param spark: spark session -- pyspark Session obj
    :param monitoring_dir: path to monitoring directory -- str
    :param interval: timestep interval -- int
    :param station_number: station number -- int
    :return: data stream writer - pyspark DataStreamWriter obj
    """
    # Set the frequency, window frquency, and sliding frequency
    freq, win, slide = get_frequencies(interval, 1, 1)
    # Create the DataFrame from the dataset and the schema
    input_sdf = (spark.readStream.format('csv')
                 .option('delimiter', ';')
                 .option('quote', '')
                 .option('header', 'false')
                 .schema(get_schema())
                 .load(monitoring_dir))

    # Filter the input sdf to return entries matching station_id
    filter_sdf = input_sdf.filter(
        input_sdf['station_number'] == station_number)
    # Add the current timestamp in a new column
    wm_sdf = filter_sdf.withColumn('my_time', sp_f.current_timestamp())
    # Define UDF to determine if a bus record in the SDF is ahead or behind
    time_udf = sp_f.udf(
        lambda x, y: 'Ahead' if x >= y else 'Behind')
    # Apply the UDF on the SDF and output the status in a new column
    key_sdf = wm_sdf.withColumn(
        'status', time_udf(
            filter_sdf['scheduled_time'], input_sdf['expected_arrival_time']))
    # Group the sdf by the timestamp column and count occurrences of ahead and
    # behind
    date_sdf = (key_sdf.withWatermark('my_time', '0 seconds')
                .groupBy(sp_f.window('my_time', win, slide),
                         sp_f.col('status'))
                .count().drop('window'))

    # Return the data stream writer (output to console sink)
    return (date_sdf.writeStream
            .format('console')
            .trigger(processingTime=freq)
            .outputMode('append'))


# ------------------------------------------
# Exercise 4
# ------------------------------------------
def ex4(spark, monitoring_dir, interval, station_number):
    """Get a list with the scheduled_time values found for each day of the
    week (241111 - CIT Tech. Park – Lotabeg).

    :param spark: spark session -- pyspark Session obj
    :param monitoring_dir: path to monitoring directory -- str
    :param interval: timestep interval -- int
    :param station_number: station number -- int
    :return: data stream writer - pyspark DataStreamWriter obj
    """
    # Set the frequency, window frquency, and sliding frequency
    freq, win, slide = get_frequencies(interval, 1, 1)
    # Create the DataFrame from the dataset and the schema
    input_sdf = (spark.readStream.format('csv')
                 .option('delimiter', ';')
                 .option('quote', '')
                 .option('header', 'false')
                 .schema(get_schema())
                 .load(monitoring_dir))

    # Filter the input sdf to return entries matching station_id
    filter_sdf = input_sdf.filter(
        input_sdf['station_number'] == station_number)
    # Add the current timestamp in a new column
    wm_sdf = filter_sdf.withColumn('my_time', sp_f.current_timestamp())
    # Select only the relevant rows
    select_sdf = wm_sdf.select(wm_sdf['my_time'],
                               wm_sdf['day_of_week'],
                               wm_sdf['scheduled_time'])
    # Drop all duplicates so only a list of unique times for each day remains
    dropped_sdf = select_sdf.drop_duplicates([
        'my_time', 'day_of_week', 'scheduled_time'])
    # Group the sdf by day_of_week and aggregate the scheduled times into a
    # list
    agg_sdf = (dropped_sdf.withWatermark('my_time', '0 seconds')
               .groupBy(sp_f.window('my_time', win, slide),
                        sp_f.col('day_of_week'))
               .agg(sp_f.collect_list('scheduled_time'))
               .drop('window'))
    # Rename the new aggregate list column
    grouped_df = agg_sdf.withColumnRenamed('collect_list(scheduled_time)',
                                           'bus_scheduled_times')

    # Return the data stream writer (output to console sink, dont truncate
    # column values so whole list is displayed)
    return (grouped_df.writeStream
            .format('console')
            .option('truncate', False)
            .trigger(processingTime=freq)
            .outputMode('append'))


# ------------------------------------------
# Exercise 5
# ------------------------------------------
def ex5(spark, monitoring_dir, interval, station_number, month_list):
    """Consider only the months of Semester 1 (i.e., months 09, 10 and 11) and
    aggregate the measurements by month and day of the week to compute the
    average waiting time (i.e., expected_arrival_time - query_time) NO SORT.
    (240491 - Patrick Street – Curraheen).

    :param spark: spark session -- pyspark Session obj
    :param monitoring_dir: path to monitoring directory -- str
    :param interval: timestep interval -- int
    :param station_number: station number -- int
    :param month_list: months to filter on -- list
    :return: data stream writer - pyspark DataStreamWriter obj
    """
    # Set the frequency, window frquency, and sliding frequency
    freq, win, slide = get_frequencies(interval, 1, 1)
    # Create the DataFrame from the dataset and the schema
    input_sdf = (spark.readStream.format('csv')
                 .option('delimiter', ';')
                 .option('quote', '')
                 .option('header', 'false')
                 .schema(get_schema())
                 .load(monitoring_dir))

    # Filter the input sdf to return entries matching station_id
    filter_sdf = input_sdf.filter(
        input_sdf['station_number'] == station_number)
    # Add the current timestamp in a new column
    wm_sdf = filter_sdf.withColumn('my_time', sp_f.current_timestamp())
    # Convert list of months to a set for more efficient hash lookups
    month_set = set(month_list)
    # Create user-defined function to format day and month into required key
    # format
    key_udf = sp_f.udf(
        lambda x, y: '{} {}'.format(x, y[3:5]) if y[3:5] in month_set else 'X')
    # Create a new column using the user-defined function from the previous
    # step
    key_sdf = wm_sdf.withColumn(
        'key', key_udf(wm_sdf['day_of_week'], wm_sdf['date']))
    # Filter the sdf so all invalid records are removed
    filter_sdf = key_sdf.filter(key_sdf['key'] != 'X')
    # Create user-defined function to caluclate the time difference in seconds
    # between the expected time and the scheduled time
    diff_udf = sp_f.udf(
        lambda x, y, z: (convert_time_to_epoch(x, y) -
                         convert_time_to_epoch(x, z)))
    # Create a new column using the user-defined function in the previous step
    # to get the time difference for each record in the df
    diff_sdf = filter_sdf.withColumn(
        'time_diff', diff_udf(filter_sdf['date'],
                              filter_sdf['expected_arrival_time'],
                              filter_sdf['query_time']))
    # Group the sdf by the newly generate key from a previous step and
    # aggregate the time_diff column to give the average time difference
    agg_sdf = (diff_sdf.withWatermark('my_time', '0 seconds')
               .groupBy(sp_f.window('my_time', win, slide),
                        sp_f.col('key'))
               .agg(sp_f.avg('time_diff'))
               .drop('window'))

    #  Return the data stream writer (output to console sink)
    return (agg_sdf.writeStream
            .format('console')
            .option('truncate', False)
            .option('numRows', 30)
            .trigger(processingTime=freq)
            .outputMode('append'))


# ------------------------------------------
# Assistive Functions
# ------------------------------------------
def convert_time_to_epoch(date_in, time_in):
    """Convert a date time string to milliseconds since epoch.

    :param date_in: date -- str
    :param time_in: time -- str
    :return: epoch time -- int
    """
    date_time = '{} {}'.format(date_in, time_in)
    return int(
        time.mktime(time.strptime(date_time, '%d/%m/%Y %H:%M:%S')))


def get_schema():
    """Create SDF schema.

    :return: SDF schema - PySpark SDF Schema
    """
    return StructType(
        [StructField('station_number', IntegerType(), True),
         StructField('station_name', StringType(), True),
         StructField('direction', StringType(), True),
         StructField('day_of_week', StringType(), True),
         StructField('date', StringType(), True),
         StructField('query_time', StringType(), True),
         StructField('scheduled_time', StringType(), True),
         StructField('expected_arrival_time', StringType(), True)])


def get_frequencies(interval, window, slide):
    """Set the frequency, window frquency, and sliding frequency values
    required for the data stream writer and aggregation functions.

    :param interval: the timestep interval -- int
    :param window: the window size -- int
    :param slide: the sliding size -- int
    :return: frequency, window duration frequency, sliding duration frequency
             -- int, int, int
    """
    wdf = str(window * interval) + ' seconds'
    sdf = str(slide * interval) + ' seconds'
    freq = str(interval) + ' seconds'

    return freq, wdf, sdf


# ---------------------------------------------------------------
# Execute Program
# ---------------------------------------------------------------
def get_source_dir_file_names(is_local, source_dir, verbose_mode):
    res = []
    if is_local:
        file_info_objects = os.listdir(source_dir)
    else:
        file_info_objects = dbutils.fs.ls(source_dir)

    for item in file_info_objects:
        file_name = str(item)
        if not is_local:
            lb_index = file_name.index("name='")
            file_name = file_name[(lb_index + 6):]
            ub_index = file_name.index("',")
            file_name = file_name[:ub_index]

        res.append(file_name)
        if verbose_mode:
            print(file_name)

    res.sort()
    return res


def streaming_simulation(is_local, source_dir, monitoring_dir,
                         interval, verbose_mode):
    files = get_source_dir_file_names(is_local, source_dir, verbose)
    time.sleep(interval * 0.1)
    start = time.time()

    if verbose_mode:
        print('Start time = ' + str(start))

    count = 0
    for file in files:
        if is_local:
            shutil.copyfile(source_dir + file, monitoring_dir + file)
        else:
            dbutils.fs.cp(source_dir + file, monitoring_dir + file)

        count = count + 1
        if verbose_mode:
            print('File ' + str(count) + ' transferred. Time since start = '
                  + str(time.time() - start))

        time.sleep((start + (count * interval)) - time.time())

    time.sleep(interval)


def my_main(spark, is_local, source_dir, monitoring_dir,
            interval, verbose_mode, ex_option):
    dsw = None

    if ex_option == 1:
        dsw = ex1(spark, monitoring_dir, interval)

    if ex_option == 2:
        dsw = ex2(spark, monitoring_dir, interval, 240101)

    if ex_option == 3:
        dsw = ex3(spark, monitoring_dir, interval, 240561)

    if ex_option == 4:
        dsw = ex4(spark, monitoring_dir, interval, 241111)

    if ex_option == 5:
        dsw = ex5(spark, monitoring_dir, interval, 240491, ['09', '10', '11'])

    ssq = dsw.start()
    ssq.awaitTermination(interval)
    streaming_simulation(is_local, source_dir, monitoring_dir,
                         interval, verbose_mode)
    ssq.stop()


# ---------------------------------------------------------------
#           PYTHON EXECUTION
# ---------------------------------------------------------------
if __name__ == '__main__':
    # 1. We use as many input arguments as needed
    option = 1

    # 2. Local or Databricks
    is_local_spark = True

    # 3. We set the path to my_dataset and my_result
    LOCAL_PATH = '/home/michael/FileStore/CorkBusData'
    DATABRICKS_PATH = '/FileStore/tables/cork_bus_data'
    SOURCE_DIR = '/my_dataset_complete/'
    MONITORING_DIR = '/my_monitoring/'
    CHECKPOINT_DIR = '/my_checkpoint/'
    RESULT_DIR = '/my_result/'

    if is_local_spark:
        source_directory = LOCAL_PATH + SOURCE_DIR
        monitoring_directory = LOCAL_PATH + MONITORING_DIR
        checkpoint_directory = LOCAL_PATH + CHECKPOINT_DIR
        result_directory = LOCAL_PATH + RESULT_DIR
    else:
        source_directory = DATABRICKS_PATH + SOURCE_DIR
        monitoring_directory = DATABRICKS_PATH + MONITORING_DIR
        checkpoint_directory = DATABRICKS_PATH + CHECKPOINT_DIR
        result_directory = DATABRICKS_PATH + RESULT_DIR

    # 4. We set the Spark Streaming parameters

    # 4.1. We specify the time interval each of our micro-batches (files)
    # appear for its processing.
    time_step_interval = 15

    # 4.2. We configure verbosity during the program run
    verbose = False

    # 5. We remove the directories
    if is_local_spark:
        # 5.1. We remove the monitoring_directory
        if os.path.exists(monitoring_directory):
            shutil.rmtree(monitoring_directory)

        # 5.2. We remove the result_directory
        if os.path.exists(result_directory):
            shutil.rmtree(result_directory)

        # 5.3. We remove the checkpoint_directory
        if os.path.exists(checkpoint_directory):
            shutil.rmtree(checkpoint_directory)
    else:
        # 5.1. We remove the monitoring_directory
        dbutils.fs.rm(monitoring_directory, True)

        # 5.2. We remove the result_directory
        dbutils.fs.rm(result_directory, True)

        # 5.3. We remove the checkpoint_directory
        dbutils.fs.rm(checkpoint_directory, True)

    # 6. We re-create the directories again
    if is_local_spark:
        # 6.1. We re-create the monitoring_directory
        os.mkdir(monitoring_directory)

        # 6.2. We re-create the result_directory
        os.mkdir(result_directory)

        # 6.3. We re-create the checkpoint_directory
        os.mkdir(checkpoint_directory)
    else:
        # 6.1. We re-create the monitoring_directory
        dbutils.fs.mkdirs(monitoring_directory)

        # 6.2. We re-create the result_directory
        dbutils.fs.mkdirs(result_directory)

        # 6.3. We re-create the checkpoint_directory
        dbutils.fs.mkdirs(checkpoint_directory)

    # 7. We configure the Spark Session
    spark_session = pyspark.sql.SparkSession.builder.getOrCreate()
    spark_session.sparkContext.setLogLevel('WARN')

    print('\n\n{}\nExercise {}\n{}'.format(SEP, option, SEP))
    # 8. We call to our main function
    my_main(spark_session, is_local_spark, source_directory,
            monitoring_directory, time_step_interval, verbose, option)
