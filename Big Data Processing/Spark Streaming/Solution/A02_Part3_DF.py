# -----------------------------------------------------
# Big Data Processing
# Assignment 2 - Spark Streaming & Structured Streaming
# Part 3 - Spark SQL
# Michael McAleer R00143621
# -----------------------------------------------------
import pyspark
import pyspark.sql.functions as sp_f
import time

from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import IntegerType, StringType, FloatType

SEP = '-------------------------------------------'


# ------------------------------------------
# FUNCTION ex1
# ------------------------------------------
def ex1(i_df):
    """Number of measurements (lines).

    :param i_df: input dataframe -- pyspark dataframe
    """
    # Count the total amount of entries in the df and print
    print(i_df.count())


# ------------------------------------------
# FUNCTION ex2
# ------------------------------------------
def ex2(i_df, station_number):
    """Amount of days in the calendar year (01/09/2016 - 31/08/2017) for which
    data is collected (240101 - UCC WGB – Lotabeg).

    :param i_df: input dataframe -- pyspark dataframe
    :param station_number: station number -- int
    """
    # Filter the input df to return entries matching station_id
    filter_df = i_df.filter(i_df['station_number'] == station_number)
    # Get the distinct dates
    distinct_df = filter_df.select('date').distinct()
    # Persist the df to memory for the count function
    distinct_df.cache()
    # Count the distinct dates and print
    print('Unique dates: {c}'.format(c=distinct_df.count()))


# ------------------------------------------
# FUNCTION ex3
# ------------------------------------------
def ex3(i_df, station_number):
    """For given station (240561 - UCC WGB – Curraheen) get the amount
    of measurements where:
        - scheduled_time >= expected_arrival_time (Ahead)
        - scheduled_time < expected_arrival_time (Behind)

    :param i_df: input dataframe -- pyspark dataframe
    :param station_number: station number -- int
    """
    # Filter the input df to return entries matching station_id
    filter_df = i_df.filter(i_df['station_number'] == station_number)
    # Persist the df to memory for the filter functions
    filter_df.persist()
    # Get the ahead count
    count_a = filter_df.filter(
        filter_df['scheduled_time'] >=
        filter_df['expected_arrival_time']).count()
    # Get the behind count
    count_b = filter_df.filter(
        filter_df['scheduled_time'] <
        filter_df['expected_arrival_time']).count()
    # Print the results of both counts
    print('Ahead: {a} Behind: {b}'.format(a=count_a, b=count_b))


# ------------------------------------------
# FUNCTION ex4
# ------------------------------------------
def ex4(i_df, station_number):
    """Get a list with the scheduled_time values found for each day of the
    week (241111 - CIT Tech. Park – Lotabeg).

    :param i_df: input dataframe -- pyspark dataframe
    :param station_number: station number -- int
    """
    # Filter the input df to return entries matching station_id
    filter_df = i_df.filter(i_df['station_number'] == station_number)
    # Persist the df to memory for the filter functions
    filter_df.persist()
    # Select only the columns we are interested in
    select_df = filter_df.select(filter_df["day_of_week"],
                                 filter_df["scheduled_time"])
    # Drop all duplicates by day_of_week/scheduled_time
    dropped_df = select_df.drop_duplicates(["day_of_week", "scheduled_time"])
    # Group the df by day_of_week key and aggregate scheduled_time  into a list
    grouped_df = dropped_df.groupBy("day_of_week").agg(
        sp_f.collect_list("scheduled_time"))
    # Rename the list of scheduled times list for output
    grouped_df = grouped_df.withColumnRenamed('collect_list(scheduled_time)',
                                              'bus_scheduled_times')
    # Collect the results and output
    result = grouped_df.collect()
    for record in result:
        print(record)


# ------------------------------------------
# FUNCTION ex5
# ------------------------------------------
def ex5(i_df, station_number, month_list):
    """Consider only the months of Semester 1 (i.e., months 09, 10 and 11) and
    aggregate the measurements by month and day of the week to compute the
    average waiting time (i.e., expected_arrival_time - query_time). Sort the
    entries by decreasing average waiting time (240491 - Patrick Street –
    Curraheen).

    :param i_df: input dataframe -- pyspark dataframe
    :param station_number: station number -- int
    :param month_list: months to filter on -- list
    """
    # Filter the input df to return entries matching station_id
    filter_df = i_df.filter(i_df['station_number'] == station_number)
    # Convert list of months to a set for more efficient hash lookups
    month_set = set(month_list)
    # Create user-defined function to format day and month into required key
    # format
    key_udf = sp_f.udf(
        lambda x, y: '{} {}'.format(x, y[3:5]) if y[3:5] in month_set else 'X')
    # Create a new column using the user-defined function from the previous
    # step
    key_df = filter_df.withColumn(
        'key', key_udf(filter_df['day_of_week'], filter_df['date']))
    # Filter the df so all invalid records are removed
    filter_df = key_df.filter(key_df['key'] != 'X')
    # Create user-defined function to caluclate the time difference in seconds
    # between the expected time and the scheduled time
    diff_udf = sp_f.udf(
        lambda x, y, z: (convert_time_to_epoch(x, y) -
                         convert_time_to_epoch(x, z)))
    # Create a new column using the user-defined function in the previous step
    # to get the time difference for each record in the df
    diff_df = filter_df.withColumn(
        'time_diff', diff_udf(filter_df['date'],
                              filter_df['expected_arrival_time'],
                              filter_df['query_time']))
    # Group the records by the new key and aggregate the time difference to get
    # the average
    avg_df = diff_df.groupBy('key').agg(sp_f.avg('time_diff'))
    # Order the results by average time difference
    solution = avg_df.orderBy(avg_df['avg(time_diff)'])
    # Output the results
    solution.show()


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


def process_input(spark_session, dataset_directory):
    """Load the dataset into a data frame with a defined schema.

    :param spark_session: spark session -- spark session object
    :param dataset_directory: path to data directory -- str
    :return: processed data frame -- pyspark data frame
    """
    my_schema = StructType(
        [StructField("station_number", IntegerType(), True),
         StructField("station_name", StringType(), True),
         StructField("direction", StringType(), True),
         StructField("day_of_week", StringType(), True),
         StructField("date", StringType(), True),
         StructField("query_time", StringType(), True),
         StructField("scheduled_time", StringType(), True),
         StructField("expected_arrival_time", StringType(), True)])

    return (spark_session.read.format("csv")
            .option("delimiter", ";")
            .option("quote", "")
            .option("header", "false")
            .schema(my_schema)
            .load(dataset_directory))


# ---------------------------------------------------------------
# PYTHON EXECUTION
# ---------------------------------------------------------------
def my_main(input_dataframe, ex_option):
    if ex_option == 1:
        ex1(input_dataframe)

    if ex_option == 2:
        ex2(input_dataframe, 240101)

    if ex_option == 3:
        ex3(input_dataframe, 240561)

    if ex_option == 4:
        ex4(input_dataframe, 241111)

    if ex_option == 5:
        ex5(input_dataframe, 240491, ['09', '10', '11'])


if __name__ == '__main__':
    # 1. We use as many input arguments as needed
    option = 1

    # 2. Local or Databricks
    is_local_spark = True

    # 3. We set the path to my_dataset and my_result
    LOCAL_PATH = '/home/michael/FileStore/CorkBusData'
    DATABRICKS_PATH = '/FileStore/tables/cork_bus_data'
    DATASET_DIR = '/my_dataset_single_file'

    if is_local_spark:
        FILE_STORE = LOCAL_PATH + DATASET_DIR
    else:
        FILE_STORE = DATABRICKS_PATH + DATASET_DIR

    # 4. We configure the Spark Session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    input_df = process_input(spark, FILE_STORE)

    # 5. We call to our main function
    print('\n\n{}\nExercise {}\n{}'.format(SEP, option, SEP))
    my_main(input_df, option)
    print()
