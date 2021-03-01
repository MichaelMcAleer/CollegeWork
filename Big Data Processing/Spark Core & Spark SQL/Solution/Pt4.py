# ------------------------------------------
# Big Data Processing
# Assignment 1 - Spark Core & Spark SQL
# Part 4 - Spark SQL
# Michael McAleer R00143621
# ------------------------------------------
import pyspark
import time

import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import IntegerType, StringType, FloatType


def load_dataset_get_data_frame(sp, d_dir):
    """Load the dataset into a data frame with a defined schema.

    :param sp: spark session -- spark session object
    :param d_dir: path to data directory -- str
    :return: processed data frame -- pyspark data frame
    """
    # Define data frame schema
    my_schema = StructType(
        [StructField('status', IntegerType(), True),
         StructField('name', StringType(), True),
         StructField('longitude', FloatType(), True),
         StructField('latitude', FloatType(), True),
         StructField('date_status', StringType(), True),
         StructField('bikes_available', IntegerType(), True),
         StructField('docks_available', IntegerType(), True)])

    # Create the data frame from the source data directory and set schema
    return (sp.read.format('csv').option('delimiter', ';')
            .option('quote', '').option('header', 'false')
            .schema(my_schema).load(d_dir))


###############################################################################

def ex1(df):
    """Exercise 1: Number of times each station ran out of bikes (sorted
    decreasingly by station).

    :param df: processed input -- pyspark data frame
    """
    # Filter input data frame to get only rows where bikes available are 0
    filter_df = df.filter((df['bikes_available'] == 0))
    # Extract the data frame columns we are interested in
    select_df = filter_df.select(df['name'], df['bikes_available'])
    # Get the count of each element in 'name'
    count_df = select_df.groupBy(select_df['name']).count()
    # Sort the results in descending order
    sorted_df = count_df.orderBy(count_df['count'].desc())
    # Collect the results and output
    result = sorted_df.collect()
    for item in result:
        print(item)


###############################################################################

def ex2(df):
    """Exercise 2: Pick one busy day with plenty of ran outs -> 27/08/2017
    Average amount of bikes per station and hour window.

    :param df: processed input -- pyspark data frame
    """
    # Filter input data frame to get only rows where date is 27-08-2017
    filter_df = df.filter(df['date_status'].contains('27-08-2017'))
    # Create user defined function to convert date_status to hour value
    hour_udf = f.udf(
        lambda x: '{hour}:00'.format(hour=x.split(' ')[1][:2]), StringType())
    # Apply udf to get new column of hour values
    result_df = filter_df.withColumn(
        'hour', hour_udf(filter_df['date_status']))
    # Extract the data frame columns we are interested in
    select_df = result_df.select(result_df['name'], result_df['hour'],
                                 result_df['bikes_available'])
    # Group the rows by name and hour value getting the average bikes
    # available for each
    avg_df = select_df.groupBy(select_df['name'], select_df['hour']).agg(
        {'bikes_available': 'avg'})
    # Order the results by name and hour
    sorted_df = avg_df.orderBy(avg_df['name'], avg_df['hour'])
    # Collect the results and output
    result = sorted_df.collect()
    for item in result:
        print(item)


###############################################################################


def ex3(df):
    """Exercise 3: Pick one busy day with plenty of ran outs -> 27/08/2017
    Get the different ran-outs to attend.

    Note: n consecutive measurements of a station being ran-out of bikes has to
    be considered a single ran-out, that should have been attended when the
    ran-out happened in the first time.

    :param df: processed input -- pyspark data frame
    """
    # Filter input data frame to get only rows where date is 27-08-2017 and
    # bikes available is 0
    filter_df = df.filter((df['date_status'].contains('27-08-2017')) &
                          (df['bikes_available'] == 0))
    # Create udf to convert date_status to seconds since epoch
    epoch_udf = f.udf(
        lambda x: int(time.mktime(time.strptime(x, '%d-%m-%Y %H:%M:%S'))),
        IntegerType())
    # Apply udf to get new column with epoch time for each row
    epoch_df = filter_df.withColumn(
        'epoch_time', epoch_udf(filter_df['date_status']))
    # Extract the data frame columns we are interested in
    select_df = epoch_df.select(epoch_df['name'], epoch_df['epoch_time'])
    # Iterate over the rows getting the values of the previous row's for name
    # and epoch time, this will be used to determine difference between station
    # intervals
    clone_df = select_df.select(
        '*',
        *([f.lag(f.col(c), default=0).over(
            Window.orderBy(
                select_df['name'], select_df['epoch_time'])).alias(
            'p_' + c) for c in select_df.columns]))
    # Create udf to take the station name and bike count of the current and
    # previous rows, if the stations name match and the time difference is less
    # than 360 seconds then don't count the row as a new ran-out, else the time
    # difference is larger or it is a new station and is to be counted as a ran
    # out occurrence
    ran_out_udf = f.udf(
        lambda a, b, c, d: 0 if ((a in c) and (b - d <= 360)) else 1,
        IntegerType())
    # Apply udf to row to get new column which tracks new ran-outs
    ran_out_df = clone_df.withColumn(
        'ran_out_new', ran_out_udf(
            clone_df['name'], clone_df['epoch_time'],
            clone_df['p_name'], clone_df['p_epoch_time']))
    # Filter the data frame so only those rows which meet the new ran-out
    # criteria are returned
    ran_out_filtered = ran_out_df.filter(ran_out_df['ran_out_new'] != 0)
    # Create udf to convert epoch time back to hour:minute
    time_udf = f.udf(
        lambda x: time.strftime('%H:%M', time.gmtime(x)), StringType())
    # Apply the udf to get readable time in new column
    time_udf = ran_out_filtered.withColumn(
        'time', time_udf(ran_out_filtered['epoch_time']))
    # Select only the rows we are interested in for output
    result_df = time_udf.select(
        time_udf['name'], time_udf['time']).orderBy(time_udf['time'])
    # Collect the results and output
    result = result_df.collect()
    for item in result:
        print(item)


###############################################################################

def ex4(df, ran_outs):
    """Exercise 4: Pick one busy day with plenty of ran outs -> 27/08/2017
    Get the station with biggest number of bikes for each ran-out to be
    attended.

    :param df: processed input -- pyspark data frame
    :param ran_outs: ran out times -- set
    """
    # Create a data frame with one column using the ran-out list
    ran_outs_df = spark.createDataFrame(ran_outs, StringType())
    ran_outs_df = ran_outs_df.withColumnRenamed('value', 'ran_out_time')
    # Filter input data frame to get only rows where date is 27-08-2017 and
    # bikes available is greater than 0
    filter_df = df.filter(
        (df['date_status'].contains('27-08-2017')) &
        (df['bikes_available'] > 0))
    # Create udf to convert date_status to hour:minute:second
    hour_udf = f.udf(
        lambda x: x.split(' ')[1], StringType())
    # Apply udf to get new column with hour:minute:time for each row
    result_df = filter_df.withColumn(
        'hour', hour_udf(filter_df['date_status']))
    # Extract the data frame columns we are interested in
    select_df = result_df.select(result_df['hour'], result_df['name'],
                                 result_df['bikes_available'])
    # Join the bike data frame with the ran out times data frame so we are left
    # only with those rows that match a ran out time
    join_df = select_df.join(ran_outs_df,
                             select_df['hour'] == ran_outs_df['ran_out_time'],
                             'left_semi')
    # Get the maximum bikes available for each hour value
    max_df = join_df.groupBy(join_df['hour']).agg({'bikes_available': 'max'})
    # Join the data frame containing only ran out time matching rows with the
    # data frame that contains the maximum values for each ran out time, the
    # result is a data frame where the two tables' hour values match and the
    # bikes available and max bikes available values match
    result_df = join_df.join(
        max_df,
        [join_df['hour'] == max_df['hour'],
         join_df['bikes_available'] == max_df['max(bikes_available)']],
        'left_semi')
    # Order the results by hour
    ordered_df = result_df.orderBy(result_df['hour'])
    # Collect the results and output
    result = ordered_df.collect()
    for item in result:
        print(item)


###############################################################################

def ex5(df):
    """Exercise 5: Total number of bikes that are taken and given back per
    station (sort the results in decreasing order in the sum of bikes taken +
    bikes given back).

    Note: The calculation of bikes taken and given considers bike available
    difference between 00:00 and 06:00, so if a station finishes the day with
    1 bike at 00:00 but starts the day at 06:00 with 2 bikes, then it is
    considered to have gained one bike

    :param df: processed input -- pyspark data frame
    """
    # Create udf to convert date_status to seconds since epoch
    epoch_udf = f.udf(
        lambda x: int(time.mktime(time.strptime(x, '%d-%m-%Y %H:%M:%S'))),
        IntegerType())
    # Apply udf to get new column with epoch time for each row
    epoch_df = df.withColumn(
        'epoch_time', epoch_udf(df['date_status']))
    # Extract the data frame columns we are interested in
    select_df = epoch_df.select(epoch_df['name'], epoch_df['epoch_time'],
                                epoch_df['bikes_available'])
    # Iterate over the rows getting the values of the previous row's to
    # determine difference between station bike counts
    clone_df = select_df.select(
        '*',
        *([f.lag(f.col(c), default=0).over(
            Window.orderBy(
                select_df['name'],
                select_df['epoch_time'])).alias(
            'p_' + c) for c in select_df.columns]))
    # Create udf to take station name and bike count of the current and
    # previous rows, if the stations names match output the value of the
    # current minus the previous station, if the value is positive bikes were
    # given, if the value is negative bikes were taken, 0 value indicates no
    # change in bike count or if the bike stations do not match and cannot be
    # counted
    taken_given_udf = f.udf(
        lambda a, b, c, d: (b - d) if a in c else 0, IntegerType())
    # Apply udf to row to get bike count difference between station intervals
    taken_given_df = clone_df.withColumn(
        'diff', taken_given_udf(
            clone_df['name'], clone_df['bikes_available'],
            clone_df['p_name'], clone_df['p_bikes_available']))
    # Persist the data frame to memory for following operations
    taken_given_df.cache()
    # Filter the data frame to get only rows where the value is negative and
    # bikes were taken
    filter_df = taken_given_df.filter(taken_given_df['diff'] < 0)
    # Convert the negative value to positive
    abs_udf = f.udf(lambda x: abs(x), IntegerType())
    # Apply udf to get absolute number
    abs_taken_df = filter_df.withColumn(
        'diff', abs_udf(filter_df['diff']))
    # Aggregate the data frame by station name and sum the values of bikes
    # taken
    taken_count_df = abs_taken_df.groupBy(abs_taken_df['name']).agg(
        {'diff': 'sum'})
    # Convert the name of the column generated with sum value to more
    # meaningful value
    taken_count_df = taken_count_df.withColumnRenamed('sum(diff)',
                                                      'bikes_taken')
    # Filter the data frame to get only rows where the value is positive and
    # bikes were given
    given_df = taken_given_df.filter(taken_given_df['diff'] > 0)
    # Aggregate the data frame by station name and sum the values of bikes
    # given
    given_count_df = given_df.groupBy(given_df['name']).agg(
        {'diff': 'sum'})
    # Convert the name of the column generated with sum value to more
    # meaningful value
    given_count_df = given_count_df.withColumnRenamed('sum(diff)',
                                                      'bikes_given')
    # Join two data frames where the stations match to get a new data frame
    # containing both bikes taken and given for each station
    result_df = taken_count_df.join(given_count_df, ["name"])
    # Order the data by bikes taken minus bikes given in descending order, this
    # can be achieved easily here by using negation on the resulting summation
    ordered_df = result_df.orderBy(
        -(result_df['bikes_taken'] + result_df['bikes_given']))
    # Collect the results and output
    result = ordered_df.collect()
    for item in result:
        print(item)


###############################################################################

if __name__ == '__main__':
    # Set the location of the dataset directory
    FILE_STORE = '/FileStore/tables/cork_bike_data/'
    # Instantiate the ran out times list
    ran_out_times = [
        '06:03:00', '08:58:00', '09:28:00', '10:58:00', '12:18:00', '12:43:00',
        '13:03:00', '13:53:00', '14:28:00', '15:48:00', '16:23:00', '16:33:00',
        '16:38:00', '17:09:00', '17:29:00', '18:24:00', '19:34:00', '20:04:00',
        '20:14:00', '20:24:00', '20:49:00', '20:59:00', '22:19:00', '22:59:00',
        '23:14:00', '23:44:00']

    # Create the Spark session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    # Set log level
    spark.sparkContext.setLogLevel('WARN')
    # Load dataset and create data frame
    input_df = load_dataset_get_data_frame(spark, FILE_STORE)
    input_df = input_df.filter(input_df['status'] == 0)
    # Call the functions
    for idx, ex in enumerate([ex1, ex2, ex3, ex4, ex5]):
        print('\n#-------------#\n'
              '| Exercise: {e} |\n'
              '#-------------#'.format(e=idx + 1))
        if idx == 3:
            ex(input_df, ran_out_times)
        else:
            ex(input_df)
