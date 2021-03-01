# ------------------------------------------
# Big Data Processing
# Assignment 1 - Spark Core & Spark SQL
# Part 2 - Spark Core
# Michael McAleer R00143621
# ------------------------------------------
import pyspark
import time


def process_line(line):
    """Process a line from the input data rdd, cleans and returns tuple of
    values.

    :param line: raw line data -- str
    :return: line values -- tuple (str * len(params))
    """
    # Instantiate the return response
    res = ()
    # Remove new line character
    line = line.replace('\n', '')
    # Split line on delimiter
    params = line.split(';')
    # If the param count is 7 return tuple of line values else empty tuple
    if len(params) == 7 and int(params[0]) == 0:
        res = (int(params[0]), str(params[1]), float(params[2]),
               float(params[3]), str(params[4]), int(params[5]),
               int(params[6]))
    return res


###############################################################################

def ex1(rdd):
    """Exercise 1: Number of times each station ran out of bikes (sorted
    decreasingly by station).

    :param rdd: processed input rdd -- pyspark rdd
    """
    # Filter the data so only valid intervals and intervals where the bike
    # count is zero is returned
    filter_rdd = (rdd
                  .filter(lambda x: True if x and x[5] == 0 else False)
                  .map(lambda x: tuple([x[1], 1])))
    # Reduce the RDD to distinct keys and key counter
    count = filter_rdd.reduceByKey(lambda x, y: x + y)
    # Sort the results by amount of times a station ran out of bikes
    sorted_rdd = count.sortBy(lambda x: x[1], ascending=False)
    # Collect the results and output
    res_val = sorted_rdd.collect()
    for item in res_val:
        print(item)


###############################################################################


def ex2(rdd):
    """Exercise 2: Pick one busy day with plenty of ran outs -> 27/08/2017
    Average amount of bikes per station and hour window.

    :param rdd: processed input rdd -- pyspark rdd
    """
    # Filter the data returning only data which matches date, transform
    # results for average bike count calculation
    filter_rdd = (
        rdd
        .filter(lambda x: True if x and '27-08-2017' in x[4] else False)
        .map(map_available_bikes_by_station_hour))
    # Reduce by key each station-hour, aggregate the total station bike count
    reduced_rdd = filter_rdd.reduceByKey(lambda x, y: x + y)
    # Divide the hourly total bike count by 12 (12 x 5min intervals per hour)
    solution_rdd = reduced_rdd.mapValues(lambda x: x / 12)
    # Sort the results by station alphabetically and ascending hour from 06:00
    sorted_rdd = solution_rdd.sortBy(lambda x: x[0])
    # Collect the results and output
    res_val = sorted_rdd.collect()
    for item in res_val:
        print(item)


def map_available_bikes_by_station_hour(x):
    """Transform data into required format for bike average calculation.

    :param x: element from dataset -- RDD element
    :return: station hour, bike count -- tuple
    """
    # Split the date time value
    date_time = x[4].split(' ')
    # Extract the hour value
    interval_hour = date_time[1][:2]
    # Create the 'Station 00:00' key
    station_hour_key = '{station} {hour}:00'.format(station=x[1],
                                                    hour=interval_hour)
    return tuple([station_hour_key, int(x[5])])

###############################################################################


def ex3(rdd):
    """Exercise 3: Pick one busy day with plenty of ran outs -> 27/08/2017
    Get the different ran-outs to attend.

    Note: n consecutive measurements of a station being ran-out of bikes has to
    be considered a single ran-out, that should have been attended when the
    ran-out happened in the first time.

    :param rdd: processed input rdd -- pyspark rdd
    """
    date = '27-08-2017'
    # Filter data to get only elements that match the date required and have
    # a bike count of zero, transform to the format (station name, epoch time)
    filter_rdd = (
        rdd
        .filter(lambda x: True if x and date in x[4] and not x[5] else False)
        .map(lambda x: tuple([str(x[1]), convert_time_to_epoch(x[4])])))
    # Group the data by station so each key contains a list of epoch times
    grouped_rdd = filter_rdd.groupByKey()
    # Flat map the values so each station is paired with each of their ran out
    # times
    ran_out_rdd = grouped_rdd.flatMapValues(ran_out_map)
    # Invert and sort the result so we get results listed by time then station
    inverted_rdd = ran_out_rdd.map(lambda x: (x[1], x[0])).sortByKey()
    # Collect the results and output
    res_val = inverted_rdd.collect()
    for item in res_val:
        print(item)


def convert_time_to_epoch(str_time_in):
    """Convert a date time string to milliseconds since epoch.

    :param str_time_in: date time -- str
    :return: epoch time -- int
    """
    return int(
        time.mktime(time.strptime(str_time_in, '%d-%m-%Y %H:%M:%S')))


def convert_epoch_to_hour_min_string(time_in):
    """Convert epoch time to hour:minute string

    :param time_in: epoch time -- int
    :return: hour:minute -- str
    """
    return time.strftime('%H:%M', time.gmtime(time_in))


def ran_out_map(x):
    """Transform value data to determine the all occurrences of a bike station
    running out of bikes taking into consideration consecutive intervals with
    no bikes.

    :param x: element from dataset -- RDD element
    :return: ran out times -- list
    """
    # Sort the list of ran out times
    ran_out_times_list = sorted(list(x))
    # Instantiate our list to hold the ran out times for each station
    ran_out_result = list()
    for i in range(0, len(ran_out_times_list)):
        # If our ran out list is empty we put the first ran out time into it
        if not ran_out_result:
            ran_out_result.append(
                convert_epoch_to_hour_min_string(ran_out_times_list[i]))
        # Else we need to compare the last ran time with the current ran out
        # time to determine if enough time has passed to count it as a ran out
        else:
            # Time has been adjusted to allow for a six minute window
            # accounting for outliers in dataset not conforming to 5 minute
            # interval standard
            if not (ran_out_times_list[i] - ran_out_times_list[i - 1]) <= 360:
                ran_out_result.append(
                    convert_epoch_to_hour_min_string(ran_out_times_list[i]))
    return ran_out_result


###############################################################################


def ex4(rdd, ro_times):
    """Exercise 4: Pick one busy day with plenty of ran outs -> 27/08/2017
    Get the station with biggest number of bikes for each ran-out to be
    attended.

    :param rdd: processed input rdd -- pyspark rdd
    :param ro_times: ran out times -- set
    """
    # Run the filter first, first step out is to reduce the size of the RDD
    filtered_rdd = (
        rdd
        .filter(lambda x: time_date_filter(x, ro_times))
        .map(lambda x: tuple([str(x[4].split(' ')[1]),
                              tuple([str(x[1]), int(x[5])])])))
    # Get the max value for each time key and sort the results by time
    result = filtered_rdd.reduceByKey(
        lambda x, y: max([x, y], key=lambda a: a[1])).sortByKey()
    # Collect the results and output
    res_val = result.collect()
    for item in res_val:
        print(item)


def time_date_filter(x, ro_set):
    """Filter data so only elements that match the date requirement and the
    hour:minute interval is in the ran out set. A set is used for hash lookup
    linear search time instead of iterative search in list.

    :param x: element from dataset -- RDD element
    :param ro_set: ran out times -- set
    :return: bool
    """
    try:
        # Split the date/time for comparison
        date_time = x[4].split(' ')
        i_date, i_time = date_time[0], date_time[1]
    except IndexError:
        # If the split command throws an index error then the input element
        # is empty
        return False

    return True if '27-08-2017' in i_date and i_time in ro_set else False


###############################################################################


def ex5(rdd):
    """Exercise 5: Total number of bikes that are taken and given back per
    station (sort the results in decreasing order in the sum of bikes taken +
    bikes given back).

    Note: The calculation of bikes taken and given considers bike available
    difference between 00:00 and 06:00, so if a station finishes the day with
    1 bike at 00:00 but starts the day at 06:00 with 2 bikes, then it is
    considered to have gained one bike

    :param rdd: processed input rdd -- pyspark rdd
    """
    # Filter data for only valid elements and transform data for calculating
    # taken and given values (station name, (epoch time, bike count))
    map_rdd = (rdd
               .filter(lambda x: True if x and x[0] == 0 else False)
               .map(lambda x: tuple([str(x[1]),
                                     tuple([convert_time_to_epoch(x[4]),
                                            int(x[5])])
                                     ])))
    # Group the elements by key so each station has a list of times and bike
    # counts
    combined_rdd = map_rdd.groupByKey()
    # Transform the values to calculate the total bikes taken and given per
    # station
    mapped_values = combined_rdd.mapValues(map_taken_given)
    # Sort the elements by the sum of the bikes taken and bikes given
    sorted_rdd = mapped_values.sortBy(lambda x: x[1][0] + x[1][1],
                                      ascending=False)
    # Collect the results and output
    res = sorted_rdd.collect()
    for item in res:
        print(item)


def map_taken_given(x):
    """Transform value data, calculate the amount of bikes taken and given back
    between intervals.

    :param x: element from dataset -- RDD element
    :return: bikes taken, bikes given -- tuple
    """
    # Instantiate interval list and taken/given counters
    x, taken, given = list(x), 0, 0
    # For every interval except for the last
    for i in range(0, len(x) - 1):
        # Get the current interval and the next interval in the list for
        # comparison
        current_cnt, nex_cnt = x[i][1], x[i + 1][1]
        # If the difference results in a positive number, bikes have been taken
        if (current_cnt - nex_cnt) > 0:
            # Increment bikes taken counter by difference
            taken += (current_cnt - nex_cnt)
        # If the difference results in a negative number, bikes have been given
        if (current_cnt - nex_cnt) < 0:
            # Increment bikes given counter by absolute difference
            given += abs(current_cnt - nex_cnt)

    return tuple([taken, given])


###############################################################################


if __name__ == '__main__':
    # Set the location of the dataset directory
    FILE_STORE = '/FileStore/tables/cork_bike_data'
    # Instantiate the ran out times set
    ran_out_times = {
        '06:03:00', '08:58:00', '09:28:00', '10:58:00', '12:18:00', '12:43:00',
        '13:03:00', '13:53:00', '14:28:00', '15:48:00', '16:23:00', '16:33:00',
        '16:38:00', '17:09:00', '17:29:00', '18:24:00', '19:34:00', '20:04:00',
        '20:14:00', '20:24:00', '20:49:00', '20:59:00', '22:19:00', '22:59:00',
        '23:14:00', '23:44:00'}
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
    for idx, ex in enumerate([ex1, ex2, ex3, ex4, ex5]):
        print('\n#-------------#\n'
              '| Exercise: {e} |\n'
              '#-------------#'.format(e=idx + 1))
        if idx == 3:
            ex(input_rdd, ran_out_times)
        else:
            ex(input_rdd)
