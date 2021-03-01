# ------------------------------------------
# Big Data Processing
# Assignment 1 - Spark Core & Spark SQL
# Part 3 - Spark SQL
# Michael McAleer R00143621
# ------------------------------------------
import pyspark

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
    """Exercise 1: Total amount of entries in the dataset.

    :param df: processed input data frame -- pyspark data frame
    """
    # Count the total amount of entries in the df and print
    print('- Total dataset entries: {c}'.format(c=df.count()))


###############################################################################

def ex2(df):
    """Exercise 2: Number of Coca-cola bikes stations in Cork.

    :param df: processed input data frame -- pyspark data frame
    """
    # Extract the distinct station names from input RDD
    distinct = df.select('name').distinct()
    # Output the amount of unique values
    print('- Total bike stations in Cork: {c}'.format(c=distinct.count()))


###############################################################################

def ex3(df):
    """Exercise 3: List of Coca-Cola bike stations.

    Note: It would be more efficient to put this in ex2() and collect from
    there so the distinct value count does not need to be computed twice. In a
    real world scenario this approach would be taken.

    :param df: processed input data frame -- pyspark data frame
    """
    # Extract the distinct station names from input RDD
    distinct_df = df.select('name').distinct()
    # Persist the RDD to memory for the count function
    distinct_df.cache()
    # Output the data
    print('Cork Bike Station List:')
    distinct_df.show(distinct_df.count(), False)


###############################################################################

def ex4(df):
    """Exercise 4: Sort the bike stations by their longitude (East to West).

    :param df: processed input data frame -- pyspark data frame
    """
    # Drop all columns which are not required
    location_df = df.drop('status', 'latitude', 'date_status',
                          'bikes_available', 'docks_available')
    # Drop all duplicates so only unique values remain
    unique_df = location_df.drop_duplicates()
    # Sort the stations by longitude from East to West (descending)
    ordered_df = unique_df.orderBy(unique_df['longitude'].desc())
    # Collect the results and output
    result = ordered_df.collect()
    print('Cork Bike Stations (East -> West):')
    for item in result:
        print(item)


###############################################################################

def ex5(df):
    """Exercise 5: Average number of bikes available at Kent Station.

    :param df: processed input data frame -- pyspark data frame
    """
    # Drop all columns which are not required
    location_df = df.drop('status', 'latitude', 'longitude',
                          'date_status', 'docks_available')
    # Filter the data elements for 'Kent Station', aggregate the bike
    # available count, output the average
    result = location_df.filter(location_df['name'] == 'Kent Station').agg(
        {'bikes_available': 'avg'})
    # Output the result
    print('Kent St. Average Bikes Available:', result.collect()[0][0])


###############################################################################

if __name__ == '__main__':
    # Set the location of the dataset directory
    FILE_STORE = '/FileStore/tables/cork_bike_data/'
    # Create the Spark session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    # Set log level
    spark.sparkContext.setLogLevel('WARN')
    # Load dataset and create data frame
    input_df = load_dataset_get_data_frame(spark, FILE_STORE)
    # Call the functions
    for i, ex in enumerate([ex1, ex2, ex3, ex4, ex5]):
        print('\n#-------------#\n'
              '| Exercise: {e} |\n'
              '#-------------#'.format(e=i + 1))
        ex(input_df)
