import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from pyspark.context import SparkContext
from awsglue.job import Job
from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType, DateType

# Initialize GlueContext
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)


# Define S3 paths with separate buckets for airline and weather data
airline_bucket = "s3://is459-project-output-data"
weather_bucket = "s3://is459-project-data"
airline_data_path = f"{airline_bucket}/weather/kaggle/airline_with_coordinates/"
weather_data_path = f"{weather_bucket}/weather/weather_data_parquet/"
output_path = f"s3://is459-project-output-data/weather/kaggle/airline_with_weather/"

# Load all files in the airline and weather folders as DynamicFrames
airline_dyf = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": [airline_data_path], "recurse": True},
    format="parquet"
)
weather_dyf = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": [weather_data_path], "recurse": True},
    format="parquet"
)

# Convert DynamicFrames to DataFrames
airline_df = airline_dyf.toDF()
weather_df = weather_dyf.toDF()

# print("Airline DataFrame Schema:")
# airline_df.printSchema()
# print("Weather DataFrame Schema:")
# weather_df.printSchema()

# Extract year, month, and day from Date if present in airline_df
if "Date" in airline_df.columns:
    airline_df = airline_df.withColumn("Year", F.year(F.col("Date")).cast("integer")) \
                          .withColumn("Month", F.month(F.col("Date")).cast("integer")) \
                          .withColumn("DayOfMonth", F.dayofmonth(F.col("Date")).cast("integer"))

# Continue with other transformations on airline_df
# Calculate CRSDepHourRaw and CRSDepMinute without changing CRSDepTime's type
airline_df = airline_df \
    .withColumn("CRSDepHourRaw", (F.col("CRSDepTime") / 100).cast(IntegerType())) \
    .withColumn("CRSDepMinute", (F.col("CRSDepTime") % 100).cast(IntegerType())) \
    .withColumn("CRSDepHour", F.when(F.col("CRSDepMinute") >= 30, F.col("CRSDepHourRaw") + 1)
                .otherwise(F.col("CRSDepHourRaw"))) \
    .withColumn("airline_lat", F.col("latitude_deg").cast(DoubleType())) \
    .withColumn("airline_long", F.col("longitude_deg").cast(DoubleType())) \
    .drop("latitude_deg", "longitude_deg")  # Remove original columns if not needed

# Parsing and casting columns in weather_df
weather_df = weather_df.withColumn("wyear", F.year(F.col("time")).cast("integer")) \
                      .withColumn("wmonth", F.month(F.col("time")).cast("integer")) \
                      .withColumn("wday", F.dayofmonth(F.col("time")).cast("integer")) \
                      .withColumn("whour", F.hour(F.col("time")).cast("integer")) \
                      .withColumn("wlatitude_deg", F.col("latitude_deg").cast("double")) \
                      .withColumn("wlongitude_deg", F.col("longitude_deg").cast("double")) \
                      .drop("latitude_deg", "longitude_deg")  # Remove original columns after renaming

# Print schemas for verification (optional)
# print("Airline DataFrame Schema after transformations:")
# airline_df.printSchema()
# print("Weather DataFrame Schema after transformations:")
# weather_df.printSchema()

# Join DataFrames based on year, month, day, hour, and nearest coordinates
joined_df = airline_df.join(
    weather_df,
    (airline_df["Year"] == weather_df["wyear"]) &
    (airline_df["Month"] == weather_df["wmonth"]) &
    (airline_df["DayOfMonth"] == weather_df["wday"]) &
    (airline_df["CRSDepHour"] == weather_df["whour"]) &
    (F.round(airline_df["airline_lat"], 4) == F.round(weather_df["wlatitude_deg"], 4)) &
    (F.round(airline_df["airline_long"], 4) == F.round(weather_df["wlongitude_deg"], 4)),
    how="left"
)

# Select and rename columns for the final output, keeping the original Date column
final_df = joined_df.select(
    "Date",
    "Year",
    "Month",
    "DayofMonth",
    "UniqueCarrier", "FlightNum", "Origin", "Dest", "CRSDepTime",
    "DepTime", "DepDelay", "ArrTime", "ArrDelay", "Cancelled", "Diverted", "AirTime",
    "Distance", "CarrierDelay", "WeatherDelay", "NASDelay", "LateAircraftDelay", "CRSArrTime", "ActualElapsedTime", "CRSElapsedTime", "TaxiIn", "TaxiOut", "CancellationCode",
    "SecurityDelay", "temperature_2m", "wind_speed_10m", "precipitation", "rain", "snowfall",
    F.col("airline_long").alias("longitude"), F.col("airline_lat").alias("latitude"),
    "CRSDepHour", "wlatitude_deg", "wlongitude_deg"
)

# Cast columns in final_df to enforce consistent types
# final_df = final_df \
#     .withColumn("CRSArrTime", F.col("CRSArrTime").cast("long")) \
#     .withColumn("CRSDepTime", F.col("CRSDepTime").cast("long")) \
#     .withColumn("DepTime", F.col("DepTime").cast(DoubleType())) \
#     .withColumn("ActualElapsedTime", F.col("ActualElapsedTime").cast(DoubleType())) \
#     .withColumn("AirTime", F.col("AirTime").cast(DoubleType())) \
#     .withColumn("ArrDelay", F.col("ArrDelay").cast(DoubleType())) \
#     .withColumn("DepDelay", F.col("DepDelay").cast(DoubleType())) \
#     .withColumn("Cancelled", F.col("Cancelled").cast(DoubleType())) \
#     .withColumn("Diverted", F.col("Diverted").cast(DoubleType())) \
#     .withColumn("Year", F.col("Year").cast(IntegerType())) \
#     .withColumn("Month", F.col("Month").cast(IntegerType())) \
#     .withColumn("DayofMonth", F.col("DayofMonth").cast(IntegerType())) \
#     .withColumn("UniqueCarrier", F.col("UniqueCarrier").cast('string')) \
#     .withColumn("CarrierDelay", F.col("CarrierDelay").cast(DoubleType())) \
#     .withColumn("WeatherDelay", F.col("WeatherDelay").cast(DoubleType())) \
#     .withColumn("NASDelay", F.col("NASDelay").cast(DoubleType())) \
#     .withColumn("LateAircraftDelay", F.col("LateAircraftDelay").cast(DoubleType()))

# Handle nullable columns if necessary by filling with default values
final_df = final_df.fillna({"CarrierDelay": 0, "WeatherDelay": 0, "NASDelay": 0, "LateAircraftDelay": 0})

# Define schema for final_df
schema = StructType([
    StructField("Date", DateType(), True),  # Assuming "Date" is a date field
    StructField("Year", IntegerType(), True),
    StructField("Month", IntegerType(), True),
    StructField("DayofMonth", IntegerType(), True),
    StructField("UniqueCarrier", StringType(), True),
    StructField("FlightNum", IntegerType(), True),
    StructField("Origin", StringType(), True),
    StructField("Dest", StringType(), True),
    StructField("CRSDepTime", IntegerType(), True),  # Assuming it can have bigint values
    StructField("DepTime", DoubleType(), True),
    StructField("DepDelay", DoubleType(), True),
    StructField("ArrTime", DoubleType(), True),
    StructField("ArrDelay", DoubleType(), True),
    StructField("Cancelled", DoubleType(), True),
    StructField("Diverted", DoubleType(), True),  # Sometimes could be 0 or 1 as float
    StructField("AirTime", DoubleType(), True),
    StructField("Distance", DoubleType(), True),
    StructField("CarrierDelay", DoubleType(), True),
    StructField("WeatherDelay", DoubleType(), True),
    StructField("NASDelay", DoubleType(), True),
    StructField("LateAircraftDelay", DoubleType(), True),
    StructField("CRSArrTime", LongType(), True),
    StructField("ActualElapsedTime", DoubleType(), True),
    StructField("CRSElapsedTime", DoubleType(), True),
    StructField("TaxiIn", DoubleType(), True),
    StructField("TaxiOut", DoubleType(), True),
    StructField("CancellationCode", StringType(), True),
    StructField("SecurityDelay", DoubleType(), True),
    StructField("temperature_2m", DoubleType(), True),
    StructField("wind_speed_10m", DoubleType(), True),
    StructField("precipitation", DoubleType(), True),
    StructField("rain", DoubleType(), True),
    StructField("snowfall", DoubleType(), True),
    StructField("longitude", DoubleType(), True),  # Renamed from "airline_long"
    StructField("latitude", DoubleType(), True),   # Renamed from "airline_lat"
    StructField("CRSDepHour", IntegerType(), True),
    StructField("wlatitude_deg", DoubleType(), True),
    StructField("wlongitude_deg", DoubleType(), True)
])

# Cast columns in final_df to enforce consistent types, apply schema by creating a DataFrame
final_df = spark.createDataFrame(final_df.rdd, schema=schema)

# Repartition for better performance during write
final_df = final_df.repartition(10)

final_df.write.mode("overwrite").parquet(output_path)

# Commit the job
job.commit()