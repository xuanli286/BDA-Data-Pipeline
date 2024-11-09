import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, concat_ws, to_date, from_unixtime
from awsglue.context import GlueContext
from awsglue.job import Job

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

s3_folder_path1 = "s3://is459-project-data/kaggle/old/"
s3_folder_path2 = "s3://is459-project-data/kaggle/new/"

parquet_dyf1 = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    format="parquet",
    connection_options={"paths": [s3_folder_path1]}
)
old_df = parquet_dyf1.toDF()

parquet_dyf2 = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    format="parquet",
    connection_options={"paths": [s3_folder_path2]}
)
new_df = parquet_dyf2.toDF()

old_df = old_df.select(sorted(old_df.columns))
old_df = old_df.withColumn("Date", concat_ws("-", col("Year"), col("Month"), col("DayofMonth")))
old_df = old_df.withColumn("Date", col("Date").cast("date"))
old_df = old_df.withColumn("Diverted", old_df["Diverted"].cast(DoubleType()))
old_df = old_df.drop('DayOfWeek', 'DayofMonth', 'Month', 'TailNum', 'Year')

new_df = new_df.select(sorted(new_df.columns))
new_df = new_df.drop('WheelsOff', 'WheelsOn')
new_df = new_df.withColumn("Date", to_date(new_df["Date"], "yyyy-MM-dd"))

old_df.write.mode("overwrite").parquet("s3://is459-project-output-data/kaggle/old")
new_df.write.mode("overwrite").parquet("s3://is459-project-output-data/kaggle/new")

job.commit()