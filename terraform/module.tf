# Athena Workgroup
resource "aws_athena_workgroup" "athena_workgroup" {
  name = "JsonQuery"

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true

    result_configuration {
      output_location = "s3://is459-project-athena-output/"
    }
  }
}

# Athena Database
resource "aws_athena_database" "athena_db" {
  name   = "s3jsondb"
  bucket = "is459-project-athena-output"
}

# Structured Table in Athena Database for Kaggle dataset
resource "aws_glue_catalog_table" "athena_table_kaggle" {
  database_name = "s3jsondb"
  name          = "kaggle"

  table_type = "EXTERNAL_TABLE"

  storage_descriptor {
    columns {
      name = "ActualElapsedTime"
      type = "double"
    }
    columns {
      name = "AirTime"
      type = "double"
    }
    columns {
      name = "ArrDelay"
      type = "double"
    }
    columns {
      name = "ArrTime"
      type = "double"
    }
    columns {
      name = "CRSArrTime"
      type = "bigint"
    }
    columns {
      name = "CRSDepTime"
      type = "bigint"
    }
    columns {
      name = "CRSElapsedTime"
      type = "double"
    }
    columns {
      name = "CancellationCode"
      type = "string"
    }
    columns {
      name = "Cancelled"
      type = "bigint"
    }
    columns {
      name = "CarrierDelay"
      type = "double"
    }
    columns {
      name = "DayOfWeek"
      type = "bigint"
    }
    columns {
      name = "DayofMonth"
      type = "bigint"
    }
    columns {
      name = "DepDelay"
      type = "double"
    }
    columns {
      name = "DepTime"
      type = "double"
    }
    columns {
      name = "Dest"
      type = "string"
    }
    columns {
      name = "Distance"
      type = "double"
    }
    columns {
      name = "Diverted"
      type = "bigint"
    }
    columns {
      name = "FlightNum"
      type = "bigint"
    }
    columns {
      name = "LateAircraftDelay"
      type = "double"
    }
    columns {
      name = "Month"
      type = "bigint"
    }
    columns {
      name = "NASDelay"
      type = "double"
    }
    columns {
      name = "Origin"
      type = "string"
    }
    columns {
      name = "SecurityDelay"
      type = "double"
    }
    columns {
      name = "TailNum"
      type = "string"
    }
    columns {
      name = "TaxiIn"
      type = "double"
    }
    columns {
      name = "TaxiOut"
      type = "double"
    }
    columns {
      name = "UniqueCarrier"
      type = "string"
    }
    columns {
      name = "WeatherDelay"
      type = "double"
    }
    columns {
      name = "Year"
      type = "bigint"
    }

    location = "s3://is459-project-output-data/kaggle/"

    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
  }

  parameters = {
    "classification" = "parquet"
  }

  depends_on = [aws_athena_database.athena_db]
}