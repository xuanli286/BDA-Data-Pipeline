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
      type = "double"
    }
    columns {
      name = "CarrierDelay"
      type = "double"
    }
    columns {
      name = "Date"
      type = "date"
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
      type = "double"
    }
    columns {
      name = "FlightNum"
      type = "int"
    }
    columns {
      name = "LateAircraftDelay"
      type = "double"
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

# Structured Table in Athena Database for Reddit & Skytrax Reviews dataset
resource "aws_glue_catalog_table" "athena_table_reddit_skytrax" {
  database_name = "s3jsondb"
  name          = "reddit_skytrax"

  table_type = "EXTERNAL_TABLE"

  parameters = {
    classification     = "csv"
    "skip.header.line.count" = "1"
  }

  storage_descriptor {
    location      = "s3://is459-project-output-data/reddit/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"
      parameters = {
        "field.delim" = ","
      }
    }

    columns {
      name = "id"
      type = "string"
    }
    columns  {
      name = "date"
      type = "string"
    }
    columns {
      name = "content"
      type = "string"
    }
    columns {
      name = "code"
      type = "string"
    }
    columns {
      name = "topic"
      type = "string"
    }
    columns {
      name = "sentiment"
      type = "string"
    }
  }

  depends_on = [aws_athena_database.athena_db]
}

# Structured Table in Athena Database for Skytrax Ranking dataset
resource "aws_glue_catalog_table" "athena_table_skytrax_rank" {
  database_name = "s3jsondb"
  name          = "skytrax_rank"

  table_type = "EXTERNAL_TABLE"

  parameters = {
    classification     = "csv"
    "skip.header.line.count" = "1"
  }

  storage_descriptor {
    location      = "s3://is459-project-output-data/skytrax/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"
      parameters = {
        "field.delim" = ","
      }
    }

    columns {
      name = "year"
      type = "int"
    }
    columns  {
      name = "rank"
      type = "int"
    }
    columns {
      name = "airline"
      type = "string"
    }
    columns {
      name = "unique_carrier"
      type = "string"
    }
  }

  depends_on = [aws_athena_database.athena_db]
}

# Structured Table in Athena Database for airport dataset
resource "aws_glue_catalog_table" "athena_table_airport" {
  database_name = "s3jsondb"
  name          = "airport"

  table_type = "EXTERNAL_TABLE"

  parameters = {
    classification     = "csv"
    "skip.header.line.count" = "1"
  }

  storage_descriptor {
    location      = "s3://is459-project-output-data/airport/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"
      parameters = {
        "field.delim" = ","
      }
    }

    columns {
      name = "airport_name"
      type = "string"
    }
    columns {
      name = "country"
      type = "string"
    }
    columns {
      name = "latitude"
      type = "double"
    }
    columns {
      name = "longitude"
      type = "double"
    }
    columns {
      name = "airport_code"
      type = "string"
    }
  }

  depends_on = [aws_athena_database.athena_db]
}

# Structured Table in Athena Database for carriers dataset
resource "aws_glue_catalog_table" "athena_table_carriers" {
  database_name = "s3jsondb"
  name          = "carriers"

  table_type = "EXTERNAL_TABLE"

  parameters = {
    classification     = "csv"
    "skip.header.line.count" = "1"
  }

  storage_descriptor {
    location      = "s3://is459-project-output-data/carrier/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.serde2.OpenCSVSerde"
      parameters = {
        "separatorChar" = ","
        "quoteChar"     = "\""
      }
    }

    columns {
      name = "code"
      type = "string"
    }
    columns  {
      name = "description"
      type = "string"
    }
  }

  depends_on = [aws_athena_database.athena_db]
}