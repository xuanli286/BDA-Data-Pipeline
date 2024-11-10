{
	"jobConfig": {
		"name": "is459-project-combine_coordinates_with_airline",
		"description": "",
		"role": "arn:aws:iam::324037293111:role/service-role/AWSGlueServiceRole-is459-project-airline-performance-crawler",
		"command": "glueetl",
		"version": "4.0",
		"runtime": null,
		"workerType": "G.1X",
		"numberOfWorkers": 10,
		"maxCapacity": 10,
		"jobRunQueuingEnabled": false,
		"maxRetries": 0,
		"timeout": 2880,
		"maxConcurrentRuns": 1,
		"security": "none",
		"scriptName": "is459-project-combine_coordinates_with_airline.py",
		"scriptLocation": "s3://aws-glue-assets-324037293111-us-east-1/scripts/",
		"language": "python-3",
		"spark": false,
		"sparkConfiguration": "standard",
		"jobParameters": [],
		"tags": [],
		"jobMode": "DEVELOPER_MODE",
		"createdOn": "2024-11-01T13:16:09.582Z",
		"developerMode": true,
		"connectionsList": [],
		"temporaryDirectory": "s3://aws-glue-assets-324037293111-us-east-1/temporary/",
		"logging": true,
		"glueHiveMetastore": true,
		"etlAutoTuning": true,
		"metrics": true,
		"observabilityMetrics": true,
		"bookmark": "job-bookmark-disable",
		"sparkPath": "s3://aws-glue-assets-324037293111-us-east-1/sparkHistoryLogs/",
		"flexExecution": false,
		"minFlexWorkers": null,
		"maintenanceWindow": null
	},
	"dag": {
		"node-1730466880607": {
			"nodeId": "node-1730466880607",
			"dataPreview": false,
			"previewAmount": 0,
			"inputs": [
				"node-1730466783105"
			],
			"name": "Amazon S3",
			"generatedNodeName": "AmazonS3_node1730466880607",
			"classification": "DataSink",
			"type": "S3",
			"streamingBatchInterval": 100,
			"format": "glueparquet",
			"compression": "uncompressed",
			"path": "s3://is459-project-output-data/kaggle2/",
			"partitionKeys": [],
			"schemaChangePolicy": {
				"enableUpdateCatalog": false,
				"updateBehavior": "UPDATE_IN_DATABASE",
				"database": null,
				"table": "kaggle2"
			},
			"updateCatalogOptions": "none",
			"autoDataQuality": {
				"isEnabled": false,
				"evaluationContext": null
			},
			"calculatedType": "",
			"codeGenVersion": 2
		},
		"node-1730466340658": {
			"nodeId": "node-1730466340658",
			"dataPreview": false,
			"previewAmount": 0,
			"inputs": [],
			"name": "AWS Glue Data Catalog",
			"generatedNodeName": "AWSGlueDataCatalog_node1730466340658",
			"classification": "DataSource",
			"type": "Catalog",
			"isCatalog": true,
			"database": "is459-project-airport-coordinates-database",
			"table": "combined_coordinates",
			"calculatedType": "",
			"runtimeParameters": [],
			"codeGenVersion": 2
		},
		"node-1730466783105": {
			"nodeId": "node-1730466783105",
			"dataPreview": false,
			"previewAmount": 0,
			"inputs": [
				"node-1730466342377",
				"node-1730466848490"
			],
			"name": "Join",
			"generatedNodeName": "Join_node1730466783105",
			"classification": "Transform",
			"type": "Join",
			"joinType": "left",
			"columns": [
				{
					"from": "node-1730466342377",
					"keys": [
						"origin"
					]
				},
				{
					"from": "node-1730466848490",
					"keys": [
						"right_origin"
					]
				}
			],
			"columnConditions": [
				"="
			],
			"parentsValid": true,
			"calculatedType": "",
			"codeGenVersion": 2
		},
		"node-1730466342377": {
			"nodeId": "node-1730466342377",
			"dataPreview": false,
			"previewAmount": 0,
			"inputs": [],
			"name": "Amazon S3",
			"generatedNodeName": "AmazonS3_node1730466342377",
			"classification": "DataSource",
			"type": "S3",
			"isCatalog": false,
			"format": "parquet",
			"paths": [
				"s3://is459-project-output-data/kaggle/"
			],
			"compressionType": null,
			"exclusions": [],
			"groupFiles": null,
			"groupSize": null,
			"recurse": true,
			"maxBand": null,
			"maxFilesInBand": null,
			"additionalOptions": {
				"boundedSize": null,
				"boundedFiles": null,
				"enableSamplePath": false,
				"samplePath": "s3://is459-project-output-data/kaggle/new/part-00000-4692b0d0-d8f7-4e5e-b708-4ab7ac0518ec-c000.snappy.parquet",
				"boundedOption": null
			},
			"outputSchemas": [
				[
					{
						"key": "actualelapsedtime",
						"fullPath": [
							"actualelapsedtime"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "airtime",
						"fullPath": [
							"airtime"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "arrdelay",
						"fullPath": [
							"arrdelay"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "arrtime",
						"fullPath": [
							"arrtime"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "crsarrtime",
						"fullPath": [
							"crsarrtime"
						],
						"type": "bigint",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "crsdeptime",
						"fullPath": [
							"crsdeptime"
						],
						"type": "bigint",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "crselapsedtime",
						"fullPath": [
							"crselapsedtime"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "cancellationcode",
						"fullPath": [
							"cancellationcode"
						],
						"type": "string",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "cancelled",
						"fullPath": [
							"cancelled"
						],
						"type": "bigint",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "carrierdelay",
						"fullPath": [
							"carrierdelay"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "dayofweek",
						"fullPath": [
							"dayofweek"
						],
						"type": "bigint",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "dayofmonth",
						"fullPath": [
							"dayofmonth"
						],
						"type": "bigint",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "depdelay",
						"fullPath": [
							"depdelay"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "deptime",
						"fullPath": [
							"deptime"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "dest",
						"fullPath": [
							"dest"
						],
						"type": "string",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "distance",
						"fullPath": [
							"distance"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "diverted",
						"fullPath": [
							"diverted"
						],
						"type": "bigint",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "flightnum",
						"fullPath": [
							"flightnum"
						],
						"type": "bigint",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "lateaircraftdelay",
						"fullPath": [
							"lateaircraftdelay"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "month",
						"fullPath": [
							"month"
						],
						"type": "bigint",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "nasdelay",
						"fullPath": [
							"nasdelay"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "origin",
						"fullPath": [
							"origin"
						],
						"type": "string",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "securitydelay",
						"fullPath": [
							"securitydelay"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "tailnum",
						"fullPath": [
							"tailnum"
						],
						"type": "string",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "taxiin",
						"fullPath": [
							"taxiin"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "taxiout",
						"fullPath": [
							"taxiout"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "uniquecarrier",
						"fullPath": [
							"uniquecarrier"
						],
						"type": "string",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "weatherdelay",
						"fullPath": [
							"weatherdelay"
						],
						"type": "double",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "year",
						"fullPath": [
							"year"
						],
						"type": "bigint",
						"glueStudioType": null,
						"children": null
					},
					{
						"key": "date",
						"fullPath": [
							"date"
						],
						"type": "timestamp",
						"glueStudioType": null,
						"children": null
					}
				]
			],
			"calculatedType": "",
			"codeGenVersion": 2,
			"inferSchemaChanged": true
		},
		"node-1730466848490": {
			"nodeId": "node-1730466848490",
			"dataPreview": false,
			"previewAmount": 0,
			"inputs": [
				"node-1730466340658"
			],
			"name": "Renamed keys for Join",
			"generatedNodeName": "RenamedkeysforJoin_node1730466848490",
			"classification": "Transform",
			"type": "ApplyMapping",
			"mapping": [
				{
					"toKey": "right_origin",
					"fromPath": [
						"origin"
					],
					"toType": "string",
					"fromType": "string",
					"dropped": false,
					"children": null
				},
				{
					"toKey": "right_longitude_deg",
					"fromPath": [
						"longitude_deg"
					],
					"toType": "double",
					"fromType": "double",
					"dropped": false,
					"children": null
				},
				{
					"toKey": "right_latitude_deg",
					"fromPath": [
						"latitude_deg"
					],
					"toType": "double",
					"fromType": "double",
					"dropped": false,
					"children": null
				}
			],
			"parentsValid": true,
			"calculatedType": "",
			"codeGenVersion": 2
		}
	},
	"hasBeenSaved": false,
	"usageProfileName": null,
	"script": "import sys\r\nimport pyarrow.parquet as pq\r\nimport pyarrow as pa\r\nimport pandas as pd\r\nfrom awsglue.transforms import *\r\nfrom awsglue.utils import getResolvedOptions\r\nfrom pyspark.context import SparkContext\r\nfrom awsglue.context import GlueContext\r\nfrom awsglue.job import Job\r\nimport boto3\r\nimport s3fs  # To use pyarrow with S3 paths\r\n\r\n# Get job name from arguments\r\nargs = getResolvedOptions(sys.argv, ['JOB_NAME'])\r\nsc = SparkContext()\r\nglueContext = GlueContext(sc)\r\nspark = glueContext.spark_session\r\njob = Job(glueContext)\r\njob.init(args['JOB_NAME'], args)\r\n\r\n# Define S3 bucket and prefixes\r\nbucket_name = \"is459-project-output-data\"\r\nsource_prefixes = [\"kaggle/old/\", \"kaggle/new/\"]\r\ntarget_prefix = \"weather/kaggle/airline_with_coordinates/\"\r\ncsv_key = \"combined_coordinates.csv\"\r\n\r\n# Initialize S3 client and filesystem for s3fs\r\ns3_client = boto3.client('s3')\r\nfs = s3fs.S3FileSystem()\r\n\r\n# Load coordinate data from the CSV file, selecting only necessary columns\r\ncsv_path = f\"s3://is459-project-data/weather/{csv_key}\"\r\ncoordinates_df = pd.read_csv(fs.open(csv_path))[[\"Origin\", \"longitude_deg\", \"latitude_deg\"]]\r\n\r\n# Iterate over each source prefix\r\nfor source_prefix in source_prefixes:\r\n    # List all Parquet files in the current source prefix\r\n    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=source_prefix)\r\n    for file_info in response.get('Contents', []):\r\n        file_key = file_info['Key']\r\n        if not file_key.endswith(\".parquet\"):\r\n            continue\r\n\r\n        # S3 path for the Parquet file\r\n        file_path = f\"s3://{bucket_name}/{file_key}\"\r\n\r\n        # Read the Parquet file into a Pandas DataFrame using s3fs\r\n        parquet_df = pq.ParquetDataset(file_path, filesystem=fs).read_pandas().to_pandas()\r\n\r\n        # Ensure consistent datatype for CRSArrTime\r\n        if 'CRSArrTime' in parquet_df.columns:\r\n            parquet_df['CRSArrTime'] = pd.to_numeric(parquet_df['CRSArrTime'], errors='coerce').fillna(0).astype('int64')\r\n\r\n        # Perform the join with the coordinates data on 'Origin', selecting necessary columns only\r\n        joined_df = parquet_df.merge(\r\n            coordinates_df,\r\n            on=\"Origin\",\r\n            how=\"left\"\r\n        )\r\n\r\n        # Convert the joined DataFrame to a PyArrow Table\r\n        table = pa.Table.from_pandas(joined_df)\r\n\r\n        # Construct the output path by replacing the source prefix with the target prefix\r\n        output_path = f\"s3://{bucket_name}/{file_key.replace(source_prefix, target_prefix)}\"\r\n\r\n        # Write the joined data to the new S3 path in Glue-compatible Parquet format\r\n        with fs.open(output_path, 'wb') as f:\r\n            pq.write_table(\r\n                table,\r\n                f,\r\n                compression=\"None\"\r\n            )\r\n\r\n        print(f\"Processed and saved file to {output_path}\")\r\n\r\njob.commit()\r\n\r\n"
}