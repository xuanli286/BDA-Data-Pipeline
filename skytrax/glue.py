import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import boto3
import json
import pandas as pd
from pyspark.sql import SparkSession

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

s3 = boto3.client('s3')
response = s3.list_objects_v2(Bucket="is459-project-data", Prefix="skytrax/top_100/")
files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.json')]
all_data = []

for file_key in files:
    obj = s3.get_object(Bucket="is459-project-data", Key=file_key)
    json_content = obj['Body'].read().decode('utf-8')
    
    data = json.loads(json_content)
    all_data.extend(data)
    
# IATA Designator code
iata = {
    'Qatar Airways': 'QR',
    'Asiana Airlines': 'OZ',
    'Singapore Airlines': 'SQ',
    'Cathay Pacific Airways': 'CX',
    'ANA All Nippon Airways': 'NH',
    'Etihad Airway': 'EY',
    'Turkish Airlines': 'TK',
    'Emirates': 'EK',
    'Thai Airways': 'TG',
    'Malaysia Airlines': 'MH',
    'Garuda Indonesia': 'GA',
    'Virgin Australia': 'VA',
    'EVA Air': 'BR',
    'Lufthansa': 'LH',
    'Qantas Airways': 'QF',
    'Korean Air': 'KE',
    'Air New Zealand': 'NZ',
    'Swiss International Air Lines': 'LX',
    'Air Canada': 'AC',
    'Hainan Airlines': 'HU',
    'Dragonair': 'KA',
    'AirAsia': 'AK',
    'Oman Air': 'WY',
    'Aegean Airlines': 'A3',
    'South African Airways': 'SA',
    'Virgin America': 'VX',
    'Bangkok Airways': 'PG',
    'British Airways': 'BA',
    'China Southern Airlines': 'CZ',
    'Jetstar Airways': 'JQ',
    'Finnair': 'AY',
    'TAM Airlines': 'JJ',
    'China Airlines': 'CI',
    'KLM Royal Dutch Airlines': 'KL',
    'Japan Airlines': 'JL',
    'SilkAir': 'MI',
    'Air China': 'CA',
    'LAN Airlines': 'LA',
    'Austrian Airlines': 'OS',
    'AirAsia X': 'D7',
    'EasyJet': 'U2',
    'TACA Airlines': 'TA',
    'WestJet': 'WS',
    'China Eastern Airlines': 'MU',
    'Hong Kong Airlines': 'HX',
    'Jetstar Asia': '3K',
    'Vietnam Airlines': 'VN',
    'Air France': 'AF',
    'Alaska Airlines': 'AS',
    'Virgin Atlantic': 'VS',
    'Southwest Airlines': 'WN',
    'jetBlue Airways': 'B6',
    'Air Astana': 'KC',
    'Azul Airlines': 'AD',
    'IndiGo': '6E',
    'Avianca': 'AV',
    'Delta Air Lines': 'DL',
    'Jet Airways': 'QJ',
    'bmi British Midland': 'BD',
    'American Eagle': 'MQ',
    'Copa Airlines': 'CM',
    'Air Nostrum': 'YW',
    'Brussels Airlines': 'SN',
    'United Airlines': 'UA',
    'Air Berlin': 'AB',
    'Kulula': 'MN',
    'TAP Portugal': 'TP',
    'SAS Scandinavian': 'SK',
    'TRIP Airlines': '',
    'SriLankan Airlines': 'UL',
    'Air Mauritius': 'MK',
    'Transaero Airlines': 'UN',
    'Baboo Airlines': 'QH',
    'Norwegian': 'DY',
    'NIKI': 'HG',
    'Air Transat': 'TS',
    'Hong Kong Express': 'UO',
    'Kenya Airways': 'KQ',
    'Kingfisher Airlines': 'IT',
    'Hawaiian Airlines': 'HA',
    'SpiceJet': 'SG',
    'Gulf Air': 'GF',
    'Alitalia': 'AZ',
    'American Airlines': 'AA',
    'Aeroflot': 'SU',
    'Tigerair': 'TR',
    'Saudia': 'SV',
    'Icelandair': 'FI',
    'Shenzhen Airlines': 'ZH',
    'Porter Airlines': 'P3',
    'Aer Arann': 'RE',
    'Tianjin Airlines': 'GS',
    'Nok Air': 'DD',
    'Royal Jordanian Airlines': 'RJ',
    'Egyptair': 'MS',
    'Air Pacific': 'FJ',
    'Aer Lingus': 'EI',
    'Aer Busan': 'BX',
    'Cyprus Airways': 'CY',
    'Skymark Airlines': 'BC',
    'Air Seychelles': 'HM',
    'Azerbaijan Airlines': 'J2',
    'Scoot': 'TR',
    'Thomson Airways': 'BY',
    'Ethiopian Airlines': 'ET',
    'Peach': 'MM',
    'Philippine Airlines': 'PR',
    'Germanwings': '4U',
    'Iberia': 'IB',
    'S7 Airlines': 'S7',
    'US Airways': 'US',
    'Vueling Airlines': 'VY',
    'TAAG Angola Airlines': 'DT',
    'Mango': 'JE',
    'Fiji Airways': 'FJ',
    'Croatia Airlines': 'OU',
    'Juneyao Air': 'HO',
    'Eurowings': 'EW',
    'LOT Polish': 'LO',
    'Aeromexico': 'AM',
    'Royal Brunei Airlines': 'BI',
    'Cathay Dragon': 'KA',
    'LATAM': 'LA',
    'Ryanair': 'FR',
    'Air Malta': 'KM',
    'Xiamen Airlines': 'MF',
    'Royal Air Maroc': 'AT',
    'Air Canada rouge': 'AC',
    'TUI Airways': 'X3*',
    'Air Dolomiti': 'EN',
    'Vistara': 'UK',
    'Jet2.com': 'LS',
    'PAL Express': '2P',
    'AtlasGlobal': 'KK',
    'LEVEL': 'LV',
    'Flynas': 'XY',
    'JetBlue Airways': 'B6',
    'Bamboo Airways': 'QH',
    'JetSmart': 'JA',
    'Spring Airlines': 'IJ',
    'StarFlyer': '7G',
    'Air Arabia': 'G9',
    'Wizz Air': 'W6',
    'STARLUX Airlines': 'JX',
    'Citilink': 'QG',
    'SunExpress': 'XQ',
    'airBaltic': 'BT',
    'Rex Airlines': 'ZL',
    'Kuwait Airways': 'KU',
    'flyDubai': 'FZ',
    'Sky Airline': 'H2',
    'Air Serbia': 'JU',
    'ITA Airways': 'AZ',
    'Buta Airways': 'J2',
    'Azul Brazilian': 'AD',
    'Volotea': 'V7',
    'Transavia France': 'TO',
    'Sun Country Airlines': 'SY',
    'JetSMART Airlines': 'JA',
    'SKY Airline': 'H2',
    'RwandAir': 'WB',
    'PLAY': 'OG',
    'Breeze Airways': 'MX',
    'Iberia Express': 'I2',
    'Allegiant Air': 'G4',
    'Air India': 'AI',
    'FlyArystan': 'KC',
    'Etihad Airways': 'EY'
}

columns = ["Year", "Rank", "Airline"]

df = pd.DataFrame(all_data, columns=columns)
df['UniqueCarrier'] = df['Airline'].map(iata)

spark_df = spark.createDataFrame(df)

output_path = "s3://is459-project-output-data/skytrax/"

spark_df.write.mode("overwrite").option("header", "true").csv(output_path)

job.commit()