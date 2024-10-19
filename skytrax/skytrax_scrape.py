from bs4 import BeautifulSoup
from datetime import datetime
import boto3
import json
import requests

current_year = datetime.now().year

airlines_list = []

idx = 1
url = f"https://www.worldairlineawards.com/worlds-top-100-airlines-{current_year}/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
for div in soup.find_all('div', class_='row mb-2 awards-list'):
    airline = div.find_all('h4', class_='mb-0 text-responsive-h4')
    airlines_list.append((current_year, idx, airline[0].text.strip()))
    idx += 1
print(f"Completed scraping {current_year}")

s3 = boto3.client('s3')
try:
    s3.put_object(
        Bucket='is459-project-data', 
        Key=f'skytrax/skytrax_top_100_airlines_{current_year}.json',
        Body=json.dumps(airlines_list),
        ContentType='application/json'
    )
    print("Files uploaded to S3 successfully")
except Exception as e:
    print("Error uploading to S3: ", e)