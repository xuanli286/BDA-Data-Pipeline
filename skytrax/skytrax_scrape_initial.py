from bs4 import BeautifulSoup
from datetime import datetime
import boto3
import json
import requests

start_year = 2012
end_year = datetime.now().year - 1

airlines_list = []

for year in range(start_year, end_year+1):
    idx = 1
    if year > 2017:
        url = f"https://www.worldairlineawards.com/worlds-top-100-airlines-{year}/"
    else:
        url = f"https://www.worldairlineawards.com/the-worlds-top-100-airlines-{year}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    if year == 2017 and idx == 1:
        for div in soup.find_all('div', class_='row mb-g awards-list'):
            airline = div.find_all('h2', class_='d-none d-lg-block text-responsive-h3')
            airlines_list.append((year, idx, airline[0].text.strip()))
            idx += 1
    for div in soup.find_all('div', class_='row mb-2 awards-list'):
        airline = div.find_all('h4', class_='mb-0 text-responsive-h4')
        airlines_list.append((year, idx, airline[0].text.strip()))
        idx += 1
    print(f"Completed scraping {year}")

s3 = boto3.client('s3')
try:
    s3.put_object(
        Bucket='is459-project-data', 
        Key='skytrax/skytrax_top_100_airlines.json',
        Body=json.dumps(airlines_list),
        ContentType='application/json'
    )
    print("Files uploaded to S3 successfully")
except Exception as e:
    print("Error uploading to S3: ", e)