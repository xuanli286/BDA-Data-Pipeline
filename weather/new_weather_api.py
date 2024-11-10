import asyncio
import aiohttp
import boto3
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Create an S3 client
s3 = boto3.client('s3')

# Define the bucket name
bucket_name = 'is459-project-data'
folder_prefix = 'weather'
MAX_CONCURRENT_REQUESTS = 10

async def fetch_data(session, latitude, longitude, start_date, end_date, retries=3):
    url = f"https://archive-api.open-meteo.com/v1/era5?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,wind_speed_10m,precipitation,rain,snowfall,visibility"
    
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                response.raise_for_status()  # Raise an error for bad responses
                return await response.json()
        except aiohttp.ClientResponseError as e:
            if e.status == 429:  # Too Many Requests
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time} seconds before retrying...")
                await asyncio.sleep(wait_time)  # Use asyncio.sleep for async wait
            else:
                print(f"Error fetching data: {e}")
                break  # Break if it's not a rate limit error
    return None  # Return None if all retries failed

async def process_coordinates(unique_coordinates, start_date, end_date):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = []

        for row in unique_coordinates:
            latitude = row['latitude_deg']
            longitude = row['longitude_deg']

            # Wrap the fetch_data call with the semaphore to limit concurrency
            task = fetch_data_with_semaphore(semaphore, session, latitude, longitude, start_date, end_date)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

async def fetch_data_with_semaphore(semaphore, session, latitude, longitude, start_date, end_date):
    async with semaphore:
        return await fetch_data(session, latitude, longitude, start_date, end_date)

def lambda_handler(event, context):
    # Fetch the object
    try:
        response = s3.get_object(Bucket=bucket_name, Key=f"{folder_prefix}/combined_coordinates.json")
        # Read the content of the file
        content = response['Body'].read()
        airline_coordinates = json.loads(content)
        print("combined_coordinates.json fetched successfully")

    except Exception as e:
        print(f"Error fetching object: {e}")
        return

    unique_coordinates = [
        {'longitude_deg': coord['longitude_deg'], 'latitude_deg': coord['latitude_deg']}
        for coord in airline_coordinates
    ]

    now_date = datetime.now()

    # Subtract one month to get the previous month
    previous_month = now_date - relativedelta(months=1)

    year = previous_month.year
    month = previous_month.month

        # Define months
    months = [
        (f"{year}-01-01", f"{year}-01-31"),
        (f"{year}-02-01", f"{year}-02-28" if year % 4 != 0 else f"{year}-02-29"),  # Leap year check
        (f"{year}-03-01", f"{year}-03-31"),
        (f"{year}-04-01", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-10-31"),
        (f"{year}-11-01", f"{year}-11-30"),
        (f"{year}-12-01", f"{year}-12-31"),
    ]

    start_date, end_date = months[month - 1]

    yearly_data = []
    
    # Process coordinates asynchronously
    results = asyncio.run(process_coordinates(unique_coordinates, start_date, end_date))

    # Flatten the results and prepare for saving
    for index, hourly_data in enumerate(results):
        if hourly_data and 'hourly' in hourly_data:
            latitude = unique_coordinates[index]['latitude_deg']
            longitude = unique_coordinates[index]['longitude_deg']
            for i in range(len(hourly_data['hourly']['time'])):
                # print(hourly_data['hourly']['time'][i])
                data_entry = {
                    'time': hourly_data['hourly']['time'][i],
                    'temperature_2m': hourly_data['hourly']['temperature_2m'][i],
                    'wind_speed_10m': hourly_data['hourly']['wind_speed_10m'][i],
                    'precipitation': hourly_data['hourly']['precipitation'][i],
                    'rain': hourly_data['hourly']['rain'][i],
                    'snowfall': hourly_data['hourly']['snowfall'][i],
                    'visibility': hourly_data['hourly']['visibility'][i],
                    'latitude_deg': latitude,
                    'longitude_deg': longitude,
                    'start_date': start_date,
                    'end_date': end_date
                }
                yearly_data.append(data_entry)

    # Save to S3 as a JSON file for the current month
    if yearly_data:
        try:
            s3.put_object(
                Bucket=bucket_name, 
                Key=f"{folder_prefix}/weather_data/{year}/{month}.json",
                Body=json.dumps(yearly_data),
                ContentType='application/json'
            )
            print(f"{folder_prefix}/weather_data/{year}/{month}.json uploaded to S3 successfully")

        except Exception as e:
            print("Error uploading to S3: ", e)