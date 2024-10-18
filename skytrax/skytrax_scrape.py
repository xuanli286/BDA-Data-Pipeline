import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

start_year = 2012
end_year = datetime.now().year

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

df = pd.DataFrame(airlines_list, columns=['Year', 'Rank', 'Airline'])
df.to_csv('skytrax_top_100_airlines.csv', index=False)