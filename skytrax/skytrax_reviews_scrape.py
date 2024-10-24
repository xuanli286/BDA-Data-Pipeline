import json
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
import pandas as pd

async def fetch_page(session, url, headers):
    async with session.get(url, headers=headers) as response:
        return await response.text()

async def scrape_reviews(airline, session, url, headers):
    print(f"Scraping {airline} reviews")
    html = await fetch_page(session, url + airline, headers)
    bs = BeautifulSoup(html, "html.parser")

    # Find the last page number
    last_page_tag = bs.find("article", {"class": "comp comp_reviews-pagination querylist-pagination position-"})
    if last_page_tag:
        last_page = int(last_page_tag.find_all("li")[-2].text.strip())
    else:
        last_page = 1

    reviews = []
    tasks = []
    for i in range(1, last_page + 1):
        page_url = f"{url+airline}/page/{i}/"
        tasks.append(fetch_page(session, page_url, headers))

    pages_content = await asyncio.gather(*tasks)

    for internal_html in pages_content:
        internal_bs = BeautifulSoup(internal_html, "html.parser")
        container = internal_bs.find("article", {"class": "comp comp_reviews-airline querylist position-content"})
        if container:
            for items in container.find_all("article"):
                verified = False
                rating = items.find("div", {"class": "rating-10"})
                if rating:
                    rating = int(rating.text.strip()[:1])
                else:
                    rating = 0
                title = items.find("h2", {"class": "text_header"})
                if title:
                    title = title.text.strip()
                    title = re.sub(r'[“”"]', '', title)
                else:
                    title = None
                username = items.find("h3").find("span", {"itemprop": "name"})
                if username:
                    username = username.text.strip()
                else:
                    username = None
                publishedDate = items.find("h3").find("time")
                if publishedDate:
                    publishedDate = publishedDate.text.strip()
                else:
                    publishedDate = None
                text = items.find("div", {"class": "text_content"}).text.strip()
                text = text.split("|")
                if len(text) == 1:
                    review = text[0].strip()
                    review = re.sub(r'[’]', "'", review)
                    review = re.sub(r'[“”]', '"', review)
                else:
                    if text[0] == '✅ Trip Verified ':
                        verified = True
                    review = text[1].strip()
                    review = re.sub(r'[’]', "'", review)
                    review = re.sub(r'[“”]', '"', review)

                recommend = items.find("table", {"class": "review-ratings"}).find_all("tr")[-1].find("td", {"class": "review-value"})
                if recommend:
                    recommend = recommend.text.strip()
                else:
                    recommend = None

                data = {
                    "airline": airline,
                    "username": username,
                    "rating": rating,
                    "title": title,
                    "publishedDate": publishedDate,
                    "verified": verified,
                    "review": review,
                    "recommend": recommend
                }

                reviews.append(data)
        else:
            continue

    print(f"Total Posts for {airline}: {len(reviews)}")
    return reviews

async def main_scraper():
    airlines = ['southwest-airlines', 'american-airlines', 'delta-air-lines', 'hawaiian-airlines', 'frontier-airlines']
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Safari/537.36"
    }
    url = "https://www.airlinequality.com/airline-reviews/"

    async with aiohttp.ClientSession() as session:
        tasks = [scrape_reviews(airline, session, url, headers) for airline in airlines]
        all_reviews = await asyncio.gather(*tasks)

    all_reviews_flat = [review for airline_reviews in all_reviews for review in airline_reviews]
    print(f"Total Posts Scraped: {len(all_reviews_flat)}")
    return all_reviews_flat

all_reviews = asyncio.run(main_scraper())

df = pd.DataFrame(all_reviews) # uncomment for csv
df.to_csv("all_skytrax_reviews.csv", index=False) # uncomment for csv

# json.dump(all_reviews, open("all_skytrax_reviews.json", "w"), indent=4) # uncomment for json
