import time
import random
import os
import json
from bs4 import BeautifulSoup
from datetime import datetime
from selenium import webdriver
# from selenium.webdriver.safari.service import Service
# from selenium.webdriver.safari.options import Options
from json.decoder import JSONDecodeError
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from datetime import datetime
import time
import json
import os
from datetime import datetime, timedelta

subreddits = ['https://www.reddit.com/r/HawaiianAirlines/top/?t=all', 'https://www.reddit.com/r/HawaiianAirlines/top/?t=year', 'https://www.reddit.com/r/HawaiianAirlines/hot/']
# subreddits = ['https://www.reddit.com/r/HawaiianAirlines/top/?t=all']

class ScrapeReddit:
    def __init__(self, headless=False):
        """Initialize the Reddit scraper with Chrome WebDriver"""
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless")  # Run in headless mode
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-notifications")
        self.chrome_options.add_argument("--disable-popup-blocking")
        # self.chrome_options.add_argument("--start-maximized")
        
        # Add user agent to avoid detection
        self.chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        # Initialize WebDriver
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=self.chrome_options
        )
        self.wait = WebDriverWait(self.driver, 10)
        self.postids = []

    def save_id_to_json(self, data, batch_number):
        """Save the post IDs to a JSON file in the local computer."""
        directory = 'ids/HawaiianAirlines'
        if not os.path.exists(directory):
            os.makedirs(directory)

        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # Filename will include the batch number
        filename = f'batch_{batch_number}_{timestamp}.json'
        file_path = os.path.join(directory, filename)

        # Save the data to JSON
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print(f"Batch {batch_number} successfully saved to {file_path}")

    def lazy_scroll(self, scroll_duration_minutes=30, batch_size=100):
        """Scroll lazily to the bottom of the page, collect post links, and save every 100 posts."""
        SCROLL_PAUSE_TIME = 5  # Time between scrolls to let content load
        post_links = []  # Store links in a list
        batch_number = 1  # To keep track of batches

        # Track the start time
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=scroll_duration_minutes)
        print(f"Scrolling will stop at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        current_height = self.driver.execute_script('return document.body.scrollHeight')

        count = 0
        while True:
            # Scroll down to the bottom of the page
            self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(SCROLL_PAUSE_TIME)

            # Get the new page height after scrolling
            new_height = self.driver.execute_script('return document.body.scrollHeight')

            # If the page height hasn't changed, we've reached the bottom, so stop
            if new_height == current_height:
                print("Reached the end of the page. No more content to load.")
                # break
                count += 1
                print('Pausing for 10 seconds...')
                time.sleep(10)
                if count > 3:

                    new_post_links = parser.find_all('a', {'slot': 'full-post-link'})

                    for post_link in new_post_links:
                        post_url = post_link.get('href')
                        post_id = post_url.split('/')[-3]
                        print(f"{post_id}")
                        if post_id not in post_links:
                            post_links.append(post_id)

                    # save id to json
                    self.save_id_to_json(post_links, batch_number)  # Save batch to JSON
                    post_links.clear()
                    break
            else:
                count = 0

            # Get the page source and parse it
            html = self.driver.page_source
            parser = BeautifulSoup(html, 'html.parser')

            # Find all post links
            new_post_links = parser.find_all('a', {'slot': 'full-post-link'})

            # Add new links to the list, avoiding duplicates
            for post_link in new_post_links:
                post_url = post_link.get('href')
                post_id = post_url.split('/')[-3]
                print(f"{post_id}")
                if post_id not in post_links:
                    post_links.append(post_id)

            # If we have collected 100 posts, save to JSON and clear the list
            if len(post_links) >= batch_size:
                print(f"Saving batch {batch_number} with {len(post_links)} posts...")
                self.save_id_to_json(post_links, batch_number)  # Save batch to JSON
                post_links.clear()  # Clear the array after saving
                batch_number += 1  # Increment batch number

            # Check if 30 minutes have passed
            current_time = datetime.now()
            if current_time >= end_time:
                # Save any remaining posts if time runs out before the batch completes
                if post_links:
                    print(f"Saving final batch {batch_number} with {len(post_links)} posts...")
                    self.save_id_to_json(post_links, batch_number)  # Save final batch
                print(f"Reached time limit of {scroll_duration_minutes} minutes. Stopping scrolling.")
                break

            # Update the current height for the next scroll
            current_height = new_height

        return # Convert the set back to a list and return

    def get_data(self, postid):
        """Fetch post data from Reddit based on the post ID."""
        base_url = "https://reddit.com/r/HawaiianAirlines/comments/"
        url = base_url + postid + ".json"
        self.driver.get(url)
        # self.driver.maximize_window()
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.find('body').get_text()
        time.sleep(3)
        return text
    
    def load_post_ids_from_json(self):
        """Load all post IDs from the saved JSON files in the 'data' directory."""
        directory = 'ids/HawaiianAirlines'
        all_post_ids = []  # This will store all the post IDs from all batches
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist, skipping loading of post IDs.")
            return
        # Iterate over all the files in the 'data' directory
        for filename in os.listdir(directory):
            if filename.endswith('.json'):  # Only process JSON files
                file_path = os.path.join(directory, filename)
                print(f"Loading post IDs from {file_path}")

                # Load the JSON data from the file
                with open(file_path, 'r') as json_file:
                    post_ids = json.load(json_file)
                    all_post_ids.extend(post_ids)  # Add the post IDs to the list

        self.postids = list(set(all_post_ids))  # Remove duplicates by converting to a set and back to a list
        print(f"Total unique post IDs loaded: {len(self.postids)}")


    def get_post_details(self):
        """Get details of posts from stored post IDs."""
        jsons = []
        count = 1
        if not self.postids:
            self.load_post_ids_from_json()
            print("No post IDs found. Please run get_posts() first.")
            return
        
        for postid in self.postids:
            print(postid, count)
            text = self.get_data(postid)
            jsons.append(text)
            time.sleep(random.randint(1, 10))
            count += 1
        
        self.jsons = jsons
        return jsons

    def get_posts(self):
        """Fetch posts from the subreddits and collect post IDs."""
        for link in subreddits:
            self.driver.get(link)
            time.sleep(5)
            post_links = self.lazy_scroll(scroll_duration_minutes=20)

    @staticmethod
    def get_post_info(json_data):
        """
        Gets the post body, all comments and their replies, 
        the user IDs of the post, comments, and replies, 
        and the timestamps of the post, comments, and replies 
        from the JSON data.
        """

        if not json_data or 'data' not in json_data[0] or 'children' not in json_data[0]['data'] or len(json_data[0]['data']['children']) == 0:
            # Skip processing if json_data structure is invalid
            print("Invalid post data, skipping...")
            return None

        date_limit = datetime(2023, 8, 1)
        post = json_data[0]['data']['children'][0]['data']
        if not post:
            print("No post data, skipping...")
            return None
        post_body = post['title']
        post_user = post['author']
        post_id = post['id']
        post_time = post['created_utc']
        self_text = post['selftext']
        comments = json_data[1]['data']['children']
        comments_list = []

        for (comment, idx) in zip(comments, range(len(comments))):
            if 'body' in comment['data'] and 'author' in comment['data'] and 'created_utc' in comment['data']:

                comment_id = comment['data']['id']
                comment_body = comment['data']['body']
                comment_user = comment['data']['author']
                comment_time = comment['data']['created_utc']
                comment_date = datetime.fromtimestamp(comment_time)
                comment_date_str = comment_date.strftime('%Y-%m-%d %H:%M:%S')
                score = comment['data']['score']
                if comment_body != '[removed]' and comment_body != '[deleted]':
                    comments_list.append({'content': comment_body, 
                                        'id':comment_id,
                                        'username': comment_user,
                                        'date': comment_date_str,
                                        'score': score,
                                        'post_id': post_id,
                                        'parent_id': 'na'})
                comment_replies = []

            # append reply to the comment to which it belongs

            # Check if 'replies' exists and is not empty or [removed]
            if 'replies' in comment['data'] and comment['data']['replies'] != '' and comment['data']['replies'] != '[removed]':
                print(f"Found replies for comment: {comment['data']['replies']}")  # Print the value of replies

                # Make sure the 'data' and 'children' keys are present in the replies structure
                if 'data' in comment['data']['replies'] and 'children' in comment['data']['replies']['data']:
                    replies = comment['data']['replies']['data']['children'] 

                    for reply in replies:
                        # Ensure that the required keys are present in each reply
                        if all(key in reply['data'] for key in ['body', 'author', 'created_utc', 'parent_id', 'link_id', 'score']):
                            
                            reply_body = reply['data']['body']
                            if reply_body != '[removed]' and reply_body != '[deleted]':
                                reply_user = reply['data']['author']
                                reply_time = reply['data']['created_utc']
                                reply_date = datetime.fromtimestamp(reply_time)
                                reply_date_str = reply_date.strftime('%Y-%m-%d %H:%M:%S')
                                parent_id = reply['data']['parent_id']
                                score = reply['data']['score']
                                subreddit_id = reply['data']['link_id']
                                id = reply['data']['id']
                                comment_replies.append({'content': reply_body, 'username': reply_user, 'date': reply_date_str, 'parent_id': parent_id, 'post_id':subreddit_id, 'score': score, 'id': id})
            if len(comments_list) > 0:
                comments_list[-1]['replies'] = comment_replies

        # Convert the timestamp to a datetime object
        post_date = datetime.fromtimestamp(post_time)

        # Format the datetime object to a readable string
        formatted_date = post_date.strftime('%Y-%m-%d %H:%M:%S')

        num_comments = post['num_comments']
        link = post['permalink']

        return {
            'post_body': post_body,
            'post_user': post_user,
            'date_posted': formatted_date,
            'num_comments': num_comments,
            'link': link,
            'score': post['score'],
            'self_text': self_text,
            "comments": comments_list,
        }

    def destroy(self):
        """Close the WebDriver."""
        self.driver.close()


# Running the scraper
reddit = ScrapeReddit()
reddit.get_posts()
reddit.load_post_ids_from_json()

data = reddit.get_post_details()

# Process the JSON data
res = []
for i in range(len(data)):
    try:
        parsed_json = json.loads(data[i])
        info = ScrapeReddit.get_post_info(parsed_json)
        if info is None:
            print("Post skipped due to date limit.")
            continue  # Skip to the next iteration
        res.append(info)
    except JSONDecodeError as e:
        print(e)
        continue

def save_to_json(data, subreddit):
    """Save the scraped data to a JSON file in the current working directory."""
    # Get the current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    # Create the timestamp for the filename
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Construct the path using the current working directory and 'data/HawaiianAirlines'
    directory_path = os.path.join(current_dir, 'data', subreddit)
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} not found, creating it now...")
        os.makedirs(directory_path, exist_ok=True)
    else:
        print(f"Directory {directory_path} exists.")

    # Construct the full path for the file
    filename = os.path.join(directory_path, f'{timestamp}_HawaiianAirlines.json')
    print(f"Saving file to: {filename}")

    # Save the data to the JSON file
    with open(filename, 'w') as f:
        json.dump(data, f)


save_to_json(res, 'HawaiianAirlines')
