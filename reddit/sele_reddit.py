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
import boto3


class ScrapeReddit:
    def __init__(self, subreddit_name, headless=False):
        """Initialize the Reddit scraper with Chrome WebDriver"""
        self.subreddit_name = subreddit_name
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
        # options = Options()
        # self.driver = webdriver.Safari(service=Service(executable_path='/usr/bin/safaridriver'), options=options)
        self.postids = []


    def save_id_to_json(self, data, batch_number):
        """Save the post IDs to a JSON file in the local computer."""
        directory = 'ids/'+ self.subreddit_name
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
        base_url = "https://reddit.com/r/{}/comments/".format(self.subreddit_name)
        # print(base_url)
        url = base_url + postid + ".json"
        print(url)
        self.driver.get(url)
        # self.driver.maximize_window()
        html = self.driver.page_source
        text = ""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            body = soup.find('body')
            if body:
                text = body.get_text()
            else:
                print(f"No 'body' found in HTML for post ID: {postid}")
                # Save the HTML content for inspection
                with open(f"html_error_{postid}.html", 'w', encoding='utf-8') as html_file:
                    html_file.write(html)
                with open('errors.json', 'a') as file:
                    file.write(json.dumps({"error": "No body found", "postid": postid}) + "\n")
        except Exception as e:
            print(e)
            print(f"Error fetching data for post ID: {postid}")
            with open('errors.json', 'a') as file:
                file.write(json.dumps({"error": str(e), "postid": postid}) + "\n")

        time.sleep(3)
        return text
    
    def load_post_ids_from_json(self):
        """Load all post IDs from the saved JSON files in the 'data' directory."""
        directory = 'ids/' + self.subreddit_name
        all_post_ids = []  # This will store all the post IDs from all batches
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist, skipping loading of post IDs.")
            return
        
        print(directory)
        # Iterate over all the files in the 'data' directory
        for filename in os.listdir(directory):
            print(filename)
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
            return []
        
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
            # self.driver.maximize_window()
            time.sleep(5)
            # post_links = self.lazy_scroll(scroll_duration_minutes=20)
            # print(f"Total unique post links collected: {len(post_links)}")

            # # print(len(post_links))
            # count = 1

            # for post_link in post_links:
            #     print(post_link)
            #     post_id = post_link.split('/')[-3]
            #     print(f"{count} - {post_id}")
            #     count += 1
            #     if post_id not in self.postids:
            #         self.postids.append(post_id)

    @staticmethod
    def get_post_info(json_data):
        """
        Gets the post body, all comments and their replies, 
        the user IDs of the post, comments, and replies, 
        and the timestamps of the post, comments, and replies 
        from the JSON data.
        """

        if not json_data or 'data' not in json_data[0] or 'children' not in json_data[0]['data'] or len(json_data[0]['data']['children']) == 0:

            if 'data' not in json_data[1] or 'children' not in json_data[1]['data']:
                
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
        post_time = post['created_utc']
        self_text = post['selftext']

        comments = json_data[1]['data']['children']
        comments_list = []
        for (comment, idx) in zip(comments, range(len(comments))):
            if 'body' in comment['data'] and 'author' in comment['data'] and 'created_utc' in comment['data']:

                comment_body = comment['data']['body']
                comment_user = comment['data']['author']
                comment_time = comment['data']['created_utc']
                comments_list.append({'body': comment_body,
                                    'user': comment_user,
                                    'time': comment_time})
                comment_replies = []

                # append reply to the comment to which it belongs

                if comment['data']['replies'] != '':
                    replies = comment['data']['replies']['data']['children']
                    for reply in replies:
                        if all(key in reply['data'] for key in ['body', 'author', 'created_utc']):
                        # if reply['data']['body'] and reply['data']['author'] and reply['data']['created_utc']:
                            reply_body = reply['data']['body']
                            reply_user = reply['data']['author']
                            reply_time = reply['data']['created_utc']
                            comment_replies.append({'body': reply_body,
                                    'user': reply_user, 'time': reply_time})
                comments_list[idx]['replies'] = comment_replies

        # Convert the timestamp to a datetime object
        post_date = datetime.fromtimestamp(post_time)

        # Format the datetime object to a readable string
        formatted_date = post_date.strftime('%Y-%m-%d %H:%M:%S')

        # if post_date < date_limit:
        #     print(f"Post from {post_body} is earlier than August 2023. Skipping.")
        #     return None

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
    
    def log_error(self, postid, error_message):
        """Log any errors during scraping."""
        error_log_file = 'error_log.json'

        # If the error log doesn't exist, create it
        if not os.path.exists(error_log_file):
            with open(error_log_file, 'w') as file:
                json.dump([], file)

        # Append error to the log
        with open(error_log_file, 'r') as file:
            error_log = json.load(file)

        error_log.append({
            'postid': postid,
            'error': str(error_message),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        with open(error_log_file, 'w') as file:
            json.dump(error_log, file, indent=4)
        print(f"Logged error for post ID {postid}: {error_message}")

    def destroy(self):
        """Close the WebDriver."""
        self.driver.close()


subreddit_name = 'SouthwestAirlines'
subreddits = ['https://www.reddit.com/r/{}/top/?t=all'.format(subreddit_name),
            'https://www.reddit.com/r/{}/top/?t=year'.format(subreddit_name),
            'https://www.reddit.com/r/{}/hot/'.format(subreddit_name)]

# def upload_to_s3(file_path, file_name):
#     s3_key = f"{s3_folder_path}{file_name}"
#     try:
#         s3.upload_file(file_path, bucket_name, s3_key)
#         print(f"Uploaded {file_name} to {s3_key}")
#     except Exception as e:
#         print(f"Error uploading {file_name} to S3: {e}")

@staticmethod
def save_to_json(self, data, subreddit, batch_number=None):
        """Save the scraped data to a JSON file immediately."""
        current_dir = os.getcwd()
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # Construct the directory path
        directory_path = os.path.join(current_dir, 'data', subreddit)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Use a different filename for batch data
        if batch_number:
            filename = f'batch_{batch_number}_{timestamp}.json'
        else:
            filename = f'{timestamp}.json'

        file_path = os.path.join(directory_path, filename)
        print(f"Saving data to: {file_path}")

        with open(file_path, 'w') as f:
            json.dump(data, f)

reddit = ScrapeReddit(subreddit_name=subreddit_name)
reddit.get_posts()
reddit.load_post_ids_from_json()

data = reddit.get_post_details()
print(data)
# Process the JSON data
res = []
for i, postid in enumerate(reddit.postids):
    try:
        text = reddit.get_data(postid)
        parsed_json = json.loads(text)
        info = ScrapeReddit.get_post_info(parsed_json)

        if info is None:
            print(f"Post {postid} skipped due to date limit or missing data.")
            continue

        res.append(info)

        # Save each post's details immediately after processing
        save_to_json([info], 'SouthwestAirlines')

    except JSONDecodeError as e:
        reddit.log_error(postid, f"JSON decoding error: {e}")
    except Exception as e:
        reddit.log_error(postid, f"Unexpected error: {e}")


save_to_json(res, subreddit_name)

# save_to_json(res, 'Southwest_Airlines')
# save_to_json(res, 'AmericanAir')


# airline_subreddits = ['SouthwestAirlines', 'Southwest_Airlines', 'AmericanAir', 'DeltaAirlines', 'HawaiianAirlines', 'frontierairlines']
