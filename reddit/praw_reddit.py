import praw
import pandas as pd
import os
from datetime import datetime
import time

client_id = "VHO7-kjPqse7AvBFNCHjlw"
client_secret = "llFFlyD4KvFHdZQPidYXLLOOXOb1-w"
user_agent = "IS434_MyApp"
username = "Specific-Net-1615"
password = "Password2023!"

# username=  "lactrodectu5"
# password= "Sianya!234"

# client_id=  "5Tf-mtY0saAWQ_mxKIx9Ow"
# client_secret = "6sOLXi45fG3sa1PCg7z1cWWVO8gQtg"
# user_agent = "IS434_Lab4"


# Initialize the Reddit client
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    username=username,
    password=password,
)


def find_and_scrape_posts(subreddit_name, keyword, start_date, end_date):
    try:
        # Reddit API initialization here (ensure you've initialized the reddit instance)
        subreddit = reddit.subreddit(subreddit_name)

        # Convert date strings to datetime objects if passed as strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Search for posts related to the keyword
        posts_data = []
        total_posts = 0
        saved_posts = 0

        print(f"Searching for keyword: '{keyword}' in subreddit: {subreddit_name}")
        
        for submission in subreddit.search(keyword, limit=1000):  # Adjust limit as needed
            total_posts += 1  # Count total posts searched
            
            # Convert post created time to datetime
            post_date = datetime.fromtimestamp(submission.created_utc)

            # Skip posts outside the date range
            if post_date < start_date or post_date > end_date:
                continue

            # Collect post data
            post_data = {
                'post_body': submission.title,
                'self_text': submission.selftext,
                'score': submission.score,
                'link': submission.permalink,
                'date_posted': post_date.strftime('%Y-%m-%d %H:%M:%S'),
                'post_user': str(submission.author),
                'num_comments': submission.num_comments
            }
            posts_data.append(post_data)
            saved_posts += 1  # Count posts that were saved after filtering

            print(f"Found post: {submission.title} (Date: {post_date})")

        # Save the posts data to a DataFrame and export to CSV
        if posts_data:
            posts_df = pd.DataFrame(posts_data)

            # Create 'posts' folder if it doesn't exist
            directory = "posts"
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directory {directory} created.")

            # Create filename and save to 'posts' folder
            posts_filename = os.path.join(directory, f"{subreddit_name}_{keyword}_posts_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.csv")
            posts_df.to_csv(posts_filename, index=False)
            print(f"Posts saved to {posts_filename}")
        else:
            print(f"No posts found for keyword '{keyword}' in subreddit '{subreddit_name}' during the specified date range.")

        # Print out the number of posts searched and saved
        print(f"Total posts searched for keyword '{keyword}': {total_posts}")
        print(f"Total posts saved for keyword '{keyword}': {saved_posts}")
    
    except Exception as e:
        # Log errors to a file
        with open('scraping_errors.log', 'a') as error_log:
            error_log.write(f"Error for subreddit '{subreddit_name}' with keyword '{keyword}': {str(e)}\n")
        print(f"An error occurred: {str(e)}")
    
    # Sleep to avoid being rate-limited
    sleep_time = 5
    print(f"Sleeping for {sleep_time} seconds to avoid rate-limiting...")
    time.sleep(sleep_time)

# Define keywords and date range
# keywords = [
#     'McDonald\'s Israel war', 'McDonald\'s price', 'McDonald\'s boycott', 
#     'McDonald\'s $5 meal', 'McDonald\'s conflict', 'McDonald\'s expensive',
#     'McDonald\'s supports Israel', 'McDonald\'s Palestine', 'Stop eating McDonald\'s',
#     'McDonald\'s protest', 'Boycott McDonald\'s', 'McDonald\'s price hike', 
#     'McDonald\'s too expensive', 'McDonald\'s promotions', 'McDonald\'s war controversy'
# ]

keywords= [
    'price', 'war', 'israel' 
]
start_date = '2023-08-01'
end_date = '2024-10-22'

# Loop through both McDonalds and McLounge subreddits for all keywords
for keyword in keywords:
    find_and_scrape_posts('MacDonalds', keyword, start_date, end_date)
    # find_and_scrape_posts('McLounge', keyword, start_date, end_date)