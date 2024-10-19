from datetime import datetime
from dotenv import load_dotenv
import os
import pandas as pd
import praw

load_dotenv()

airline_subreddits = ['SouthwestAirlines', 'Southwest_Airlines', 'AmericanAir', 'DeltaAirlines', 'HawaiianAirlines', 'frontierairlines']

reddit = praw.Reddit(
    client_id=os.environ.get('REDDIT_ID'),
    client_secret=os.environ.get('REDDIT_SECRET'),
    user_agent=os.environ.get('REDDIT_USER'),
    username=os.environ.get('REDDIT_USERNAME'),
    password=os.environ.get('REDDIT_PASS'),
    project_name=os.environ.get('PROJ_NAME')
)

posts = []
comments = []

for subreddit in airline_subreddits:
    for post in reddit.subreddit(subreddit).new():
        posts.append({
            'id':str(post.id),
            'date': str(datetime.fromtimestamp(post.created_utc)),
            'title':str(post.title),
            'content':str(post.selftext),
            'username':str(post.author),
            'commentCount':int(post.num_comments),
            'score':int(post.score),
            'subreddit':str(post.subreddit)
        })
        if post.num_comments > 0:
            submission = reddit.submission(id=post.id)
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                if comment.author is None or str(comment.author) == "AutoModerator":
                    continue
                comments.append({
                    'id': str(comment.id),
                    'date': str(datetime.fromtimestamp(comment.created_utc)),
                    'content': str(comment.body),
                    'username': str(comment.author.name),
                    'score': int(comment.score),
                    'post_id': str(post.id),
                    'parent_id': str(comment.parent_id),
                })

posts_df = pd.DataFrame(posts)
posts_df.to_csv('reddit_posts.csv')

comments_df = pd.DataFrame(comments)
comments_df.to_csv('reddit_comments.csv')

print(f'Total Posts: {len(posts)}')
print(f'Total Comments: {len(comments)}')