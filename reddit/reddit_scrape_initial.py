from datetime import datetime
from dotenv import load_dotenv
import boto3
import json
import os
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

s3 = boto3.client('s3')
try:
    s3.put_object(
        Bucket='is459-project-data', 
        Key='reddit/reddit_posts.json',
        Body=json.dumps(posts),
        ContentType='application/json'
    )
    s3.put_object(
        Bucket='is459-project-data', 
        Key=f'reddit/reddit_comments.json',
        Body=json.dumps(comments),
        ContentType='application/json'
    )
    print("Files uploaded to S3 successfully")
except Exception as e:
    print("Error uploading to S3: ", e)

print(f'Total Posts: {len(posts)}')
print(f'Total Comments: {len(comments)}')