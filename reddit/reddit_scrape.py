from datetime import datetime, timedelta
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
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=7)

for subreddit in airline_subreddits:
    for post in reddit.subreddit(subreddit).new():
        try:
            if datetime.fromtimestamp(post.created_utc) < start_date or datetime.fromtimestamp(post.created_utc) > end_date:
                continue
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
        except Exception as e:
            print("Error: " + str(post.id))
            print(e)
            continue

s3 = boto3.client('s3')
try:
    s3.put_object(
        Bucket='is459-project-data', 
        Key=f'reddit/new_posts/reddit_posts_{end_date.strftime("%Y-%m-%d")}.json',
        Body=json.dumps(posts),
        ContentType='application/json'
    )
    s3.put_object(
        Bucket='is459-project-data', 
        Key=f'reddit/new_comments/reddit_comments_{end_date.strftime("%Y-%m-%d")}.json',
        Body=json.dumps(comments),
        ContentType='application/json'
    )
    print("Files uploaded to S3 successfully")
except Exception as e:
    print("Error uploading to S3: ", e)

print(f'Total Posts: {len(posts)}')
print(f'Total Comments: {len(comments)}')