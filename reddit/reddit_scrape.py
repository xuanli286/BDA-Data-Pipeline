from datetime import datetime, timedelta
from dotenv import load_dotenv
import boto3
import os
import pandas as pd
import praw

load_dotenv()

def lambda_handler(event, context):
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

    posts_df = pd.DataFrame(posts)
    posts_filename = f'reddit_posts_{end_date.strftime("%Y-%m-%d")}.csv'
    posts_df.to_csv(posts_filename)

    comments_df = pd.DataFrame(comments)
    comments_filename = f'reddit_comments_{end_date.strftime("%Y-%m-%d")}.csv'
    comments_df.to_csv(comments_filename)

    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(posts_filename, 'is459-project-data', f'reddit/{posts_filename}')
        s3_client.upload_file(comments_filename, 'is459-project-data', f'reddit/{comments_filename}')
        print("Files uploaded to S3 successfully")
    except Exception as e:
        print("Error uploading to S3: ", e)

    os.remove(posts_filename)
    os.remove(comments_filename)

    print(f'Total Posts: {len(posts)}')
    print(f'Total Comments: {len(comments)}')