from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import os
import praw

load_dotenv()

query = 'USA AND flight delay'

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
start_date = datetime(2023, 1, 1) 
end_date = datetime.utcnow()

for post in reddit.subreddit("all").search(query=query, sort="new", time_filter="all"):
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
            
with open (f"./reddit_posts_dump.json", "w") as f:
    json.dump(posts, f, ensure_ascii=False)
with open (f"./reddit_comments_dump.json", "w") as f:
    json.dump(comments, f, ensure_ascii=False)

print(f'Total Posts: {len(posts)}')
print(f'Total Comments: {len(comments)}')