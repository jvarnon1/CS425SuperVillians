import praw
from psaw import PushshiftAPI
import datetime as dt
import csv
from pathlib import Path
import pandas as pd

# Create columns in csv
def createNewpanda(rows):
    df = pd.DataFrame(rows, columns=['author', 'author_id', 'author_total_karma', 'author_created_utc',
                       'author_has_verified_email', 'author_is_employee', 'author_is_mod', 'author_link_karma',
                        'submission_title', 'submission_id', 'submission_location', 'submission_link',
                        'created_utc', 'distinguished', 'edited', 'is_original_content',
                        'is_self', 'gilded', 'link_flair_text', 'locked',
                        'num_comments', 'over_18', 'score', 'selftext',
                        'spoiler', 'stickied', 'upvote_ratio',
                        'clicked'])
    return df

# Populate data in existing csv
def populatepanda(gen_list, rows):
    for post in gen_list:
        redditor = post.author
        try:
            if redditor is None or redditor.created_utc is None:
                continue
            if hasattr(post, "poll_data"):
                continue
            print("Writing panda data for redditor: " + redditor.name)
            rows.append([redditor.name, redditor.id, redditor.total_karma, redditor.created_utc,
                                redditor.has_verified_email, redditor.is_employee, redditor.is_mod, redditor.link_karma,
                                post.title, post.id, post.permalink, post.url,
                                post.created_utc, post.distinguished, post.edited, post.is_original_content,
                                post.is_self, post.gilded, post.link_flair_text, post.locked,
                                post.num_comments, post.over_18, post.score, post.selftext,
                                post.spoiler, post.stickied, post.upvote_ratio,
                                post.clicked])
        except:
            print("Error writing data for redditor: " + str(post.author) + ". Continuing.")
    return rows

# Initialize Reddit access with client data
r = praw.Reddit(
    client_id="fdbTZrKPiwaOJg",
    client_secret="jcUoKw0Eol4oWaCcBJ8JvzqUi5U",
    user_agent="python_script:425ML_App:v1.0.0 (by /u/425rhett)",
)
api = PushshiftAPI(r)

start_epoch = int(dt.datetime(2020, 10, 26).timestamp())

gen = api.search_submissions(after=start_epoch,
                                  subreddit='politics',
                                  limit=250)
gen_list = list(gen)
rows = []
rows = populatepanda(gen_list, rows)
df = createNewpanda(rows)
# Get a sample of the data 
sample = df.sample(frac = 0.2)

# write to sheets of an excel notebook
with pd.ExcelWriter('politics_data.xlsx') as writer:
    df.to_excel(writer, sheet_name = 'data')
    sample.to_excel(writer, sheet_name='sample')




