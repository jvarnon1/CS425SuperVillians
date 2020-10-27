import praw
from psaw import PushshiftAPI
import datetime as dt
import csv
from pathlib import Path

# Create columns in csv
def createNewCSV():
    with open('politics_data.csv', 'w', newline='') as csvfile:
        file_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Writing column names
        file_writer.writerow(['author', 'author_id', 'author_total_karma', 'author_created_utc',
                              'author_has_verified_email', 'author_is_employee', 'author_is_mod', 'author_link_karma',

                              'submission_title', 'submission_id', 'submission_location', 'submission_link',
                              'created_utc', 'distinguished', 'edited', 'is_original_content',
                              'is_self', 'gilded', 'link_flair_text', 'locked',
                              'num_comments', 'over_18', 'score', 'selftext',
                              'spoiler', 'stickied', 'upvote_ratio',
                              'clicked'])


# Populate data in existing csv
def populateCSV(gen_list):
    with open('politics_data.csv', 'a', newline='') as csvfile:
        file_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for post in gen_list:
            redditor = post.author
            try:
                if redditor is None or redditor.created_utc is None:
                    continue
                if hasattr(post, "poll_data"):
                    continue

                print("Writing csv data for redditor: " + redditor.name)
                file_writer.writerow([redditor.name, redditor.id, redditor.total_karma, redditor.created_utc,
                                    redditor.has_verified_email, redditor.is_employee, redditor.is_mod, redditor.link_karma,
                                    post.title, post.id, post.permalink, post.url,
                                    post.created_utc, post.distinguished, post.edited, post.is_original_content,
                                    post.is_self, post.gilded, post.link_flair_text, post.locked,
                                    post.num_comments, post.over_18, post.score, post.selftext,
                                    post.spoiler, post.stickied, post.upvote_ratio,
                                    post.clicked])
            except:
                print("Error writing data for redditor: " + str(post.author) + ". Continuing.")


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
                                  limit=50)
gen_list = list(gen)

file = Path("politics_data.csv")
if file.is_file():
    populateCSV(gen_list)
else:
    createNewCSV()
    populateCSV(gen_list)
