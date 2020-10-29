import praw
from psaw import PushshiftAPI
import datetime as dt
import pandas as pd

'''
r = praw.Reddit(
    client_id="fdbTZrKPiwaOJg",
    client_secret="jcUoKw0Eol4oWaCcBJ8JvzqUi5U",
    user_agent="python_script:425ML_App:v1.0.0 (by /u/425rhett)",
)
'''
api = PushshiftAPI()

start_epoch = int(dt.datetime(2020, 10, 26).timestamp())
gen = api.search_submissions(after=start_epoch, subreddit='politics',
                            filter=['author', ] limit=250)

df = pd.DataFrame([thing.d_ for thing in gen])
# Get a sample of the data 
sample = df.sample(frac = 0.2)

# write to sheets of an excel notebook
with pd.ExcelWriter('politics_data.xlsx') as writer:
    df.to_excel(writer, sheet_name = 'data')
    sample.to_excel(writer, sheet_name='sample')

