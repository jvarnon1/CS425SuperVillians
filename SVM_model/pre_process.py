# Read our labeled data from Excel into a pandas data frame

import pandas as pd
data = pd.read_excel('SVM_model\politics_data_labeled.xlsx', index_col=0, sheet_name='data', header = 0)

# elminate columns that are completely useless
# remove redundant features
data = data.drop(columns = ['author', 'submission_id', 'submission_location'])
# change bool values to integer values
data['author_has_verified_email'] = data['author_has_verified_email'].astype(int)
data['author_is_mod'] = data['author_is_mod'].astype(int)
# eliminate empty or nonuniqe feature values.
for column in data:
    feature = data[column]
    #print( column, ": ", feature.nunique())
    if (feature.nunique() < 2):
        data = data.drop(columns=column)
    #else:
     #   print(column, ": ", feature.nunique())
# for now drop the object type column features
data = data.drop(columns = ['author_id', 'submission_title', 'submission_link', 'link_flair_text'])
data.info()
