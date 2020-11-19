import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

features = ['author_total_karma', 'author_has_verified_email',
'author_is_employee', 'author_is_mod', 'author_link_karma', 'num_comments', 'score','upvote_ratio']
df = pd.read_csv ("reddit_data.csv")
df = df.replace({True: 1, False: 0})
#print(df)
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['reliable']].values

x = StandardScaler().fit_transform(x)
pca = PCA(n_components =5)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5'])

finalDf = pd.concat([principalDf, df[['reliable']]], axis = 1)

np.set_printoptions(precision=3)
print("How much of our variance is explained?")
print(pca.explained_variance_ratio_)
print()
print() 

print("Which features matter most?")
o = ["%.3f" % elem for elem in abs(pca.components_[0])]
o1 = ["%.3f" % elem for elem in abs(pca.components_[1])]
o2 = ["%.3f" % elem for elem in abs(pca.components_[2])]
o3 = ["%.3f" % elem for elem in abs(pca.components_[3])]
o4 = ["%.3f" % elem for elem in abs(pca.components_[4])]
print(o)
print(o1)
print(o2)
print(o3)
print(o4)
#print(abs(pca.components_))