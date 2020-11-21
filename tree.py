import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn

# Using only features with numeric values
features = ['author_total_karma', 'author_has_verified_email',
'author_is_employee', 'author_is_mod', 'author_link_karma', 'num_comments', 'score','upvote_ratio']

df = pd.read_csv("reddit_data.csv")
df = df.replace({True: 1, False: 0})
X = df.loc[:, features].values
# Separating out the target
y = df.loc[:, ["reliable"]].values
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=0)

# Grid Search
tuned_parameters = [
    {
        'criterion': ['entropy'],
        'max_depth': [3, 7, 10]
    },
    {
        'criterion': ['gini'],
        'max_depth': [3, 5, 7, 9, 11, 13, 15]
    }
]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        tree.DecisionTreeClassifier(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    # Setting values for predictions later
    if score == "precision":
        precision_depth = clf.best_params_["max_depth"]
        precision_criterion = clf.best_params_["criterion"]
    else:
        recall_depth = clf.best_params_["max_depth"]
        recall_criterion = clf.best_params_["criterion"]

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


# Create predictions and tree plots
# Using depth of 7 as that was the most common one when training from multiple datasets
precision_clf = tree.DecisionTreeClassifier(max_depth=7, criterion=precision_criterion)
precision_clf = precision_clf.fit(X_train, y_train)
tree.plot_tree(precision_clf, feature_names=features, class_names=["Reliable", "Non-reliable"])
plt.show()

recall_clf = tree.DecisionTreeClassifier(max_depth=7, criterion=recall_criterion)
recall_clf = recall_clf.fit(X_train, y_train)
tree.plot_tree(recall_clf, feature_names=features, class_names=["Reliable", "Non-reliable"])
plt.show()

precision_predictions = precision_clf.predict(X_test)
recall_predictions = recall_clf.predict(X_test)

# Make confusion matrices from predicted data
f, axes = plt.subplots(1, 2)

tn, fp, fn, tp = confusion_matrix(y_test, precision_predictions).ravel()
array = [[tn, fp],
         [fn, tp]]

df = pd.DataFrame(array, range(2), range(2))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df, fmt="d", annot=True, annot_kws={"size": 16}, vmin=0, vmax=15, ax=axes[0], cmap="mako") # font size
axes[0].set_title("Precision Optimized Decision Tree")
axes[0].set(xlabel='Precited Values', ylabel='Actual Values')
print("Accuracy for precision tree:")
print(accuracy_score(y_test, precision_predictions))

tn, fp, fn, tp = confusion_matrix(y_test, recall_predictions).ravel()
array = [[tn, fp],
         [fn, tp]]

df2 = pd.DataFrame(array, range(2), range(2))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df2, fmt="d", annot=True, annot_kws={"size": 16}, vmin=0, vmax=15, ax=axes[1], cmap="mako") # font size
axes[1].set_title("Recall Optimized Decision Tree")
axes[1].set(xlabel='Precited Values', ylabel='Actual Values')
print("Accuracy for recall tree:")
print(accuracy_score(y_test, recall_predictions))

plt.show()








