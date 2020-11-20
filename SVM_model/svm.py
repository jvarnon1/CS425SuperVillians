import numpy as np
import pandas as pd
import pylab as pl
from sklearn import svm, datasets, preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from pre_process import data

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
#data.info()
#print(data.head())
dataset= pd.DataFrame(data=data[:].values,columns=None, copy=True)
#print(dataset.head())
# separate target from features
X = dataset.loc[:,0:8]
Y = dataset.loc[:,9]
#print(X.head())
# standardize the data using sklearn
X = preprocessing.scale(X)

# Use Sklearn to get splits in our data for training and testing.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=0)
y_train_converted = y_train.values.ravel() #ensuring numerical calculations

# Now, we create several instances of SVCs that utilize varying kernels.
# We're not normalizing our data here because we want to plot the support vectors.
svc     = svm.SVC(kernel='linear').fit(x_train, y_train)
poly_svc_one = svm.SVC(kernel='poly', C=100, degree=1).fit(x_train, y_train_converted)
poly_svc_two = svm.SVC(kernel='poly', C=100, degree=2).fit(x_train, y_train_converted)
poly_svc_three = svm.SVC(kernel='poly', C=10, degree=3).fit(x_train, y_train_converted)
rbf_svc_one = svm.SVC(kernel='rbf', C=10, gamma=0.1).fit(x_train, y_train_converted)
rbf_svc_two = svm.SVC(kernel='rbf', C=100, gamma=0.1).fit(x_train, y_train_converted)
rbf_svc_three = svm.SVC(kernel='rbf', C=10, gamma=0.5).fit(x_train, y_train_converted)
rbf_svc_four = svm.SVC(kernel='rbf', C=100, gamma=0.5).fit(x_train, y_train_converted)

# Now, we run our test data through our trained models.
predicted_linear = svc.predict(x_test)
predicted_poly_one = poly_svc_one.predict(x_test)
predicted_poly_two = poly_svc_two.predict(x_test)
predicted_poly_three = poly_svc_three.predict(x_test)
predicted_rbf_one = rbf_svc_one.predict(x_test)
predicted_rbf_two = rbf_svc_two.predict(x_test)
predicted_rbf_three = rbf_svc_three.predict(x_test)
predicted_rbf_four = rbf_svc_four.predict(x_test)

# Print our accuracies.
'''
    print("SVM + Linear \t\t-> " + str(accuracy_score(y_test, predicted_linear)))
    print("SVM + Poly (D=1)\t-> " + str(accuracy_score(y_test, predicted_poly_one)))
    print("SVM + Poly (D=2)\t-> " + str(accuracy_score(y_test, predicted_poly_two)))
    print("SVM + Poly (D=3)\t-> " + str(accuracy_score(y_test, predicted_poly_three)))
    print("SVM + RBF-10/0.1 \t-> " + str(accuracy_score(y_test, predicted_rbf_one)))
    print("SVM + RBF-100/0.1 \t-> " + str(accuracy_score(y_test, predicted_rbf_two)))
    print("SVM + RBF-10/0.5 \t-> " + str(accuracy_score(y_test, predicted_rbf_three)))
    print("SVM + RBF-100/0.5 \t-> " + str(accuracy_score(y_test, predicted_rbf_four)))
'''

# # # # # # # # # # # # # # # # # # # # #
# Coarse Grid Search                    #
#   - Broad sweep of hyperparemeters.   #
# # # # # # # # # # # # # # # # # # # # #
'''
    # Set the parameters by cross-validation

    tuned_parameters = [
        {
            'kernel': ['linear'], 
            'C': [1, 10, 100, 1000]
        },
        {
            'kernel': ['poly'], 
            'degree': [2, 3, 4],
            'C': [1, 10, 100, 1000]
        },
        {
            'kernel': ['rbf'], 
            'gamma': [1e-3, 1e-4],
            'C': [1, 10, 100, 1000]
        }
    ]


    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
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
        y_true, y_pred = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_pred))
        print()
'''    
# # # # # # # # # # # # # # # # # # # # # # # # # #
# Fine Grid Search                                #
#   - A more detailed sweep of hyperparemeters.   #
# # # # # # # # # # # # # # # # # # # # # # # # # #
'''
    # Set the parameters by cross-validation
    tuned_parameters = {
        'kernel': ['poly'], 
        'degree': [2, 4],
        'C': [1, 5, 10, 100]
        }


    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()
'''
# # # # # # # # # # # # # # # # # # # # # # # # # #
# Fine Grid Search  2                             #
#   - A more detailed sweep of hyperparemeters.   #
# # # # # # # # # # # # # # # # # # # # # # # # # #
'''
    # Set the parameters by cross-validation
    tuned_parameters = {
        'kernel': ['poly'], 
        'degree': [2, 4],
        'C': [1, 50, 100]
        }


    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()
'''
# # # # # # # # # # # # # # # # # # # # # # # # # #
# Fine Grid Search  3                             #
#   - A more detailed sweep of hyperparemeters.   #
# # # # # # # # # # # # # # # # # # # # # # # # # #

# Set the parameters by cross-validation
tuned_parameters = {
    'kernel': ['poly'], 
    'degree': [2],
    'C': [10, 20, 30, 40]
    }


scores = ['precision', 'recall']

for score in scores:
    #print("# Tuning hyper-parameters for %s" % score)
    #print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(x_train, y_train)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    '''
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
    print()
    '''    
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()
    