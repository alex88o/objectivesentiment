#	Author:	A.Ortis
#	
#	This script perform the RBF SVC training and test. If the value of C is provided, it only search for optimal gamma.
#	This script takes a .mat as input containing the train/test sets. And an optional further value for the parameter C.


from __future__ import print_function

import sys
from sys import argv
from scipy.io import loadmat
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn import preprocessing

from pprint import pprint

#data_dir = './dataForSVM/'
#data_path= data_dir + 'DS1_v_t_os_all_features.mat'
data_path = argv[1]
dataset = loadmat(file_name = data_path, chars_as_strings = True)

dataset = dataset['DATA']

X_train = dataset['training_instance_matrix'][0][0]
y_train = dataset['training_label_vector'][0][0]
X_test = dataset['testing_instance_matrix'][0][0]
y_test = dataset['testing_label_vector'][0][0]

import numpy as np

#X_train = np.array(X_train).shape
#X_test =  np.array(X_test).shape
#y_train =  np.array(y_train).shape
#y_test =  np.array(y_test).shape


# Scale the train set and the test set before SVM
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_tr = [x[0] for x in y_train]
y_te = [x[0] for x in y_test]

y_train = y_tr*3
y_test =  y_te*3

#print  np.array(X_train).shape
#print  np.array(X_test).shape
#print  np.array(y_train).shape
#print  np.array(y_test).shape
#sys.exit(0) 



# Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

gamma_exp_range = range(-15,4,2)
gamma_range = [2**x for x in gamma_exp_range]
# NB: a causa di computazione elevata, utilizziamo solo due valori di gamma
gamma_range = [1e-3, 1e-4]

if len(argv)>2:
	C_range = [argv[2]]	# C value provided
else:
	C_exp_range = range(-5,16,2)
	C_range = [2**x for x in C_exp_range] 
	# Usiamo in intervallo piu corto per questioni di efficienza
	C_range = [0.00001, 0.0001, 0.01, 1, 10, 100, 1000]


tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range,
                     'C': C_range}]

print("Input file:\t",data_path)
print("Tuned parameters:")
pprint.pprint(tuned_parameters)

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10,n_jobs=2,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

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
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    print(accuracy_score(y_true, y_pred))
    print()


