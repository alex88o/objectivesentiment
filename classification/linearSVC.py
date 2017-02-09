# Script create from newSVMtest.py, which exploits TRUNCATED and NORMALIZED CCA3 PROJECTIONS


# Effettua training di SVM lineare, variando il parametro C oppure usando quello dato in input (eventualmente come secondo parametro)
from __future__ import print_function

import sys
from sys import argv
from scipy.io import loadmat
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from pprint import pprint

#from utils import mat_dataset_to_dict


# Split the dataset in two equal parts
#XX_train, XX_test, yy_train, yy_test = train_test_split( X, y, test_size=0.5, random_state=0)

data_path = argv[1] 
#data_path= data_dir + 'DS1_v_t_os_all_trunc_n.mat'
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
#y_train =[y_tr, y_tr, y_tr]
#y_test = [y_te, y_te, y_te]

#print  np.array(X_train).shape
#print  np.array(X_test).shape
#print  np.array(y_train).shape
#print  np.array(y_test).shape
#print y_train[1:5]
#print y_train[1+len(y_tr):5+len(y_tr)]
#sys.exit(0) 



# Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


if len(argv)>2:
        C_range = [argv[2]]     # C value provided
else:
        C_exp_range = range(-5,16,2)
        C_range = [2**x for x in C_exp_range]
	C_range = [0.00001, 0.0001, 0.01, 1, 10, 100, 1000]

tuned_parameters = [{'C': C_range}]
# Usiamo un intervallo piu corto



#scores = ['precision', 'recall']
scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    # cv: cross-validation folds number
#    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=2,n_jobs=-1,
    clf = GridSearchCV(LinearSVC(), tuned_parameters, cv=2,n_jobs=2,
                       scoring = score)
#                      scoring = '%s_macro' % score)
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
    print("Accuracy score:\t%0.3f" % accuracy_score(y_true, y_pred))
    print()


