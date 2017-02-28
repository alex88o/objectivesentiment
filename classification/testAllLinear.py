#!/usr/bin/env python
#title           :testAllLinear.py
#description     :Perform 10 runs train/test from a set of experimental setting files and the best value for C
#author          :A. Ortis
#date            :20170228
#version         :0.1
#usage           :python testAllLinear.py
#notes           :
#python_version  :2.6.6  
#==============================================================================

#from __future__ import print_function
import sys
from sys import argv
import json
from scipy.io import loadmat
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import h5py
import numpy as np
import os

import pprint


def findNviews(fname):
	s = fname.split('.')
	fname = s[0]
	s = fname.split('_')
	s = s[1:]
	res = 2
	print fname
	for i in range(len(s)):    #[3,4,5]:
		if s[i] == 'katsurai' or s[i] == 'all':
			res = i
			break
	return res
		
best_C = 0.03125
# Add elements to C_list to try other values for C
C_list = [best_C]
# Directory with input data for SVM
DIR = './dataForSVM'
datasets_list = os.listdir(DIR)
NUM_TRIALS = 10

for C in C_list:
	for idx,setFileName in enumerate(datasets_list):

		#Check input file
		path = DIR+"/"+setFileName
		if not os.path.isfile(path):
			continue
		s = setFileName.split('.')
		if not s[-1] == 'mat':
			continue

		#Number of considered views
		n_views = findNviews(setFileName)
		#Create a log file for each training/testing
		f = open(setFileName[:-3]+'_10runreport.txt','w')
		f.write("Running on dataset\t"+ setFileName + "\t n.views:\t" +  str(n_views))
		print "Running on dataset\t"+ setFileName + "\t n.views:\t" + str(n_views)
		print "C:\t" + str(C)

		try:	
			dataset = loadmat(file_name = path, chars_as_strings = True)
			dataset = dataset['DATA']

			X_train = dataset['training_instance_matrix'][0][0]
			y_train = dataset['training_label_vector'][0][0]
			X_test = dataset['testing_instance_matrix'][0][0]
			y_test = dataset['testing_label_vector'][0][0]


		except:	
			# An exception is thrown if the .mat file version is 7.5 or higher
			# Read .mat file version 7.5 or higher
			dataset_file = h5py.File(path,'r')

			dataset = dataset_file['DATA']

			X_train = np.array(dataset['training_instance_matrix'])
			X_train = X_train.transpose()

			y_train = np.array( dataset['training_label_vector'])
			y_train = y_train.transpose()

			X_test = np.array(dataset['testing_instance_matrix'])
			X_test = X_test.transpose()

			y_test = np.array(dataset['testing_label_vector'])
			y_test = y_test.transpose()


		n_test_images = len(y_test)
		y_tr = [x[0] for x in y_train]
		y_te = [x[0] for x in y_test]


		y_train = y_tr*n_views
		y_test =  y_te*n_views

		# Scale the data set before SVM
		# Fit the scaler with the train features
		scaler = preprocessing.StandardScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		# Transform the test features according to the learned scaler
		X_test = scaler.transform(X_test)

		X_train = np.array(X_train)
		y_train = np.array(y_train)

		scores = []
		print "\n\n"

		# Perform NUM_TRIALS runs and compute the average accuracy
		for i in range(NUM_TRIALS):
		     f.write("Run n.\t" + str(i+1))
		     print "Run n.\t" + str(i+1)
			
		     #Shuffle the train feature
		     shuf_idx = np.random.permutation(len(y_train))
		     X_train = X_train[shuf_idx,:]
		     y_train = y_train[shuf_idx]

		     #Fit the model 
	#	     clf = LinearSVC(C=best_C, random_state = i)
		     clf = LinearSVC(C=best_C)
		     clf.fit(X_train, y_train)
		     
                     #Perform testing
		     y_true, y_pred = y_test, clf.predict(X_test)

		     #Compute accuracy
		     acc = accuracy_score(y_true, y_pred)
		     f.write("Accuracy score:\t%0.5f" % acc)
		     print("Accuracy score:\t%0.5f" % acc)

		     #Collect the accuracy scores
		     scores.append(acc)

		# Mean accuracy and standard deviation
		print "Average Score: {0} STD: {1}".format(np.mean(scores), np.std(scores))
		f.write("Average Score: {0} STD: {1}".format(np.mean(scores), np.std(scores)))
		f.close()




