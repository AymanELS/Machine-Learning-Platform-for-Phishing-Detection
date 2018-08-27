from sklearn import svm  
from sklearn import datasets
from collections import Counter 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import Imbalanced_Dataset
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.feature_extraction import DictVectorizer
import sys
import configparser
import re
import os
import pickle
from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_file
import logging
logger = logging.getLogger('root')

config=configparser.ConfigParser()
config.read('Config_file.ini')

####### Dataset (features for each item) X and Classess y (phish or legitimate)
def Feature_Selection(X,y):
	#if config["Feature_Selection"]["Chi-2"] == "True"
	#	X_Best=SelectKBest(chi2, k=2).fit_transform(X,y)
	#if config["Feature_Selection"]["Information_Gain"] == "True"
	#	X_Best=SelectKBest(mutual_info_classif, k=2).fit_transform(X,y)
	vec = joblib.load('vectorizer.pkl')
	res=dict(zip(vec.get_feature_names(),mutual_info_classif(X, y)))
	#sorted_d = sorted(res.items(), key=lambda x: x[1])
	logger.debug(res)
	#return X_Best

def Select_Best_Features(X_train, y_train, X_test, k):
	selection= sklearn.feature_selection.SelectKBest(chi2, k)
	selection.fit(X_train, y_train)
	X_train=selection.transform(X_train)
	X_test = selection.transform(X_test)
	# Print out the list of best features
	return X_train, X_test

def load_dataset():
	email_training_regex=re.compile(r"email_features_training_?\d?.txt")
	#email_testing_regex=re.compile(r"email_features_training_?\d?.txt")
	link_training_regex=re.compile(r"link_features_training_?\d?.txt")
	#link_testing_regex=re.compile(r"link_features_training_?\d?.txt")
	try:
		if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
			file_feature_training=re.findall(email_training_regex,''.join(os.listdir('.')))[-1]
			logger.debug("file_feature_training: {}".format(file_feature_training))
			#file_feature_testing=re.findall(email_testing_regex,''.join(os.listdir('.')))[-1]
		
		if config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
			file_feature_training=re.findall(link_training_regex,''.join(os.listdir('.')))[-1]
			#file_feature_testing=re.findall(link_testing_regex,''.join(os.listdir('.')))[-1]
	except Exception as e:
		logger.warning("exception: " + str(e))
	
	if config["Imbalanced Datasets"]["Load_imbalanced_dataset"] == "True":
		X, y = Imbalanced_Dataset.load_imbalanced_dataset(file_feature_training)
		#X_test, y_test=Imbalanced_Dataset.load_imbalanced_dataset(file_feature_testing)
	else:
		logger.debug("Imbalanced_Dataset not activated")
		X, y = load_svmlight_file(file_feature_training)
		#X_test, y_test = load_svmlight_file(file_feature_testing)
	return X, y#, X_test, y_test

def main():
	X, y = Imbalanced_Dataset.load_imbalanced_dataset("email_features_training_3.txt")
	Feature_Selection(X,y)


if __name__ == '__main__':
	config=configparser.ConfigParser()
	config.read('Config_file.ini')
	original = sys.stdout
	sys.stdout= open("log.txt",'w')
	main()
	sys.stdout=original
