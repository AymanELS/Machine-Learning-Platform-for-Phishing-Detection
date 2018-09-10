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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
import logging
import math
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


def Feature_Ranking(X,y,k, feature_list_dict_train):
	#RFE
	if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
		vectorizer=joblib.load("Data_Dump/Emails_Training/vectorizer.pkl")
	elif config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
		vectorizer=joblib.load("Data_Dump/URLs_Training/vectorizer.pkl")
	if config["Feature Ranking"]["Recursive Feature Elimination"] == "True":
		model = LogisticRegression()
		rfe = RFE(model, k)
		rfe.fit(X,y)
		X=rfe.transform(X)
		f=open("Data_Dump/Feature_ranking_rfe.txt",'w')
		res= dict(zip(vectorizer.get_feature_names(),rfe.ranking_))
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		with open("Data_Dump/Feature_ranking_rfe.txt",'w') as f:
			for (key, value) in sorted_d:
				f.write("{}: {}\n".format(key,value))
		return X, rfe

	#Chi-2
	elif config["Feature Ranking"]["Chi-2"] == "True":
		model= sklearn.feature_selection.SelectKBest(chi2, k)
		model.fit(X, y)
		res= dict(zip(vectorizer.get_feature_names(),model.scores_))
		for key, value in res.items():
			if math.isnan(res[key]):
				res[key]=0
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		with open("Data_Dump/Feature_ranking_chi2.txt",'w') as f:
			for (key, value) in sorted_d:
				f.write("{}: {}\n".format(key,value))
		X=model.transform(X)
		return X, model

	# Information Gain 
	elif config["Feature Ranking"]["Information Gain"] == "True":
		model = DecisionTreeClassifier(criterion='entropy')
		model.fit(X,y)
		# dump feature Ranking in a file
		res= dict(zip(vectorizer.get_feature_names(),model.feature_importances_))
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		with open("Data_Dump/Feature_ranking_IG.txt",'w') as f:
			for (key, value) in sorted_d:
				f.write("{}: {}\n".format(key,value))
		# create new model with the best k features
		new_list_dict_features=[]
		for i in range(k):
			key = sorted_d[i][0]
			#logger.info("key: {}".format(key))
			if "=" in key:
				key=key.split("=")[0]
			if i==0:
				for j in range(len(feature_list_dict_train)):
					#logger.info("key: {}, value {}".format(key, feature_list_dict_train[j][key]))
					new_list_dict_features.append({key: feature_list_dict_train[j][key]})
					#logger.info(new_list_dict_features)
			else:
				for j in range(len(feature_list_dict_train)):
					new_list_dict_features[j][key]=feature_list_dict_train[j][key]
					#logger.info(new_list_dict_features)
		#logger.info("new_list_dict_features: {}".format(len(new_list_dict_features[0])))
		vectorizer=DictVectorizer()
		X_train=vectorizer.fit_transform(new_list_dict_features)
		return X_train, vectorizer

	#Gini
	elif config["Feature Ranking"]["Gini"] == "True":
		model = DecisionTreeClassifier(criterion='gini')
		model.fit(X,y)
		res= dict(zip(vectorizer.get_feature_names(),model.feature_importances_))
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		with open("Data_Dump/Feature_ranking_Gini.txt",'w') as f:
			for (key, value) in sorted_d:
				f.write("{}: {}\n".format(key,value))
		
		new_list_dict_features=[]
		for i in range(k):
			key = sorted_d[i][0]
			if "=" in key:
				key=key.split("=")[0]
			if i==0:
				for j in range(len(feature_list_dict_train)):
					new_list_dict_features.append({key: feature_list_dict_train[j][key]})
			else:
				for j in range(len(feature_list_dict_train)):
					new_list_dict_features[j][key]=feature_list_dict_train[j][key]

		vectorizer=DictVectorizer()
		X_train=vectorizer.fit_transform(new_list_dict_features)
		return X_train, vectorizer
	

def Select_Best_Features_Training(X, y, k):
	selection= sklearn.feature_selection.SelectKBest(chi2, k)
	selection.fit(X, y)
	X=selection.transform(X)
	# Print out the list of best features
	return X, selection

	

def Select_Best_Features_Testing(X, selection, k, feature_list_dict_test ):
	if config["Feature Ranking"]["Recursive Feature Elimination"] == "True":
		X = selection.transform(X)
		logger.info("X_Shape: {}".format(X.shape))
		return X
	elif config["Feature Ranking"]["Chi-2"] == "True":
		X = selection.transform(X)
		logger.info("X_Shape: {}".format(X.shape))
		return X
	elif config["Feature Ranking"]["Information Gain"] == "True":
		best_features=[]
		with open("Data_Dump/Feature_ranking_IG.txt", 'r') as f:
			for line in f.readlines():
				best_features.append(line.split(':')[0])
		new_list_dict_features=[]
		for i in range(k):
			key=best_features[i]
			if "=" in key:
				key=key.split("=")[0]
			if i==0:
				for j in range(len(feature_list_dict_test)):
					new_list_dict_features.append({key: feature_list_dict_test[j][key]})
			else:
				for j in range(len(feature_list_dict_test)):
					new_list_dict_features[j][key]=feature_list_dict_test[j][key]
		X=selection.transform(new_list_dict_features)
		logger.info("X_Shape: {}".format(X.shape))
		return X
	elif config["Feature Ranking"]["Gini"] == "True":
		best_features=[]
		with open("Data_Dump/Feature_ranking_Gini.txt", 'r') as f:
			for line in f.readlines():
				best_features.append(line.split(':')[0])
		new_list_dict_features=[]
		for i in range(k):
			key=best_features[i]
			#logger.info("key: {}".format(key))
			if "=" in key:
				key=key.split("=")[0]
			if i==0:
				for j in range(len(feature_list_dict_test)):
					new_list_dict_features.append({key: feature_list_dict_test[j][key]})
			else:
				for j in range(len(feature_list_dict_test)):
					new_list_dict_features[j][key]=feature_list_dict_test[j][key]
		logger.info(new_list_dict_features)
		logger.info("new_list_dict_features shape: {}".format(len(new_list_dict_features[0])))
		X=selection.transform(new_list_dict_features)
		return X

	

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
