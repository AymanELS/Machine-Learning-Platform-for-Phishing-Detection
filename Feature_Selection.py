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


def Feature_Ranking(X,y,k):
	#RFE
	vectorizer=joblib.load("Data_Dump/Emails_Training/vectorizer.pkl")
	if config["Feature Ranking"]["Recursive Feature Elimination"] == "True":
		logger.info("Load Model ######")
		model = LogisticRegression()
		logger.info("Load RFE ######")
		rfe = RFE(model, k, verbose=1)
		logger.info("Fit RFE ######")
		rfe.fit(X,y)
		logger.info("Transform X ######")
		X=rfe.transform(X)
		f=open("Data_Dump/Feature_ranking_rfe.txt",'w')
		res= dict(zip(vectorizer.get_feature_names(),rfe.ranking_))
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		f.write(str(sorted_d))
		f.close()
		#np.savetxt(f,rfe.ranking_)
			#f.write(str(sorted_d))
		#logger.info(rfe.ranking_)

	# Information Gain 
	elif config["Feature Ranking"]["Information Gain"] == "True":
		model = DecisionTreeClassifier(criterion='entropy')
		#model = ExtraTreesClassifier(criterion='entropy')
		model.fit(X,y)
		#X=model.transform(X)
		f=open("Data_Dump/Feature_ranking_IG.txt",'w')
		res= dict(zip(vectorizer.get_feature_names(),model.feature_importances_))
		
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		#np.savetxt(f,rfe.ranking_)
		f.write(str(sorted_d))
		f.close()
		#logger.info(model.feature_importances_)

	#Gini	
	elif config["Feature Ranking"]["Gini"] == "True":
		model = DecisionTreeClassifier(criterion='gini')
		model.fit(X,y)
		f=open("Data_Dump/Feature_ranking_Gini.txt",'w')
		res= dict(zip(vectorizer.get_feature_names(),model.feature_importances_))
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		#np.savetxt(f,rfe.ranking_)
		f.write(str(sorted_d))
		f.close()
		#logger.info(model.feature_importances_)

	#Chi-2
	elif config["Feature Ranking"]["Chi-2"] == "True":
		model= sklearn.feature_selection.SelectKBest(chi2, k)
		model.fit(X, y)
		f=open("Data_Dump/Feature_ranking_chi2.txt",'w')
		res= dict(zip(vectorizer.get_feature_names(),model.scores_))
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		#np.savetxt(f,rfe.ranking_)
		f.write(str(sorted_d))
		f.close()
		X=model.transform(X)
	return X, model

def Select_Best_Features_Training(X, y, k):
	selection= sklearn.feature_selection.SelectKBest(chi2, k)
	selection.fit(X, y)
	X=selection.transform(X)
	# Print out the list of best features
	return X, selection

	

def Select_Best_Features_Testing(X, selection):
	X = selection.transform(X)
	# Print out the list of best features
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
