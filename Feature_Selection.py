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
from sklearn.feature_selection import SelectFromModel
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


def Feature_Ranking(X, y, k):
	#RFE
	if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
		emails=True
		urls=False
		if not os.path.exists("Data_Dump/Feature_Ranking_Emails"):
			os.makedirs("Data_Dump/Feature_Ranking_Emails")
		vectorizer=joblib.load("Data_Dump/Emails_Training/vectorizer.pkl")
		if config["Feature Selection"]["with Tfidf"]=="True":
			vectorizer_tfidf=joblib.load("Data_Dump/Emails_Training/tfidf_vectorizer.pkl")

	elif config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
		urls=True
		emails=False
		if not os.path.exists("Data_Dump/Feature_Ranking_URLs"):
			os.makedirs("Data_Dump/Feature_Ranking_URLs")
		vectorizer=joblib.load("Data_Dump/URLs_Training/vectorizer.pkl")
		if config["Feature Selection"]["with Tfidf"]=="True":
			vectorizer_tfidf=joblib.load("Data_Dump/URLs_Training/tfidf_vectorizer.pkl")
	if config["Feature Selection"]["Recursive Feature Elimination"] == "True":
		model = LogisticRegression()
		from sklearn.svm import LinearSVC
		model = LinearSVC()
		rfe = RFE(model, k, verbose=2, step=0.005)
		rfe.fit(X,y)
		X=rfe.transform(X)
		if config["Feature Selection"]["with Tfidf"]=="True":
			features_list=(vectorizer.get_feature_names())+(vectorizer_tfidf.get_feature_names())
		else:
			features_list=(vectorizer.get_feature_names())
		res= dict(zip(features_list,rfe.ranking_))
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		

		if config["Feature Selection"]["with Tfidf"]=="True":
			if emails:
				joblib.dump(X, "Data_Dump/Emails_Training/X_train_with_tfidf_RFE_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_Emails/Feature_ranking_rfe_with_tfidf.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
			if urls:
				joblib.dump(X, "Data_Dump/URLs_Training/X_train_with_tfidf_RFE_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_URLs/Feature_ranking_rfe_with_tfidf.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
		else:
			if emails:
				joblib.dump(X, "Data_Dump/Emails_Training/X_train_RFE_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_Emails/Feature_ranking_rfe.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
			if urls:
				joblib.dump(X, "Data_Dump/URLs_Training/X_train_RFE_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_URLs/Feature_ranking_rfe.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
		return X, rfe

	#Chi-2
	elif config["Feature Selection"]["Chi-2"] == "True":
		model= sklearn.feature_selection.SelectKBest(chi2, k)
		model.fit(X, y)
		if config["Feature Selection"]["with Tfidf"]=="True":
			features_list=(vectorizer.get_feature_names())+(vectorizer_tfidf.get_feature_names())
		else:
			features_list=(vectorizer.get_feature_names())
		res= dict(zip(features_list,model.scores_))
		for key, value in res.items():
			if math.isnan(res[key]):
				res[key]=0
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)

		X=model.transform(X)
		if config["Feature Selection"]["with Tfidf"]=="True":
			if emails:
				joblib.dump(X, "Data_Dump/Emails_Training/X_train_with_tfidf_Chi2_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_Emails/Feature_ranking_chi2_with_tfidf.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
			if urls:
				joblib.dump(X, "Data_Dump/URLs_Training/X_train_with_tfidf_Chi2_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_URLs/Feature_ranking_chi2_with_tfidf.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
		else:
			if emails:
				joblib.dump(X, "Data_Dump/Emails_Training/X_train_Chi2_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_Emails/Feature_ranking_chi2.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
			if urls:
				joblib.dump(X, "Data_Dump/URLs_Training/X_train_Chi2_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_URLs/Feature_ranking_chi2.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
		return X, model

	# Information Gain
	elif config["Feature Selection"]["Information Gain"] == "True":
		model= sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='entropy'), threshold=-np.inf, max_features=k)
		model.fit(X,y)
		# dump Feature Selection in a file
		if config["Feature Selection"]["with Tfidf"]=="True":
			features_list=(vectorizer.get_feature_names())+(vectorizer_tfidf.get_feature_names())
		else:
			features_list=(vectorizer.get_feature_names())
		res= dict(zip(features_list,model.estimator_.feature_importances_))
		for key, value in res.items():
			if math.isnan(res[key]):
				res[key]=0
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		# create new model with the best k features
		X=model.transform(X)

		if config["Feature Selection"]["with Tfidf"]=="True":
			if emails:
				joblib.dump(X, "Data_Dump/Emails_Training/X_train_with_tfidf_IG_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_Emails/Feature_ranking_IG_with_tfidf.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
			if urls:
				joblib.dump(X, "Data_Dump/URLs_Training/X_train_with_tfidf_IG_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_URLs/Feature_ranking_IG_with_tfidf.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
		else:
			if emails:
				joblib.dump(X, "Data_Dump/Emails_Training/X_train_IG_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_Emails/Feature_ranking_IG.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))

			if urls:
				joblib.dump(X, "Data_Dump/URLs_Training/X_train_IG_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_URLs/Feature_ranking_IG.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
		return X, model

	#Gini
	elif config["Feature Selection"]["Gini"] == "True":
		model= sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='gini'), threshold=-np.inf, max_features=k)
		model.fit(X,y)
		if config["Feature Selection"]["with Tfidf"]=="True":
			features_list=(vectorizer.get_feature_names())+(vectorizer_tfidf.get_feature_names())
		else:
			features_list=(vectorizer.get_feature_names())
		res= dict(zip(features_list,model.estimator_.feature_importances_))
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		for key, value in res.items():
			if math.isnan(res[key]):
				res[key]=0
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		if emails:	
			with open("Data_Dump/Feature_Ranking_Emails/Feature_ranking_Gini.txt",'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
		if urls:
			with open("Data_Dump/Feature_Ranking_URLs/Feature_ranking_Gini.txt",'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
		# create new model with the best k features
		X=model.transform(X)
		if config["Feature Selection"]["with Tfidf"]=="True":
			if emails:
				joblib.dump(X, "Data_Dump/Emails_Training/X_train_with_tfidf_Gini_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_Emails/Feature_ranking_Gini_with_tfidf.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
			if urls:
				joblib.dump(X, "Data_Dump/URLs_Training/X_train_with_tfidf_Gini_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_URLs/Feature_ranking_Gini_with_tfidf.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))

		else:
			if emails:
				joblib.dump(X, "Data_Dump/Emails_Training/X_train_Gini_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_Emails/Feature_ranking_Gini.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
			if urls:
				joblib.dump(X, "Data_Dump/URLs_Training/X_train_Gini_{}.pkl".format(k))
				with open("Data_Dump/Feature_Ranking_URLs/Feature_ranking_Gini.txt",'w') as f:
					for (key, value) in sorted_d:
						f.write("{}: {}\n".format(key,value))
		return X, model

	# Information Gain
	elif config["Feature Selection"]["Information Gain"] == "True":
		model= sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='entropy'), threshold=-np.inf, max_features=k)
		model.fit(X,y)
		# dump Feature Selection in a file
		if config["Feature Selection"]["with Tfidf"]=="True":
			features_list=(vectorizer.get_feature_names())+(vectorizer_tfidf.get_feature_names())
		else:
			features_list=(vectorizer.get_feature_names())
		res= dict(zip(features_list,model.estimator_.feature_importances_))
		for key, value in res.items():
			if math.isnan(res[key]):
				res[key]=0
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		if emails:
			with open("Data_Dump/Feature_Ranking_Emails/Feature_ranking_IG.txt",'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
		if urls:
			with open("Data_Dump/Feature_Ranking_URLs/Feature_ranking_IG.txt",'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
		# create new model with the best k features
		X=model.transform(X)
		if config["Feature Selection"]["with Tfidf"]=="True":
			if emails:
				joblib.dump(X, "Data_Dump/Emails_Training/X_train_with_tfidf_IG_{}.pkl".format(k))
			if urls:
				joblib.dump(X, "Data_Dump/URLs_Training/X_train_with_tfidf_IG_{}.pkl".format(k))
		else:
			if emails:
				joblib.dump(X, "Data_Dump/Emails_Training/X_train_IG_{}.pkl".format(k))
			if urls:
				joblib.dump(X, "Data_Dump/URLs_Training/X_train_IG_{}.pkl".format(k))
		return X, model

	#Gini
	elif config["Feature Selection"]["Gini"] == "True":
		model= sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='gini'), threshold=-np.inf, max_features=k)
		model.fit(X,y)
		if config["Feature Selection"]["with Tfidf"]=="True":
			features_list=(vectorizer.get_feature_names())+(vectorizer_tfidf.get_feature_names())
		else:
			features_list=(vectorizer.get_feature_names())
		res= dict(zip(features_list,model.estimator_.feature_importances_))
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		for key, value in res.items():
			if math.isnan(res[key]):
				res[key]=0
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		if emails:	
			with open("Data_Dump/Feature_Ranking_Emails/Feature_ranking_Gini.txt",'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
		if urls:
			with open("Data_Dump/Feature_Ranking_URLs/Feature_ranking_Gini.txt",'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
		# create new model with the best k features
		X=model.transform(X)
		if config["Feature Selection"]["with Tfidf"]=="True":
			if emails:
				joblib.dump(X, "Data_Dump/Emails_Training/X_train_with_tfidf_Gini_{}.pkl".format(k))
			if urls:
				joblib.dump(X, "Data_Dump/URLs_Training/X_train_with_tfidf_Gini_{}.pkl".format(k))
		else:
			if emails:
				joblib.dump(X, "Data_Dump/Emails_Training/X_train_Gini_{}.pkl".format(k))
			if urls:
				joblib.dump(X, "Data_Dump/URLs_Training/X_train_Gini_{}.pkl".format(k))
		return X, model
	

def Select_Best_Features_Training(X, y, k):
	selection= sklearn.feature_selection.SelectKBest(chi2, k)
	selection.fit(X, y)
	X=selection.transform(X)
	# Print out the list of best features
	return X, selection

#<<<<<<< HEAD
def Select_Best_Features_Testing(X, selection):
	if config["Feature Selection"]["Recursive Feature Elimination"] == "True":
		X = selection.transform(X)
		logger.info("X_Shape: {}".format(X.shape))
		return X
	elif config["Feature Selection"]["Chi-2"] == "True":
		X = selection.transform(X)
		logger.info("X_Shape: {}".format(X.shape))
		return X
	elif config["Feature Selection"]["Information Gain"] == "True":
		X = selection.transform(X)
		logger.info("X_Shape: {}".format(X.shape))
		return X
	elif config["Feature Selection"]["Gini"] == "True":
		X = selection.transform(X)
		logger.info("X_Shape: {}".format(X.shape))
		return X


	
#=======
#def Select_Best_Features_Testing(X, selection):
#	print (selection)
#	try:
#		X = selection.transform(X)
#	# Print out the list of best features
#	except AttributeError as e:
#		print (e)
#	return X
#>>>>>>> b62393cd258add9238fd4d2d3d8dc626851086d7

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
