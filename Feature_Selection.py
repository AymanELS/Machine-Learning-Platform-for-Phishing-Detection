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
from sklearn.feature_extraction.text import TfidfVectorizer
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
import Features_Support
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
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
	


def Feature_Ranking(X, y, k, vectorizer, tfidf_vectorizer):
	#X=Features_Support.Preprocessing(X)
	#X_train, X_test, y, y_test = train_test_split(X_train, y, test_size=0.1, random_state=0)
	if config["Email or URL feature Extraction"]["extract_features_emails"]=="True":
		path_ranking="Data_Dump/Feature_ranking_emails"
	if config["Email or URL feature Extraction"]["extract_features_URLs"]=="True":
		path_ranking="Data_Dump/Feature_ranking_urls"
	if not os.path.exists(path_ranking):
		os.makedirs(path_ranking)
	if config["Feature Selection"]["with Tfidf"] == "True":
		features_list=(vectorizer.get_feature_names())+(tfidf_vectorizer.get_feature_names())
	else:
		features_list=(vectorizer.get_feature_names())
	#RFE
	if config["Feature Selection"]["Recursive Feature Elimination"] == "True":
		model = LinearSVC()
		rfe = RFE(model, k, verbose=2, step=0.01)
		X=rfe.fit_transform(X,y)
		#X_test=rfe.transform(X_test, y_test)
		res= dict(zip(features_list,rfe.ranking_))
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		if config["Feature Selection"]["with Tfidf"] == "True":
			with open(os.path.join(path_ranking,"Feature_ranking_RFE_TF_{}.txt".format(k)),'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
			joblib.dump(X, os.path.join(path_ranking,"X_train_processed_RFE_TF_{}.pkl".format(k)))
			joblib.dump(rfe, os.path.join(path_ranking,"RFE_TF_{}.pkl".format(k)))
			#joblib.dump(X_test, "Data_Dump/Emails_training/X_test_processed_TF_FH_RFE_{}.pkl".format(k))
			print("Printing RFE TFIDF: Done!")
		else:
			with open(os.path.join(path_ranking,"Feature_ranking_RFE_{}.txt".format(k)),'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
			joblib.dump(X, os.path.join(path_ranking,"X_train_processed_RFE_{}.pkl".format(k)))
			joblib.dump(rfe, os.path.join(path_ranking,"RFE_{}.pkl".format(k)))
			#joblib.dump(X_test, "Data_Dump/Emails_training/X_test_processed_FH_RFE_{}.pkl".format(k))
			print("Printing RFE NTF: Done!")
		return X, rfe
	# Information Gain
	if config["Feature Selection"]["Information Gain"] == "True":
		logger.debug("####IG###")
		model= sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='entropy'), threshold=-np.inf, max_features=k)
		logger.debug("model fit ...")
		model.fit(X, y)
		logger.debug("model fit done")
		res= dict(zip(features_list,model.estimator_.feature_importances_))
		for key, value in res.items():
			if math.isnan(res[key]):
				res[key]=0
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		logger.debug("model transform ...")
		X=model.transform(X)
#		X_test=model.transform(X_test)
		logger.debug("model transform done")
		if config["Feature Selection"]["with Tfidf"] == "True":
			with open(os.path.join(path_ranking,"Feature_ranking_IG_TF_{}.txt".format(k)),'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
			joblib.dump(X, os.path.join(path_ranking,"X_train_processed_IG_TF_{}.pkl".format(k)))
			joblib.dump(model, os.path.join(path_ranking,"IG_TF_{}.pkl".format(k)))
#			joblib.dump(X_test, "Data_Dump/Emails_Training/X_test_processed_TF_FH_IG_{}.pkl".format(k))
			logger.debug("IG TF Done")
		else:
			with open(os.path.join(path_ranking,"Feature_ranking_IG_{}.txt".format(k)),'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
			joblib.dump(X, os.path.join(path_ranking,"X_train_processed_IG_{}.pkl".format(k)))
			joblib.dump(model, os.path.join(path_ranking,"IG_{}.pkl".format(k)))
#			joblib.dump(X_test, "Data_Dump/Emails_Training/X_test_processed_FH_IG_{}.pkl".format(k))			
			logger.debug("IG NTF Done")
		return X, model

	#Chi-2
	if config["Feature Selection"]["Chi-2"] == "True":
		logger.debug("####Chi2###")
		model= sklearn.feature_selection.SelectKBest(chi2, k=k)
		model.fit(X, y)
		res= dict(zip(features_list,model.scores_))
		for key, value in res.items():
			if math.isnan(res[key]):
				res[key]=0
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		X=model.transform(X)
#		X_test=model.transform(X_test)
		if config["Feature Selection"]["with Tfidf"] == "True":
			with open(os.path.join(path_ranking,"Feature_ranking_Chi2_TF_{}.txt".format(k)),'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
			joblib.dump(X, os.path.join(path_ranking,"X_train_processed_Chi2_TF_{}.pkl".format(k)))
			joblib.dump(model, os.path.join(path_ranking,"Chi2_TF_{}.pkl".format(k)))
#			joblib.dump(X_test, "Data_Dump/Emails_Training/X_test_processed_TF_Chi2_{}.pkl".format(k))
			logger.debug("Chi2 TF Done")
		else:
			with open(os.path.join(path_ranking,"Feature_ranking_Chi2_{}.txt".format(k)),'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
			joblib.dump(X, os.path.join(path_ranking,"X_train_processed_Chi2_{}.pkl".format(k)))
			joblib.dump(model, os.path.join(path_ranking,"Chi2_{}.pkl".format(k)))
#			joblib.dump(X_test, "Data_Dump/Emails_Training/X_test_processed_Chi2_{}.pkl".format(k))			
			logger.debug("Chi2 NTF Done")
		return X, model

	#Gini
	if config["Feature Selection"]["Gini"] == "True":
		model= sklearn.feature_selection.SelectFromModel(DecisionTreeClassifier(criterion='gini'), threshold=-np.inf, max_features=k)
		logger.debug("model fit ...")
		model.fit(X, y)
		logger.debug("model fit done")
		res= dict(zip(features_list,model.estimator_.feature_importances_))
		for key, value in res.items():
			if math.isnan(res[key]):
				res[key]=0
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		X=model.transform(X)
#		X_test=model.transform(X_test)
		if config["Feature Selection"]["with Tfidf"] == "True":
			with open(os.path.join(path_ranking,"Feature_ranking_Gini_TF_{}.txt".format(k)),'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
			joblib.dump(X, os.path.join(path_ranking,"X_train_processed_Gini_TF_{}.pkl".format(k)))
			joblib.dump(model, os.path.join(path_ranking,"Gini_TF_{}.pkl".format(k)))
#			joblib.dump(X_test, "Data_Dump/Emails_training/X_test_processed_TF_Gini_{}.pkl".format(k))
			logger.debug("Gini TF Done")
		else:
			with open(os.path.join(path_ranking,"Feature_ranking_Gini_{}.txt".format(k)),'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
			joblib.dump(X, os.path.join(path_ranking,"X_train_processed_Gini_{}.pkl".format(k)))
			joblib.dump(model, os.path.join(path_ranking,"Gini_{}.pkl".format(k)))
#			joblib.dump(X_test, "Data_Dump/Emails_training/X_test_processed_Gini_{}.pkl".format(k))			
			logger.debug("Gini NTF Done")
		return X, model
	#LSA
	if config["Feature Selection"]["LSA"] == "True":
		logger.debug("####LSA###")	
		svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
		svd.fit(X, y)
		#res= dict(zip(features_list,svd.score_samples(X)))
		#for key, value in res.items():
		#	if math.isnan(res[key]):
		#		res[key]=0
		#sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		X=svd.transform(X)
		#X_test=svd.transform(X_test)
		if config["Feature Selection"]["with Tfidf"] == "True":	
			#with open(os.path.join(path_ranking,"Feature_Selection_LSA_TF_{}.txt".format(k)),'w') as f:
			#	for (key, value) in sorted_d:
			#		f.write("{}: {}\n".format(key,value))
			joblib.dump(X, os.path.join(path_ranking,"X_train_processed_LSA_TF_{}.pkl".format(k)))
			joblib.dump(svd, os.path.join(path_ranking,"LSA_TF_{}.pkl".format(k)))
		else:
			#with open(os.path.join(path_ranking,"Feature_Selection_LSA_{}.txt".format(k)),'w') as f:
			#	for (key, value) in sorted_d:
			#		f.write("{}: {}\n".format(key,value))
			joblib.dump(X, os.path.join(path_ranking,"X_train_processed_LSA_{}.pkl".format(k)))
			joblib.dump(svd, os.path.join(path_ranking,"LSA_{}.pkl".format(k)))
			#joblib.dump(X_test, "Data_Dump/Emails_Training/X_test_processed_TF_LSA_{}.pkl".format(k))
		return X, svd

	#PCA
	if config["Feature Selection"]["PCA"] == "True":
		logger.debug("####PCA###")	
		pca = PCA(n_components=k, svd_solver='auto')
		X=X.toarray()
		#X_test=X_test.toarray()
		pca.fit(X, y)
		#res= dict(zip(features_list,pca.score_samples(X)))
		#for key, value in res.items():
		#	if math.isnan(res[key]):
		#		res[key]=0
		#sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		X=pca.transform(X)
		#X_test=pca.transform(X_test)
		if config["Feature Selection"]["with Tfidf"] == "True":	
			#with open(os.path.join(path_ranking,"Feature_Selection_PCA_TF_{}.txt".format(k)),'w') as f:
			#	for (key, value) in sorted_d:
			#		f.write("{}: {}\n".format(key,value))
			joblib.dump(X, os.path.join(path_ranking,"X_train_processed_PCA_TF_{}.pkl".format(k)))
			joblib.dump(pca, os.path.join(path_ranking,"PCA_TF_{}.pkl".format(k)))
			#joblib.dump(X_test, "Data_Dump/Emails_Training/X_test_processed_TF_PCA_{}.pkl".format(k))
		else:
			#with open(os.path.join(path_ranking,"Feature_Selection_PCA_{}.txt".format(k)),'w') as f:
			#	for (key, value) in sorted_d:
			#		f.write("{}: {}\n".format(key,value))
			joblib.dump(X, os.path.join(path_ranking,"X_train_processed_PCA_{}.pkl".format(k)))
			joblib.dump(pca, os.path.join(path_ranking,"PCA_{}.pkl".format(k)))
			#joblib.dump(X_test, "Data_Dump/Emails_Training/X_test_processed_TF_PCA_{}.pkl".format(k))
		return X, pca

def Select_Best_Features_Testing(X_test, selection):
	if config["Feature Selection"]["PCA"] == "True":
		X_test=X_test.toarray()
	X_test=selection.transform(X_test)
	return X_test


def Feature_Ranking_tfidf(tfidf, y, k, tfidf_vectorizer):
	X=Features_Support.Preprocessing(tfidf)
	features_list=tfidf_vectorizer.get_feature_names()
	if not os.path.exists("Data_Dump/Feature_Ranking"):
		os.makedirs("Data_Dump/Feature_Ranking")
	if config["Feature Selection"]["Recursive Feature Elimination"] == "True":
		model = LinearSVC()
		rfe = RFE(model, k, verbose=2, step=0.01)
		X=rfe.fit_transform(X,y)
		#X=rfe.fit_transform(X_train,y)
		#X=rfe.transform(X)
		#features_list=(vectorizer.get_feature_names())#+(tfidf_vectorizer.get_feature_names())
		#features_list=(vectorizer.get_feature_names())
		res= dict(zip(features_list,rfe.ranking_))
		sorted_d = sorted(res.items(), key=lambda x: x[1], reverse=True)
		if config["Feature Selection"]["with Tfidf"] == "True":
			with open("Data_Dump/Feature_Ranking/Feature_ranking_rfe_TF_FH_{}.txt".format(k),'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
			#joblib.dump(X, "Data_Dump/Emails_training/X_train_processed_TF_FH_RFE_{}.pkl".format(k))
			#joblib.dump(X_test, "Data_Dump/Emails_training/X_test_processed_TF_FH_RFE_{}.pkl".format(k))
			print("Printing RFE TFIDF: Done!")
		else:
			with open("Data_Dump/Feature_Ranking/Feature_ranking_rfe_FH_{}.txt".format(k),'w') as f:
				for (key, value) in sorted_d:
					f.write("{}: {}\n".format(key,value))
			joblib.dump(X, "Data_Dump/Emails_training/X_train_processed_FH_RFE_{}.pkl".format(k))
			joblib.dump(X_test, "Data_Dump/Emails_training/X_test_processed_FH_RFE_{}.pkl".format(k))
			print("Printing RFE NTF: Done!")