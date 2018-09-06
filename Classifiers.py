from sklearn import svm
from sklearn import datasets
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from imblearn.datasets import make_imbalance
from imblearn.under_sampling import RandomUnderSampler,CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.datasets import load_svmlight_file
from keras.losses import mean_squared_error
#import User_options
import sklearn
from math import pow
import Evaluation_Metrics
import Imbalanced_Dataset
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import tensorflow as tf
import math
import os, os.path
from pathlib import Path
import re
import configparser
import Features
import pickle
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
import logging
#from collections import deque

logger = logging.getLogger('root')

config=configparser.ConfigParser()
config.read('Config_file.ini')

####### Dataset (features for each item) X and Classess y (phish or legitimate)


def load_dataset():
	email_training_regex=re.compile(r"email_features_training_?\d?.txt")
	email_testing_regex=re.compile(r"verbose=1email_features_testing_?\d?.txt")

	link_training_regex=re.compile(r"link_features_training_?\d?.txt")
	link_testing_regex=re.compile(r"link_features_testing_?\d?.txt")
	try:
		if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
			file_feature_training=re.findall(email_training_regex,''.join(os.listdir('.')))[-1]
			file_feature_testing=re.findall(email_testing_regex,''.join(os.listdir('.')))[-1]

		if config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
			file_feature_training=re.findall(link_training_regex,''.join(os.listdir('.')))[-1]
			file_feature_testing=re.findall(link_testing_regex,''.join(os.listdir('.')))[-1]
	except Exception as e:
		logger.error("exception: " + str(e))

	if config["Imbalanced Datasets"]["Load_imbalanced_dataset"] == "True":
		X, y = Imbalanced_Dataset.load_imbalanced_dataset(file_feature_training)
		logger.debug(file_feature_training)
		X_test, y_test=Imbalanced_Dataset.load_imbalanced_dataset(file_feature_testing)
		logger.debug(file_feature_testing)
	else:
		logger.info("Imbalanced_Dataset not activated")
		logger.debug(file_feature_training)
		logger.debug(file_feature_testing)
		X, y = load_svmlight_file(file_feature_training)
		X_test, y_test = load_svmlight_file(file_feature_testing)
	return X, y, X_test, y_test

def load_dictionary():

	list_dict_train=joblib.load('list_dict_train.pkl')
	list_dict_test=joblib.load('list_dict_test.pkl')
	vec=DictVectorizer()
	Sparse_Matrix_Features_train=vec.fit_transform(list_dict_train)
	Sparse_Matrix_Features_test=vec.transform(list_dict_test)

	labels_train=joblib.load('labels_train.pkl')
	labels_test=joblib.load('labels_test.pkl')
	#preprocessing
	return Sparse_Matrix_Features_train, labels_train, Sparse_Matrix_Features_test, labels_test


def SVM(X,y, X_test, y_test):
	#print(X)
	clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
   		decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
   		max_iter=-1, probability=False, random_state=None, shrinking=True,
   		tol=0.001, verbose=False)
	clf.fit(X, y)
	y_predict=clf.predict(X_test)
	logger.info("SVM >>>>>>>")
	Evaluation_Metrics.eval_metrics(clf, X, y, y_test, y_predict)

######## Random Forest
def RandomForest(X,y, X_test, y_test):
		clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
		 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
		  min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1,
		   random_state=None, verbose=0, warm_start=False, class_weight=None)
		clf.fit(X,y)
		y_predict=clf.predict(X_test)
		logger.info("RF >>>>>>>")
		Evaluation_Metrics.eval_metrics(clf, X, y, y_test, y_predict)

###### Decition Tree
def DecisionTree(X,y, X_test, y_test):
		clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
		 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
		 min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
		clf.fit(X,y)
		y_predict=clf.predict(X_test)
		logger.info("DT >>>>>>>")
		Evaluation_Metrics.eval_metrics(clf, X, y, y_test, y_predict)

##### Gaussian Naive Bayes
def GaussianNaiveBayes(X,y, X_test, y_test):
		gnb = GaussianNB(priors=None)
		#X=X.toarray()
		#X_test=X_test.toarray()
		gnb.fit(X,y)
		y_predict=gnb.predict(X_test)
		logger.info("GNB >>>>>>>")
		Evaluation_Metrics.eval_metrics(gnb, X, y, y_test, y_predict)

##### Multinomial Naive Bayes
def MultinomialNaiveBayes(X,y, X_test, y_test):
		#X=X.toarray()
		#X_test=X_test.toarray()
		mnb=MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
		mnb.fit(X,y)
		y_predict=mnb.predict(X_test)
		logger.info("MNB >>>>>>>")
		Evaluation_Metrics.eval_metrics(mnb, X, y, y_test, y_predict)

##### Logistic Regression
def LogisticRegression(X,y, X_test, y_test):
		clf=sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
			class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
			 verbose=0, warm_start=False, n_jobs=1)
		clf.fit(X,y)
		y_predict=clf.predict(X_test)
		logger.info("LR >>>>>>>")
		Evaluation_Metrics.eval_metrics(clf, X, y, y_test, y_predict)

##### k-Nearest Neighbor
def kNearestNeighbor(X,y, X_test, y_test):
		clf=KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='auto', leaf_size=30, p=2,
		 metric='minkowski', metric_params=None, n_jobs=1,)
		clf.fit(X,y)
		y_predict=clf.predict(X_test)
		logger.info("KNN >>>>>>>")
		Evaluation_Metrics.eval_metrics(clf, X, y, y_test, y_predict)

##### KMeans
def KMeans(X,y, X_test, y_test):
		clf=sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
 		verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
		clf.fit(X,y)
		logger.info("Kmeans")
		y_predict=clf.predict(X_test)
		Evaluation_Metrics.eval_metrics_cluster(y_test, y_predict)
		#Evaluation_Metrics.eval_metrics(clf, X, y, y_test, y_predict)

##### Bagging
def Bagging(X,y, X_test, y_test):
		clf=BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, max_samples=1.0, max_features=1.0,
		 bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None,
		  verbose=0)
		clf.fit(X,y)
		y_predict=clf.predict(X_test)
		logger.info("Bagging_scores >>>>>>>")
		Evaluation_Metrics.eval_metrics(clf, X, y, y_test, y_predict)

#### Boosting
def Boosting(X,y, X_test, y_test):
		clf = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R',
		 random_state=None)
		clf.fit(X,y)
		y_predict=clf.predict(X_test)
		logger.info("Boosting >>>>>>>")
		Evaluation_Metrics.eval_metrics(clf, X, y, y_test, y_predict)

############### imbalanced learning
def DNN(X,y, X_test, y_test):
		#X_test, y_test = load_dataset("feature_vector_extract_test.txt")
		K.set_learning_phase(1) #set learning phase
		model_dnn = Sequential()
		print(X.shape)
		dim=X.shape[1]
		print(dim) ##
		print("Start Building Model")
		model_dnn.add(Dense(80, kernel_initializer='normal', activation='relu', input_dim=dim)) #units in Dense layer as same as the input dim
		model_dnn.add(Dense(1, activation='sigmoid'))
		model_dnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		print("model compile end >>>>>>")
		model_dnn.fit(X, y, epochs=150, batch_size=100, verbose=2)
		y_predict=model_dnn.predict(X_test)
		logger.info("DNN >>>>>>>")
		Evaluation_Metrics.eval_metrics(model_dnn, X, y, y_test, y_predict)

def HDDT():
	#java -cp <path to weka-hddt.jar> weka.classifiers.trees.HTree -U -A -B -t <training file> -T <testing file>
	weka_hddt_path="weka-hddt-3-7-1.jar"
	subprocess.call(['java', '-cp', weka_hhdt_path,'weka.classifiers.trees.HTree', '-U', '-A' '-B' '-t', y_predict, y_test])
##To-Do: Add DNN and OLL
####
def classifiers(X,y, X_test, y_test):
	logger.info("##### Classifiers #####")
	summary=Features.summary
	summary.write("\n##############\n\nClassifiers Used:\n")
	#X,y, X_test, y_test=load_dataset()
	#X,y, X_test, y_test=load_dictionary()
	if config["Classifiers"]["SVM"] == "True":
		SVM(X,y, X_test, y_test)
		summary.write("SVM\n")
	if config["Classifiers"]["RandomForest"] == "True":
		RandomForest(X,y, X_test, y_test)
		summary.write("Random Forest\n")
	if config["Classifiers"]["DecisionTree"] == "True":
		DecisionTree(X,y, X_test, y_test)
		summary.write("Decision Tree \n")
	if config["Classifiers"]["GaussianNaiveBayes"] == "True":
		GaussianNaiveBayes(X,y, X_test, y_test)
		summary.write("Gaussian Naive Bayes \n")
	if config["Classifiers"]["MultinomialNaiveBayes"] == "True":
		MultinomialNaiveBayes(X,y, X_test, y_test)
		summary.write("Multinomial Naive Bayes \n")
	if config["Classifiers"]["LogisticRegression"] == "True":
		LogisticRegression(X,y, X_test, y_test)
		summary.write("Logistic Regression\n")
	if config["Classifiers"]["kNearestNeighbor"] == "True":
		kNearestNeighbor(X,y, X_test, y_test)
		summary.write("kNearest Neighbor\n")
	if config["Classifiers"]["KMeans"] == "True":
		KMeans(X,y, X_test, y_test)
		summary.write("kMeans \n")
	if config["Classifiers"]["Bagging"] == "True":
		Bagging(X,y, X_test, y_test)
		summary.write("Bagging \n")
	if config["Classifiers"]["Boosting"] == "True":
		Boosting(X,y, X_test, y_test)
		summary.write("Boosting \n")
	if config["Classifiers"]["DNN"] == "True":
		DNN(X,y, X_test, y_test)
		summary.write("DNN \n")

def fit_MNB(X,y):
	mnb=MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
	mnb.fit(X,y)
	logger.info("MNB >>>>>>>")
	joblib.dump(mnb,"Data_Dump/Emails_Training/MNB_model.pkl")
	
