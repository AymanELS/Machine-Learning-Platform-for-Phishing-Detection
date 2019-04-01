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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cluster import KMeans
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
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

def fit_classifier(clf, X, y, X_train_balanced=None, y_train_balanced=None):
	if X_train_balanced is not None and y_train_balanced is not None:
		clf.fit(X_train_balanced,y_train_balanced)
	else:
		clf.fit(X,y)



def SVM(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
	if clf is None:
		if config["Classifiers"]["weighted"] == "True":
			clf = svm.SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
	        	        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
		                max_iter=-1, probability=False, random_state=None, shrinking=True,
		                tol=0.001, verbose=False)
		else:
			clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
		   		decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
		   		max_iter=-1, probability=False, random_state=None, shrinking=True,
		   		tol=0.001, verbose=False)
		logger.info("SVM >>>>>>>")
		if config["Evaluation Metrics"]["cross_val_score"]=="True":
			score=Evaluation_Metrics.Cross_validation(clf, X, y)
			logger.info(score)
			return score, None
		else:
			clf.fit(X, y)
			y_predict=clf.predict(X_test)
			eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
			return eval_metrics_SVM, clf
	else:
		y_predict=clf.predict(X_test)
		eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
		return eval_metrics_SVM, clf

######## Random Forest
def RandomForest(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
	if clf is None:
		if config["Classifiers"]["weighted"] == "True":
			clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
				min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
				min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1,
				random_state=None, verbose=0, warm_start=False, class_weight='balanced')
		else:
			clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
				min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
				min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1,
				random_state=None, verbose=0, warm_start=False, class_weight=None)
		logger.info("RF >>>>>>>")
		if config["Evaluation Metrics"]["cross_val_score"]=="True":
			score=Evaluation_Metrics.Cross_validation(clf, X, y)
			logger.info(score)
			return score, None
		else:
			fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
			y_predict=clf.predict(X_test)
			eval_metrics_RF = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
			return eval_metrics_RF, clf
	else:
		y_predict=clf.predict(X_test)
		eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
		return eval_metrics_SVM, clf

###### Decition Tree
def DecisionTree(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
	if clf is None:
		if config["Classifiers"]["weighted"] == "True":
			clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
	                        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
	                        min_impurity_decrease=0.0, min_impurity_split=None, class_weight='balanced', presort=False)
		else:
			clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
				min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
				min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
		logger.info("DT >>>>>>>")
		if config["Evaluation Metrics"]["cross_val_score"]=="True":
			score=Evaluation_Metrics.Cross_validation(clf, X, y)
			logger.info(score)
			return score, None
		else:
			fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
			y_predict=clf.predict(X_test)			
			eval_metrics_DT = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
			return eval_metrics_DT, clf
	else:
		y_predict=clf.predict(X_test)
		eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
		return eval_metrics_SVM, clf


##### Gaussian Naive Bayes
def GaussianNaiveBayes(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None,clf=None):
	if clf is None:
		if config["Classifiers"]["weighted"] == "True":
			logger.warn("GaussianNaiveBayes does not support weighted classification")
			return
		clf = GaussianNB(priors=None)
		logger.info("GNB >>>>>>>")
		if config["Evaluation Metrics"]["cross_val_score"]=="True":
			score=Evaluation_Metrics.Cross_validation(clf, X, y)
			logger.info(score)
			return score, None
		else:
			fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
			y_predict=gnb.predict(X_test)
			eval_metrics_NB = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
			return eval_metrics_NB, clf
	else:
		y_predict=clf.predict(X_test)
		eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
		return eval_metrics_SVM, clf

##### Multinomial Naive Bayes
def MultinomialNaiveBayes(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
	if clf is None:
		if config["Classifiers"]["weighted"] == "True":
	                logger.warn("MultinomialNaiveBayes does not support weighted classification")
	                return
		clf=MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
		logger.info("MNB >>>>>>>")
		if config["Evaluation Metrics"]["cross_val_score"]=="True":
			score=Evaluation_Metrics.Cross_validation(clf, X, y)
			logger.info(score)
			return score, None
		else:
			fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
			y_predict=mnb.predict(X_test)
			eval_metrics_MNB = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
			return eval_metrics_MNB, clf
	else:
		y_predict=clf.predict(X_test)
		eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
		return eval_metrics_SVM, clf
	
##### Logistic Regression
def LogisticRegression(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
	if clf is None:
		if config["Classifiers"]["weighted"] == "True":
			clf=sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
				class_weight='balanced', random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
				verbose=0, warm_start=False, n_jobs=1)
		else:
			clf=sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
				class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
				verbose=0, warm_start=False, n_jobs=1)
		logger.info("LR >>>>>>>")
		if config["Evaluation Metrics"]["cross_val_score"]=="True":
			score=Evaluation_Metrics.Cross_validation(clf, X, y)
			logger.info(score)
			return score, None
		else:
			fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
			y_predict=clf.predict(X_test)
			eval_metrics_LR = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
			return eval_metrics_LR, clf
	else:
		y_predict=clf.predict(X_test)
		eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
		return eval_metrics_SVM, clf

##### ELM
def ELM(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
	if clf is None:
                if config["Classifiers"]["weighted"] == "True":
	                logger.warn("kNearestNeighbor does not support weighted classification")
	                return
	
                srhl_tanh = MLPRandomLayer(n_hidden=10, activation_func='tanh')
                clf = GenELMClassifier(hidden_layer=srhl_tanh)
                logger.info("ELM >>>>>>>")
                if config["Evaluation Metrics"]["cross_val_score"]=="True":
                        score=Evaluation_Metrics.Cross_validation(clf, X, y)
                        logger.info(score)
                        return score, None
                else:
                        fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
                        y_predict=clf.predict(X_test)
                        eval_metrics_ELM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
                        return eval_metrics_ELM, clf
	else:
		y_predict=clf.predict(X_test)
		eval_metrics_ELM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
		return eval_metrics_ELM, clf

##### k-Nearest Neighbor
def kNearestNeighbor(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
	if clf is None:
		if config["Classifiers"]["weighted"] == "True":
	                logger.warn("kNearestNeighbor does not support weighted classification")
	                return
	
		clf=KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='auto', leaf_size=30, p=2,
			metric='minkowski', metric_params=None, n_jobs=1,)
		logger.info("KNN >>>>>>>")
		if config["Evaluation Metrics"]["cross_val_score"]=="True":
			score=Evaluation_Metrics.Cross_validation(clf, X, y)
			logger.info(score)
			return score, None
		else:
			fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
			y_predict=clf.predict(X_test)
			eval_metrics_KNN = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
			return eval_metrics_KNN, clf
	else:
		y_predict=clf.predict(X_test)
		eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
		return eval_metrics_SVM, clf
	
##### KMeans
def KMeans(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
	if clf is None:
		if config["Classifiers"]["weighted"] == "True":
	                logger.warn("KMeans does not support weighted classification")
	                return
	
		clf=sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
			verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
		logger.info("Kmeans >>>>>>>")
		if config["Evaluation Metrics"]["cross_val_score"]=="True":
			score=Evaluation_Metrics.Cross_validation(clf, X, y)
			logger.info(score)
			return score, None
		else:
			fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
			y_predict=clf.predict(X_test)
			eval_metrics_kmeans = Evaluation_Metrics.eval_metrics_cluster(y_test, y_predict)
			return eval_metrics_kmeans, clf
			#Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
	else:
		y_predict=clf.predict(X_test)
		eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
		return eval_metrics_SVM, clf

##### Bagging
def Bagging(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
	if clf is None:
		if config["Classifiers"]["weighted"] == "True":
			base_classifier=DecisionTreeClassifier(class_weight='balanced')
		else:
			base_classifier=DecisionTreeClassifier()
	
		clf=BaggingClassifier(base_estimator=base_classifier, n_estimators=10, max_samples=1.0, max_features=1.0,
			bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None,
			verbose=0)
		logger.info("Bagging_scores >>>>>>>")
		if config["Evaluation Metrics"]["cross_val_score"]=="True":
			score=Evaluation_Metrics.Cross_validation(clf, X, y)
			logger.info(score)
			return score, None
		else:
			fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
			y_predict=clf.predict(X_test)
			eval_metrics_bagging = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
			return eval_metrics_bagging, clf
	else:
		y_predict=clf.predict(X_test)
		eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
		return eval_metrics_SVM, clf

#### Boosting
def Boosting(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None, clf=None):
	if clf is None:
		if config["Classifiers"]["weighted"] == "True":
			base_classifier=DecisionTreeClassifier(class_weight='balanced')
		else:
			base_classifier=DecisionTreeClassifier()
	
		clf = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R',
			random_state=None)
		logger.info("Boosting >>>>>>>")
		if config["Evaluation Metrics"]["cross_val_score"]=="True":
			score=Evaluation_Metrics.Cross_validation(clf, X, y)
			logger.info(score)
			return score, None
		else:
			fit_classifier(clf, X, y, X_train_balanced, y_train_balanced)
			y_predict=clf.predict(X_test)
			eval_metrics_boosting = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
			return eval_metrics_boosting, clf
	else:
		y_predict=clf.predict(X_test)
		eval_metrics_SVM = Evaluation_Metrics.eval_metrics(clf, y_test, y_predict)
		return eval_metrics_SVM, clf

############### imbalanced learning
def DNN(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None):
	if config["Classifiers"]["weighted"] == "True":
                logger.warn("DNN does not support weighted classification")
                return
	from sklearn.model_selection import StratifiedKFold
	np.set_printoptions(threshold=np.nan)
	def model_build(dim):
		logger.debug("Start Building DNN Model >>>>>>")
		K.set_learning_phase(1) #set learning phase
		model_dnn = Sequential()
		model_dnn.add(Dense(80, kernel_initializer='normal', activation='relu', input_dim=dim)) #units in Dense layer as same as the input dim
		model_dnn.add(Dense(1, activation='sigmoid'))
		model_dnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		logger.debug("model compile end >>>>>>")
		return model_dnn

	dim=X.shape[1]	
	# logger.info(X[0].transpose().shape)
	model_dnn = model_build(dim)	
	if config["Evaluation Metrics"]["cross_val_score"]=="True":		
		#return -1
		seed = 7
		np.random.seed(seed)
		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
		cvscores = []
		for train_index, test_index in kfold.split(X, y):
			y_np_array = np.array(y)
			y_train = y_np_array[train_index]
			y_test = y_np_array[test_index]
			model_dnn.fit(X[train_index], y_train, epochs=150, batch_size=10, verbose=0) #fit the model
			scores = model_dnn.evaluate(X[test_index], y_test, verbose=0) #evaluate the model
			cvscores.append(scores[1])
		return np.mean(cvscores)
	
	else:
		model_dnn.fit(X, y, epochs=150, batch_size=100, verbose=0)
		y_predict = model_dnn.predict(X_test)
		eval_metrics_DNN = Evaluation_Metrics.eval_metrics(model_dnn, X, y, y_test, y_predict.round())
		return eval_metrics_DNN

def HDDT():
	#java -cp <path to weka-hddt.jar> weka.classifiers.trees.HTree -U -A -B -t <training file> -T <testing file>
	
	weka_hddt_path="weka-hddt-3-7-1.jar"
	subprocess.call(['java', '-cp', weka_hhdt_path,'weka.classifiers.trees.HTree', '-U', '-A' '-B' '-t', y_predict, y_test])

##To-Do: Add DNN and OLL
####
def rank_classifier(eval_clf_dict, metric_str):
	"""

	"""
	dict_metric_str = {}
	sorted_eval_clf_dict = {}
	#create the dictionary with the metric
	for clf, val in eval_clf_dict.items():
		# print (clf)
		# print (val[metric_str])
		try:
			dict_metric_str[clf] = val[metric_str]
		except KeyError:
			logger.warning("Does not work for classifier: {}".format(clf))
	logger.info("The ranked classifiers on the metric: {}".format(metric_str))
	sorted_dict_metric_str = sorted(((value, key) for (key,value) in dict_metric_str.items()), reverse=True)
	# print (sorted_dict_metric_str)
	for tuple in sorted_dict_metric_str:
		sorted_eval_clf_dict[tuple[1]] = eval_clf_dict[tuple[1]]
	logger.info(sorted_eval_clf_dict)

def classifiers(X,y, X_test, y_test, X_train_balanced=None, y_train_balanced=None):
	logger.info("##### Classifiers #####")
	summary=Features.summary
	summary.write("\n##############\n\nClassifiers Used:\n")
	eval_metrics_per_classifier_dict = {}
	if config["Classification"]["load model"] != "True":
		if X_test is None and config["Evaluation Metrics"]["cross_val_score"] != "True":
			X, X_test, y, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1)
		if config["Evaluation Metrics"]["cross_val_score"] != "True":
			if config["Imbalanced Datasets"]["make_imbalanced_dataset"] == "True":
				X_train_balanced, y_train_balanced = Imbalanced_Dataset.Make_Imbalanced_Dataset(X, y)
	trained_model = None
	if not os.path.exists("Data_Dump/Models"):
		os.makedirs("Data_Dump/Models")
	if config["Classifiers"]["SVM"] == "True":
		if config["Classification"]["load model"] == "True":
			trained_model = joblib.load("Data_Dump/Models/model_svm.pkl")
		eval_SVM, model = SVM(X,y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
		eval_metrics_per_classifier_dict['SVM'] = eval_SVM
		if config["Classification"]["save model"] == "True" and model is not None:
			joblib.dump(model, "Data_Dump/Models/model_svm.pkl")
		summary.write("SVM\n")
	if config["Classifiers"]["RandomForest"] == "True":
		if config["Classification"]["load model"] == "True":
			trained_model = joblib.load("Data_Dump/Models/model_RF.pkl")
		eval_RF, model = RandomForest(X,y, X_test, y_test, X_train_balanced, y_train_balanced,trained_model)
		eval_metrics_per_classifier_dict['RF'] = eval_RF
		if config["Classification"]["save model"] == "True" and model is not None:
			joblib.dump(model, "Data_Dump/Models/model_RF.pkl")
		summary.write("Random Forest\n")
	if config["Classifiers"]["DecisionTree"] == "True":
		if config["Classification"]["load model"] == "True":
			trained_model = joblib.load("Data_Dump/Models/model_DT.pkl")
		eval_DT, model = DecisionTree(X,y, X_test, y_test, None, None, trained_model)
		eval_metrics_per_classifier_dict['Dec_tree'] = eval_DT
		if config["Classification"]["save model"] == "True" and model is not None:
			joblib.dump(model, "Data_Dump/Models/model_DT.pkl")
		summary.write("Decision Tree \n")
	if config["Classifiers"]["GaussianNaiveBayes"] == "True":
		if config["Classification"]["load model"] == "True":
			trained_model = joblib.load("Data_Dump/Models/model_GNB.pkl")
		eval_NB, model = GaussianNaiveBayes(X,y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
		eval_metrics_per_classifier_dict['NB'] = eval_NB
		if config["Classification"]["save model"] == "True" and model is not None:
			joblib.dump(model, "Data_Dump/Models/model_GNB.pkl")
		summary.write("Gaussian Naive Bayes \n")
	if config["Classifiers"]["MultinomialNaiveBayes"] == "True":
		if config["Classification"]["load model"] == "True":
			trained_model = joblib.load("Data_Dump/Models/model_MNB.pkl")
		eval_MNB, model =  MultinomialNaiveBayes(X,y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
		eval_metrics_per_classifier_dict['MNB'] = eval_MNB
		if config["Classification"]["save model"] == "True" and model is not None:
			joblib.dump(model, "Data_Dump/Models/model_MNB.pkl")
		summary.write("Multinomial Naive Bayes \n")
	if config["Classifiers"]["LogisticRegression"] == "True":
		if config["Classification"]["load model"] == "True":
			trained_model = joblib.load("Data_Dump/Models/model_LR.pkl")
		eval_LR, model = LogisticRegression(X,y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
		eval_metrics_per_classifier_dict['LR'] = eval_LR
		if config["Classification"]["save model"] == "True" and model is not None:
			joblib.dump(model, "Data_Dump/Models/model_LR.pkl")
		summary.write("Logistic Regression\n")
	if config["Classifiers"]["ELM"] == "True":
		if config["Classification"]["load model"] == "True":
			trained_model = joblib.load("Data_Dump/Models/model_ELM.pkl")
		eval_elm, model = ELM(X,y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
		eval_metrics_per_classifier_dict['ELM'] = eval_elm
		if config["Classification"]["save model"] == "True" and model is not None:
			joblib.dump(model, "Data_Dump/Models/model_ELM.pkl")
		summary.write("ELM\n")
	if config["Classifiers"]["kNearestNeighbor"] == "True":
		if config["Classification"]["load model"] == "True":
			trained_model = joblib.load("Data_Dump/Models/model_KNN.pkl")
		eval_knn, model = kNearestNeighbor(X,y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
		eval_metrics_per_classifier_dict['KNN'] = eval_knn
		if config["Classification"]["save model"] == "True" and model is not None:
			joblib.dump(model, "Data_Dump/Models/model_KNN.pkl")
		summary.write("kNearest Neighbor\n")
	if config["Classifiers"]["KMeans"] == "True":
		if config["Classification"]["load model"] == "True":
			trained_model = joblib.load("Data_Dump/Models/model_Kmeans.pkl")
		eval_kmeans, model = KMeans(X,y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
		eval_metrics_per_classifier_dict['KMeans'] = eval_kmeans
		if config["Classification"]["save model"] == "True" and model is not None:
			joblib.dump(model, "Data_Dump/Models/model_Kmeans.pkl")
		summary.write("kMeans \n")
	if config["Classifiers"]["Bagging"] == "True":
		if config["Classification"]["load model"] == "True":
			trained_model = joblib.load("Data_Dump/Models/model_bagging.pkl")
		eval_bagging, model = Bagging(X,y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
		eval_metrics_per_classifier_dict['Bagging'] = eval_bagging
		if config["Classification"]["save model"] == "True" and model is not None:
			joblib.dump(model, "Data_Dump/Models/model_bagging.pkl")
		summary.write("Bagging \n")
	if config["Classifiers"]["Boosting"] == "True":
		if config["Classification"]["load model"] == "True":
			trained_model = joblib.load("Data_Dump/Models/model_boosting.pkl")
		eval_boosting, model = Boosting(X,y, X_test, y_test, X_train_balanced, y_train_balanced, trained_model)
		eval_metrics_per_classifier_dict['Boosting'] = eval_boosting
		if config["Classification"]["save model"] == "True" and model is not None:
			joblib.dump(model, "Data_Dump/Models/model_boosting.pkl")
		summary.write("Boosting \n")
	if config["Classifiers"]["DNN"] == "True":
		if config["Classification"]["load model"] == "True":
			trained_model = joblib.load("Data_Dump/Models/model_DNN.pkl")
		eval_dnn = DNN(X,y, X_test, y_test, X_train_balanced, y_train_balanced)
		eval_metrics_per_classifier_dict['DNN'] = eval_dnn
		summary.write("DNN \n")
	logger.info(eval_metrics_per_classifier_dict)
	if config["Classification"]["Rank Classifiers"] == "True":
		rank_classifier(eval_metrics_per_classifier_dict, config["Classification"]["rank on metric"])

def fit_MNB(X,y):
	mnb=MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
	mnb.fit(X,y)
	logger.info("MNB >>>>>>>")
	joblib.dump(mnb,"Data_Dump/Emails_Training/MNB_model.pkl")
