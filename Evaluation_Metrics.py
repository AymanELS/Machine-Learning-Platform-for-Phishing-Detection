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
from imblearn.metrics import geometric_mean_score
#from imblearn.metrics import Balanced_accuracy_score
#import User_options
import sklearn
import configparser
#from collections import deque
import Features
import tensorflow as tf

config=configparser.ConfigParser()
config.read('Config_file.ini')

def Confusion_matrix(y_test, y_predict):
		confusion_matrix=sklearn.metrics.confusion_matrix(y_test, y_predict)
		tn, fp, fn, tp=confusion_matrix.ravel()
		print("Confusion Matrix: TN, FP, FN, TP")
		print(tn, fp, fn, tp)

def Confusion_matrix2(y_test, y_predict):
		sess = tf.Session()
		with sess.as_default():
			y_test=y_test.eval()
			y_predict=y_predict.eval()
		confusion_matrix=sklearn.metrics.confusion_matrix(y_test, y_predict)
		tn, fp, fn, tp=confusion_matrix.ravel()
		print("Confusion Matrix: TN, FP, FN, TP")
		print(tn, fp, fn, tp)

def Matthews_corrcoef(y_test, y_predict):
		Mcc=sklearn.metrics.matthews_corrcoef(y_test, y_predict)
		print("Matthews_CorrCoef")
		print(Mcc)
		#return Mcc

def ROC_AUC(y_test, y_predict):
		ROC_AUC=sklearn.metrics.roc_auc_score(y_test, y_predict)
		print("ROC_AUC")
		print(ROC_AUC)
		#return ROC_AUC

def Precision(y_test, y_predict):
		precision=sklearn.metrics.precision_score(y_test, y_predict)
		print("Precision")
		print(precision)
		#return precision

def Recall(y_test, y_predict):
		recall=sklearn.metrics.recall_score(y_test, y_predict)
		print("Recall")
		print(recall)
		#return Recall

def F1_score(y_test, y_predict):
		f1_score=sklearn.metrics.f1_score(y_test, y_predict)
		print("F1_score")
		print(f1_score)
		#return F1_score

def Cross_validation(clf, X, y):
		score = cross_val_score(clf, X, y, cv=10)
		print("10 fold Cross_Validation")
		print(score.mean())

def Homogenity(y_test,y_predict):
		homogenity=sklearn.metrics.homogeneity_score(y_test,y_predict)
		print("Homogenity")
		print(homogenity)

def Completeness(y_test,y_predict):
		completeness=sklearn.metrics.completeness_score(y_test,y_predict)
		print("Completeness")
		print(completeness)
def V_measure(y_test,y_predict):
		v_measure=sklearn.metrics.v_measure_score(y_test,y_predict)
		print("V_measure")
		print(v_measure)


def Geomteric_mean_score(y_test,y_predict):
		g_mean=geometric_mean_score(y_test,y_predict)
		print("G_mean")
		print(g_mean)

def Balanced_accuracy_score(y_test,y_predict):
		b_accuracy=sklearn.metrics.balanced_accuracy_score(y_test,y_predict)
		print("Balanced_accuracy_score")
		print(b_accuracy)

def eval_metrics(clf, X, y, y_test, y_predict):
	summary=Features.summary
	summary.write("\n\nEvaluation metrics used:\n")
	summary.write("\n\n Supervised metrics:\n")
	if config["Evaluation Metrics"]["Confusion_matrix"] == "True":
		Confusion_matrix(y_test, y_predict)
		summary.write("Confusion_matrix\n")
	if config["Evaluation Metrics"]["Matthews_corrcoef"] == "True":
		Matthews_corrcoef(y_test, y_predict)
		summary.write("Matthews_corrcoef\n")
	if config["Evaluation Metrics"]["ROC_AUC"] == "True":
		ROC_AUC(y_test, y_predict)
		summary.write("ROC_AUC\n")
	if config["Evaluation Metrics"]["Precision"] == "True":
		Precision(y_test, y_predict)
		summary.write("Precision\n")
	if config["Evaluation Metrics"]["Recall"] == "True":
		Recall(y_test, y_predict)
		summary.write("Recall\n")
	if config["Evaluation Metrics"]["F1_score"] == "True":
		F1_score(y_test, y_predict)
		summary.write("F1_score\n")
	#if config["Evaluation Metrics"]["Cross_validation"] == "True":
	#	Cross_validation(clf, X, y)
	#	summary.write("Cross_validation\n")
	if config["Evaluation Metrics"]["Geomteric_mean_score"] == "True":
		Geomteric_mean_score(y_test,y_predict)
		summary.write("Geomteric_mean_score\n")
	#if config["Evaluation Metrics"]["Balanced_accuracy_score"] == "True":
	#	Balanced_accuracy_score(y_test,y_predict)
	#	summary.write("Balanced_accuracy_score\n")
	#	# write results to summary
	
def eval_metrics_cluster(y_test, y_predict):
	summary=Features.summary
	summary.write("\n\nEvaluation metrics used:\n")
	summary.write("\n\n clustering metrics:\n")
	if config["Evaluation Metrics"]["Homogenity"] == "True":
		Homogenity(y_test, y_predict)
		summary.write("Homogenity\n")
	if config["Evaluation Metrics"]["Completeness"] == "True":
		Completeness(y_test, y_predict)
		summary.write("Completeness\n")
	if config["Evaluation Metrics"]["V_measure"] == "True":
		V_measure(y_test,y_predict)
		summary.write("V_measure\n")
	
		