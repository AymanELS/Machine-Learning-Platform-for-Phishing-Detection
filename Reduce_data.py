import os
import sys
from sklearn.externals import joblib
#import User_options
import configparser
import numpy as np
#from collections import deque
import logging
import argparse
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion


def main():
	if not os.path.exists('Data_Dump/Feature_Reduce'):
		os.makedirs('Data_Dump/Feature_Reduce')

	path1_matrix= '/home/avisha/Feature_Extraction_Platform/Data_Dump_openphish_2/URLs_Training/X_train_unprocessed.pkl'#input("\nEnter path to feature matrix (X_train_unprocessed) of dataset 1: ")
	path1_labels= '/home/avisha/Feature_Extraction_Platform/Data_Dump_openphish_2/URLs_Training/y_train.pkl'#input("\nEnter path to labels (y_train) of dataset 1: ")
	path1_vectorizer= '/home/avisha/Feature_Extraction_Platform/Data_Dump_openphish_2/URLs_Training/vectorizer.pkl' #input("\nEnter path to vectorizer.pkl of dataset1: ") 


	X_train1=joblib.load(path1_matrix)
	y_train1=joblib.load(path1_labels)
	vectorizer1=joblib.load(path1_vectorizer)

	number_legitimate1 = y_train1.count(0)

	number_to_extract = 100 

	#Transforming back the sparse matrix into dictionary of features (feature vectors):
	feature_vector1=vectorizer1.inverse_transform(X_train1)

	list_Features_1=vectorizer1.get_feature_names()

	new_feature_vector=[{} for i in range(number_to_extract)]
	for feature in list_Features_1:
		#print("feature: {}".format(feature))
		for i in range(number_to_extract):
			if feature in feature_vector1[i].keys():
				new_feature_vector[i][feature]=feature_vector1[i][feature]
			else:
				new_feature_vector[i][feature]=0

	with open("Data_Dump/Feature_Reduce/openphish_100_features.txt",'w') as f:
		for i in new_feature_vector:
			f.write("{}\n".format(i))

	X_new = vectorizer1.transform(new_feature_vector)

	# new_labels = [y_train1[i] for i in range(number_to_extract)]
	new_labels = y_train1[:number_to_extract]

	joblib.dump(vectorizer1, 'Data_Dump/Feature_Reduce/openphish_100_vectorizer.pkl')
	joblib.dump(X_new,'Data_Dump/Feature_Reduce/openphish_100_features.pkl')
	joblib.dump(new_labels, 'Data_Dump/Feature_Reduce/openphish_100_labels.pkl')

if __name__ == '__main__':
	main()