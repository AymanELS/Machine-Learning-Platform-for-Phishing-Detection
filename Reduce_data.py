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

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('--matrix', type=str, required=True, nargs=2,
                    help='a list of paths to the unprocessed pickle files.')
parser.add_argument('--labels', type=str, required=True, nargs=2,
                    help='list of paths to the label files.')
parser.add_argument('--vectorizer', type=str, required=True, nargs=2,
                    help='a list of paths to the vectorizer pickle files.')
parser.add_argument('--mode', type=int, required=True, default=1, 
					help='modes of extraction; if 0 - extract the percentage of URLs specified; if 1 - extract the number of URLs specified')
parser.add_argument('--percentage_to_extract', type=int, default=50,  
					help='takes a percentage of instances to be extracted.')
parser.add_argument('--number_of_instances', type=int, default=100,  
					help='takes a number of instances to be extracted.')
parser.add_argument('--dataset', type=str, required=True, 
					help='name of the dataset')
parser.add_argument('--output_dir', type=str, required=True,
                    help='path to output directory for dumping the generated pickle files.')

args = parser.parse_args()

def load_datasets(matrix, labels, vectorizer_path):
    X_train=joblib.load(matrix)
    y_train=joblib.load(labels)
    vectorizer=joblib.load(vectorizer_path)
    return X_train, y_train, vectorizer

def main():
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	print("Loading dataset")
    X_train1, y_train1, vectorizer1 = load_datasets(args.matrix[0], args.labels[0], args.vectorizer[0])
	
	# path1_matrix= '/home/avisha/Feature_Extraction_Platform/Data_Dump_openphish_2/URLs_Training/X_train_unprocessed.pkl'#input("\nEnter path to feature matrix (X_train_unprocessed) of dataset 1: ")
	# path1_labels= '/home/avisha/Feature_Extraction_Platform/Data_Dump_openphish_2/URLs_Training/y_train.pkl'#input("\nEnter path to labels (y_train) of dataset 1: ")
	# path1_vectorizer= '/home/avisha/Feature_Extraction_Platform/Data_Dump_openphish_2/URLs_Training/vectorizer.pkl' #input("\nEnter path to vectorizer.pkl of dataset1: ") 


	# X_train1=joblib.load(path1_matrix)
	# y_train1=joblib.load(path1_labels)
	# vectorizer1=joblib.load(path1_vectorizer)

	number_total1 = len(y_train1)
	if args.mode == 0:
		number_to_extract =  ((args.percentage_to_extract)/100)*number_total1
	if args.mode == 1:
		number_to_extract = args.number_of_instances

	#Transforming back the sparse matrix into dictionary of features (feature vectors):
	feature_vector1=vectorizer1.inverse_transform(X_train1)
	list_Features_1=vectorizer1.get_feature_names()

	new_feature_vector=[{} for i in range(number_to_extract)]
	print ("Extracting {} instances from the dataset".format(number_to_extract))
	for feature in list_Features_1:
		#print("feature: {}".format(feature))
		for i in range(number_to_extract):
			if feature in feature_vector1[i].keys():
				new_feature_vector[i][feature]=feature_vector1[i][feature]
			else:
				new_feature_vector[i][feature]=0

	with open(os.path.join(args.output_dir, args.dataset+str(number_to_extract)+'.txt'),'w') as f:
		for i in new_feature_vector:
			f.write("{}\n".format(i))

	X_new = vectorizer1.transform(new_feature_vector)

	# new_labels = [y_train1[i] for i in range(number_to_extract)]
	new_labels = y_train1[:number_to_extract]

	joblib.dump(vectorizer1, os.path.join(args.output_dir, dataset+str(number_to_extract)+'_vectorizer.pkl'))
	joblib.dump(X_new,	os.path.join(args.output_dir, dataset+str(number_to_extract)+'_X_train_unprocessed.pkl'))
	joblib.dump(new_labels, os.path.join(args.output_dir, dataset+str(number_to_extract)+'_y_train.pkl')

if __name__ == '__main__':
	main()