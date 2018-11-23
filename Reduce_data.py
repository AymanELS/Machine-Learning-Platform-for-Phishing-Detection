import os
import sys
from sklearn.externals import joblib
import configparser
import numpy as np
import logging
import argparse
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
import random

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('--matrix', type=str, required=True,
                    help='path to the unprocessed pickle files.')
parser.add_argument('--label', type=str, required=True,
                    help='path to the label files.')
parser.add_argument('--vectorizer', type=str, required=True,
                    help='path to the vectorizer pickle files.')
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
	X_train, y_train, vectorizer = load_datasets(args.matrix, args.label, args.vectorizer)
	
	number_total = len(y_train)
	print("Dataset Size:{}".format(number_total))
	if args.mode == 0:
		number_to_extract =  ((args.percentage_to_extract)/100)*number_total
	if args.mode == 1:
		number_to_extract = args.number_of_instances
	if number_total < number_to_extract:
		print('Sample size exceeded population size.')
		exit()
	samples = random.sample(range(1, number_total), number_to_extract)
	print("Transforming back the sparse matrix into dictionary of features (feature vectors):")
	feature_vector=vectorizer.inverse_transform(X_train)
	list_Features=vectorizer.get_feature_names()

	new_feature_vector=[{} for i in range(number_to_extract)]
	print("Extracting {} instances from the dataset".format(number_to_extract))
	new_labels = []#y_train[:number_to_extract]
	for i in range(number_to_extract):
		index = samples[i]
		new_labels.append(y_train[index])
		for feature in list_Features:
			if feature in feature_vector[index].keys():
				new_feature_vector[i][feature]=feature_vector[index][feature]
			else:
				new_feature_vector[i][feature]=0

	with open(os.path.join(args.output_dir, args.dataset+str(number_to_extract)+'.txt'),'w') as f:
		for i in new_feature_vector:
			f.write("{}\n".format(i))

	X_new = vectorizer.transform(new_feature_vector)
	
	joblib.dump(vectorizer, os.path.join(args.output_dir, args.dataset+str(number_to_extract)+'_vectorizer.pkl'))
	joblib.dump(X_new, os.path.join(args.output_dir, args.dataset+str(number_to_extract)+'_X_train_unprocessed.pkl'))
	joblib.dump(new_labels, os.path.join(args.output_dir, args.dataset+str(number_to_extract)+'_y_train.pkl'))

if __name__ == '__main__':
	main()
