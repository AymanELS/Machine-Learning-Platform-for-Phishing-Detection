import os
import sys
import ast
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
import pickle
import argparse

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('--features', type=str, required=True,
                    help='path to the human-readable feature file.')
parser.add_argument('--output_dir', type=str, required=True,
                    help='path to output directory for dumping the generated pickle files.')
parser.add_argument('--dataset_name', type=str, required=True,
                    help='A unique identifier for the dataset to name the generated files based on that')

args = parser.parse_args()

def is_float(value):
	try:
		float(value)
		return True
	except:
		return False

def convert_from_text_todict(path_to_features_readable):
	with open(path_to_features_readable, 'r') as i_f:
		feature_vector_as_strings = []
		lines = i_f.readlines()
		for line in lines:
			if line != '\n' and line.startswith('URL:') == False:
				feature_vector_as_strings.append(line)
	##########################################
	list_feature_vector_as_strings = []
	for i, string in enumerate(feature_vector_as_strings):
		split_feature = string.split()
		list_feature_vector_as_strings.append(split_feature)
	###############################################
	dict_feature_vector_as_strings = []
	for list_ in list_feature_vector_as_strings:
		dict = {}
		for i,feature in enumerate(list_):
			item = feature.split(':')
			if item[1].isnumeric():
				dict[(item[0])[1:-1]] = int(item[1])
			elif is_float(item[1]):
				dict[(item[0])[1:-1]] = float(item[1])
			else:
				dict[(item[0])[1:-1]] = (item[1])[1:-1]
		dict_feature_vector_as_strings.append(dict)
	return (dict_feature_vector_as_strings)

##########################################
def convert_to_vectorizer(list_of_feature_vectors, dataset_name):
	vectorizer = DictVectorizer()
	vectorizer.fit(list_of_feature_vectors)
	sparse_matrix_features = vectorizer.transform(list_of_feature_vectors)
	print ("Vectorizer shape:{}".format(sparse_matrix_features.shape))
	joblib.dump(sparse_matrix_features,
		    os.path.join(args.output_dir, "X_train_unprocessed_" + dataset_name + ".pkl"))
	joblib.dump(vectorizer,
		    os.path.join(args.output_dir, "vectorizer_" + dataset_name + ".pkl"))

if __name__ == '__main__':
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	dict_feature_vectors_openphish = convert_from_text_todict(args.features)
	convert_to_vectorizer(dict_feature_vectors_openphish, args.dataset_name)
