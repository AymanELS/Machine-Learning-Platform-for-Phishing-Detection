import os
import sys
import ast
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
import pickle

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
	# print (len(lines))
	# print (len(feature_vector_as_strings))

	##########################################
	list_feature_vector_as_strings = []
	for i, string in enumerate(feature_vector_as_strings):
		split_feature = string.split()
		# print (i, len(split_feature)) **check the 97 length feature
		list_feature_vector_as_strings.append(split_feature)

	# print (len(list_feature_vector_as_strings))
	# print (len(list_feature_vector_as_strings[100]))

	###############################################
	dict_feature_vector_as_strings = []
	for list_ in list_feature_vector_as_strings:
		dict = {}
		# count = 0
		for i,feature in enumerate(list_):
			item = feature.split(':')
			# print (i, item)
			if item[1].isnumeric():
				dict[(item[0])[1:-1]] = int(item[1])
			elif is_float(item[1]):
				dict[(item[0])[1:-1]] = float(item[1])
			else:
				dict[(item[0])[1:-1]] = (item[1])[1:-1]
		# if len(dict.keys()) > 96:
		# 	count += 1
		dict_feature_vector_as_strings.append(dict)

	# print (len(dict_feature_vector_as_strings))
	# print (len(dict_feature_vector_as_strings[100].keys()))
	# print (count)
	return (dict_feature_vector_as_strings)

##########################################
def convert_to_vectorizer(list_of_feature_vectors, dataset_name):
	# for i in range(len(list_of_feature_vectors)):
	# 	print (len(list_of_feature_vectors[i].keys()))
	vectorizer = DictVectorizer()
	vectorizer.fit(list_of_feature_vectors)
	# print (vectorizer.get_feature_names())
	sparse_matrix_features = vectorizer.transform(list_of_feature_vectors)
	print (sparse_matrix_features.shape)	
	joblib.dump(sparse_matrix_features,"Data_Dump/Converted_Data/URLs_Training/X_train_unprocessed" + dataset_name + ".pkl")
	joblib.dump(vectorizer,"Data_Dump/Converted_Data/URLs_Training/vectorizer" + dataset_name + ".pkl")



if __name__ == '__main__':
	if not os.path.exists("Data_Dump/Converted_Data/URLs_Training/"):
		os.makedirs("Data_Dump/Converted_Data/URLs_Training/")

	path_to_features = '/home/avisha/Feature_Extraction_Platform/Data_Dump_openphish_2/URLs_Backup/url_phish_openphish.txt_feature_vector.txt'	
	dataset_name = 'openphish'
	
	dict_feature_vectors_openphish = convert_from_text_todict(path_to_features)
	convert_to_vectorizer(dict_feature_vectors_openphish, dataset_name)

	path_to_features = '/home/avisha/Feature_Extraction_Platform/Data_Dump_alexa_login/URLs_Backup/dataset_alexa_login_train_legit_alexa_login.txt_feature_vector.txt'	
	dataset_name = 'alexa_login'
	
	dict_feature_vectors_openphish = convert_from_text_todict(path_to_features)
	convert_to_vectorizer(dict_feature_vectors_openphish, dataset_name)


