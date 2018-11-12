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
parser.add_argument('--output_dir', type=str, required=True,
                    help='path to output directory for dumping the generated pickle files.')

args = parser.parse_args()

config=configparser.ConfigParser()
config.read('Config_file.ini')

def load_datasets(matrix, labels, vectorizer_path):
    X_train=joblib.load(matrix)
    y_train=joblib.load(labels)
    vectorizer=joblib.load(vectorizer_path)
    return X_train, y_train, vectorizer

def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Loading dataset")
    X_train_first_dataset, y_train_first_dataset, first_dataset_vectorizer = load_datasets(args.matrix[0], args.labels[0], args.vectorizer[0])
    X_train_second_dataset, y_train_second_dataset, second_dataset_vectorizer = load_datasets(args.matrix[1], args.labels[1], args.vectorizer[1])

    first_dataset_legit_size=y_train_first_dataset.count(0)
    second_dataset_legit_size=y_train_second_dataset.count(0)

    first_dataset_phish_size=y_train_first_dataset.count(1)
    second_dataset_phish_size=y_train_second_dataset.count(1)
    print("Transforming back the sparse matrix into dictionnary of features (feature vectors)")
    first_dataset_feature_vector=first_dataset_vectorizer.inverse_transform(X_train_first_dataset)
    second_dataset_feature_vector=second_dataset_vectorizer.inverse_transform(X_train_second_dataset)

    with open(os.path.join(args.output_dir, "feature_vector_1.txt"),'w') as f:
        for i in first_dataset_feature_vector:
            f.write("{}\n".format(i))
    with open(os.path.join(args.output_dir, "feature_vector_2.txt"),'w') as f:
        for i in second_dataset_feature_vector:
            f.write("{}\n".format(i))

    first_dataset_list_features=first_dataset_vectorizer.get_feature_names()
    second_dataset_list_features=second_dataset_vectorizer.get_feature_names()

    with open(os.path.join(args.output_dir, "feature_list_1.txt"),'w') as f:
        for i in first_dataset_list_features:
            f.write("%s\n" %i)
    with open(os.path.join(args.output_dir, "feature_list_2.txt"),'w') as f:
        for i in second_dataset_list_features:
            f.write("%s\n" %i)

    print("remove one-hot encoding from feature vectors")
    first_dataset_feature_vector_original=[]
    for feature in first_dataset_list_features:
        if '=' in feature:
            feature_name=feature.split("=")[0]
        else:
            feature_name=feature
        if feature_name not in first_dataset_feature_vector_original:
            first_dataset_feature_vector_original.append(feature_name)

    second_dataset_feature_vector_original=[]
    for feature in second_dataset_list_features:
        if '=' in feature:
            feature_name=feature.split("=")[0]
        else:
            feature_name=feature
        if feature_name not in second_dataset_feature_vector_original:
            second_dataset_feature_vector_original.append(feature_name)

    common_features=[]
    for feature in first_dataset_feature_vector_original:
        if feature in second_dataset_feature_vector_original:
            if feature not in common_features:
                common_features.append(feature)

    for feature in second_dataset_feature_vector_original:
        if feature in first_dataset_feature_vector_original:
            if feature not in common_features:
                common_features.append(feature)

    common_features=sorted(common_features)

    print("new list of features with one-hot encoding:")
    new_feature_list=[]
    for feature in first_dataset_list_features:
        if "=" in feature:
            feature_name=feature.split("=")[0]
        else:
            feature_name=feature
        if feature_name in common_features:
            if feature not in new_feature_list:
                new_feature_list.append(feature)
    for feature in second_dataset_list_features:
        if "=" in feature:
            feature_name=feature.split("=")[0]
        else:
            feature_name=feature
        if feature_name in common_features:
            if feature not in new_feature_list:
                new_feature_list.append(feature)

    new_feature_list=sorted(new_feature_list)
    with open(os.path.join(args.output_dir, "new_feature_list.txt"),'w') as f:
        for i in new_feature_list:
            f.write("%s\n" %i)

    print("build new list of features")
    new_feature_vector=[{} for i in range(len(y_train_first_dataset+y_train_second_dataset))]

    for feature in new_feature_list:
        #print("feature: {}".format(feature))
        for i in range(first_dataset_legit_size):
            if feature in first_dataset_feature_vector[i].keys():
                new_feature_vector[i][feature]=first_dataset_feature_vector[i][feature]
            else:
                new_feature_vector[i][feature]=0
        for i in range(second_dataset_legit_size):
            if feature in second_dataset_feature_vector[i].keys():
                new_feature_vector[first_dataset_legit_size+i][feature]=second_dataset_feature_vector[i][feature]
            else:
                new_feature_vector[first_dataset_legit_size+i][feature]=0

        for i in range(first_dataset_phish_size):
            if feature in first_dataset_feature_vector[first_dataset_legit_size+i].keys():
                new_feature_vector[first_dataset_legit_size+second_dataset_legit_size+i][feature]=first_dataset_feature_vector[first_dataset_legit_size+i][feature]
            else:
                new_feature_vector[first_dataset_legit_size+second_dataset_legit_size+i][feature]=0
        for i in range(second_dataset_phish_size):
            if feature in second_dataset_feature_vector[second_dataset_legit_size+i].keys():
                new_feature_vector[first_dataset_legit_size+second_dataset_legit_size+first_dataset_phish_size+i][feature]=second_dataset_feature_vector[second_dataset_legit_size+i][feature]
            else:
                new_feature_vector[first_dataset_legit_size+second_dataset_legit_size+first_dataset_phish_size+i][feature]=0


    with open(os.path.join(args.output_dir, "new_feature_vector.txt"),'w') as f:
        for i in new_feature_vector:
            f.write("{}\n".format(i))

    print("Applying new transformation to both feature vector and create new sparse matrices")
    vectorizer=DictVectorizer()
    vectorizer.fit(new_feature_vector)

    list_Features_new=vectorizer.get_feature_names()

    with open(os.path.join(args.output_dir, "feature_vector_new.txt"),'w') as f:
        for i in list_Features_new:
            f.write("%s\n" %i)

    print("legitimate count 1: {}".format(first_dataset_legit_size))
    print("legitimate count 2: {}".format(second_dataset_legit_size))
    print("phish count 1: {}".format(first_dataset_phish_size))
    print("phish count 2: {}".format(second_dataset_phish_size))


    X_train=vectorizer.transform(new_feature_vector)
    print("X_1 Shape: {}".format(X_train_first_dataset.shape))
    print("X_2 Shape: {}".format(X_train_second_dataset.shape))
    print("X_new Shape: {}".format(X_train.shape))
    y_train=[0 for i in range(first_dataset_legit_size+second_dataset_legit_size)] + [1 for i in range(y_train_first_dataset.count(1) + y_train_second_dataset.count(1))]
    print("y length: {}".format(len(y_train)))
    print("new legitimate count: {}".format(y_train.count(0)))
    print("new phish count: {}".format(y_train.count(1)))


    joblib.dump(X_train, os.path.join(args.output_dir, "X_combined_unprocessed.pkl"))
    joblib.dump(y_train, os.path.join(args.output_dir, "y_combined_unprocessed.pkl"))
    joblib.dump(vectorizer, os.path.join(args.output_dir, "vectorizer_combined.pkl"))

if __name__ == "__main__":
    # execute only if run as a
    main()
