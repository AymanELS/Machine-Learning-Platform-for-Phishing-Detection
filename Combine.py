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


config=configparser.ConfigParser()
config.read('Config_file.ini')


def main():

    if not os.path.exists("Data_Dump/Feature_Combine"):
        os.makedirs("Data_Dump/Feature_Combine")
    

        

    path1_matrix= '/home/avisha/Feature_Extraction_Platform/Data_Dump/Feature_Reduce/openphish_100_features.pkl'#input("\nEnter path to feature matrix (X_train_unprocessed) of dataset 1: ")
    path1_labels= '/home/avisha/Feature_Extraction_Platform/Data_Dump/Feature_Reduce/openphish_100_labels.pkl'#input("\nEnter path to labels (y_train) of dataset 1: ")
    path1_vectorizer= '/home/avisha/Feature_Extraction_Platform/Data_Dump/Feature_Reduce/openphish_100_vectorizer.pkl' #input("\nEnter path to vectorizer.pkl of dataset1: ")


    # path1_matrix= '/home/avisha/Feature_Extraction_Platform/Data_Dump_openphish_2/URLs_Training/X_train_unprocessed.pkl'#input("\nEnter path to feature matrix (X_train_unprocessed) of dataset 1: ")
    # path1_labels= '/home/avisha/Feature_Extraction_Platform/Data_Dump_openphish_2/URLs_Training/y_train.pkl'#input("\nEnter path to labels (y_train) of dataset 1: ")
    # path1_vectorizer= '/home/avisha/Feature_Extraction_Platform/Data_Dump_openphish_2/URLs_Training/vectorizer.pkl' #input("\nEnter path to vectorizer.pkl of dataset1: ")

    path2_matrix= '/home/avisha/Feature_Extraction_Platform/Data_Dump_alexa_login/URLs_Training/X_train_unprocessed.pkl'#input("\nEnter path to feature matrix (X_train_unprocessed) of dataset 2: ")
    path2_labels= '/home/avisha/Feature_Extraction_Platform/Data_Dump_alexa_login/URLs_Training/y_train.pkl' #input("\nEnter path to labels (y_train) of dataset 2: ")
    path2_vectorizer= '/home/avisha/Feature_Extraction_Platform/Data_Dump_alexa_login/URLs_Training/vectorizer.pkl' #input("\nEnter path to vectorizer.pkl of dataset2: ")

    X_train1=joblib.load(path1_matrix)
    y_train1=joblib.load(path1_labels)
    vectorizer1=joblib.load(path1_vectorizer)

    X_train2=joblib.load(path2_matrix)
    y_train2=joblib.load(path2_labels)
    vectorizer2=joblib.load(path2_vectorizer)

    #X_train1=joblib.load("Data_Dump/Feature_Combine/X_train_unprocessed1.pkl")
    #y_train1=joblib.load("Data_Dump/Feature_Combine/y_train1.pkl")
    #vectorizer1=joblib.load("Data_Dump/Feature_Combine/vectorizer1.pkl")

    # X_train2=joblib.load("Data_Dump/Feature_Combine/X_train_unprocessed2.pkl")
    # y_train2=joblib.load("Data_Dump/Feature_Combine/y_train2.pkl")
    # vectorizer2=joblib.load("Data_Dump/Feature_Combine/vectorizer2.pkl")

    number_legitimate1=y_train1.count(0)
    number_legitimate2=y_train2.count(0)
    number_phish1=y_train1.count(1)
    number_phish2=y_train2.count(1)

    
    #Transforming back the sparse matrix into dictionnary of features (feature vectors):
    feature_vector1=vectorizer1.inverse_transform(X_train1)
    feature_vector2=vectorizer2.inverse_transform(X_train2)

    with open("Data_Dump/Feature_Combine/feature_vector_1.txt",'w') as f:
        for i in feature_vector1:
            f.write("{}\n".format(i))
    with open("Data_Dump/Feature_Combine/feature_vector_2.txt",'w') as f:
        for i in feature_vector2:
            f.write("{}\n".format(i))

    list_Features_1=vectorizer1.get_feature_names()   
    list_Features_2=vectorizer2.get_feature_names()

    with open("Data_Dump/Feature_Combine/feature_list_1.txt",'w') as f:
        for i in list_Features_1:
            f.write("%s\n" %i)
    with open("Data_Dump/Feature_Combine/feature_list_2.txt",'w') as f:
        for i in list_Features_2:
            f.write("%s\n" %i)

    # remove one-hot encoding from feature vectors
    feature_vector1_original=[]
    for feature in list_Features_1:
        if '=' in feature:
            feature_name=feature.split("=")[0]
        else:
            feature_name=feature
        if feature_name not in feature_vector1_original:
            feature_vector1_original.append(feature_name)

    feature_vector2_original=[]
    for feature in list_Features_2:
        if '=' in feature:
            feature_name=feature.split("=")[0]
        else:
            feature_name=feature
        if feature_name not in feature_vector2_original:        
            feature_vector2_original.append(feature_name)
    #with open("Data_Dump/Feature_Combine/feature_vector1_original.txt",'w') as f:
    #    for i in feature_vector1_original:
    #        f.write("%s\n" %i)
    #with open("Data_Dump/Feature_Combine/feature_vector2_original.txt",'w') as f:
    #    for i in feature_vector2_original:
    #        f.write("%s\n" %i)

    # list of common feature without one hot encoding
    common_features=[]
    for feature in feature_vector1_original:
        if feature in feature_vector2_original:
            if feature not in common_features:
                common_features.append(feature)

    for feature in feature_vector2_original:
        if feature in feature_vector1_original:
            if feature not in common_features:
                common_features.append(feature)

    common_features=sorted(common_features)

    # new list of features with one-hot encoding:
    new_feature_list=[]
    for feature in list_Features_1:
        if "=" in feature:
            feature_name=feature.split("=")[0]
        else:
            feature_name=feature
        if feature_name in common_features:
            if feature not in new_feature_list:
                new_feature_list.append(feature)
    for feature in list_Features_2:
        if "=" in feature:
            feature_name=feature.split("=")[0]
        else:
            feature_name=feature
        if feature_name in common_features:
            if feature not in new_feature_list:
                new_feature_list.append(feature) 

    new_feature_list=sorted(new_feature_list)
    with open("Data_Dump/Feature_Combine/new_feature_list.txt",'w') as f:
        for i in new_feature_list:
            f.write("%s\n" %i)
    # #combined list of features:
    # common_Features=sorted(feature_vector1_original + list(set(feature_vector1_original)-set(feature_vector2_original)))

    #combined list of features:
    #all_Features=feature_vector_1
    #for feature in feature_vector_2:
    #    if feature not in all_Features:
    #        all_Features.append(feature)

    #all_Features=sorted(all_Features)

    
    #build new list of features
    new_feature_vector=[{} for i in range(len(y_train1+y_train2))]


    for feature in new_feature_list:
        #print("feature: {}".format(feature))
        for i in range(number_legitimate1):
            if feature in feature_vector1[i].keys():
                new_feature_vector[i][feature]=feature_vector1[i][feature]
            else:
                new_feature_vector[i][feature]=0
        for i in range(number_legitimate2):
            if feature in feature_vector2[i].keys():
                new_feature_vector[number_legitimate1+i][feature]=feature_vector2[i][feature]
            else:
                new_feature_vector[number_legitimate1+i][feature]=0

        for i in range(number_phish1):
            if feature in feature_vector1[number_legitimate1+i].keys():
                new_feature_vector[number_legitimate1+number_legitimate2+i][feature]=feature_vector1[number_legitimate1+i][feature]
            else:
                new_feature_vector[number_legitimate1+number_legitimate2+i][feature]=0
        for i in range(number_phish2):
            if feature in feature_vector2[number_legitimate2+i].keys():
                new_feature_vector[number_legitimate1+number_legitimate2+number_phish1+i][feature]=feature_vector2[number_legitimate2+i][feature]
            else:
                new_feature_vector[number_legitimate1+number_legitimate2+number_phish1+i][feature]=0


    with open("Data_Dump/Feature_Combine/new_feature_vector.txt",'w') as f:
        for i in new_feature_vector:
            f.write("{}\n".format(i))
    #create new feature vector:
    #new_feature_vector=feature_vector1_legit + feature_vector2_legit + feature_vector1_phish + feature_vector2_phish

          
    #getting the feature labels from the vectorizers:
    #new_transformer=([("vectorizer1",vectorizer1),("Vectorizer2",vectorizer2)])

    #Applying new transformation to both feature vector and create new sparse matrices
     
    
    vectorizer=DictVectorizer()
    #vectorizer=FeatureUnion([("vectorizer1",vectorizer1),("vectorizer2",vectorizer2)])
    vectorizer.fit(new_feature_vector)

    list_Features_new=vectorizer.get_feature_names()   
      
    with open("Data_Dump/Feature_Combine/feature_vector_new.txt",'w') as f:
        for i in list_Features_new:
            f.write("%s\n" %i)

    print("legitimate count 1: {}".format(number_legitimate1))
    print("legitimate count 2: {}".format(number_legitimate2))
    print("phish count 1: {}".format(number_phish1))
    print("phish count 2: {}".format(number_phish2))
    
    
    X_train=vectorizer.transform(new_feature_vector)
    print("X_1 Shape: {}".format(X_train1.shape))
    print("X_2 Shape: {}".format(X_train2.shape))
    print("X_new Shape: {}".format(X_train.shape))
    y_train=[0 for i in range(number_legitimate1+number_legitimate2)] + [1 for i in range(y_train1.count(1) + y_train2.count(1))]
    print("y length: {}".format(len(y_train)))
    #y_train=[y_train1[i] for i in range(number_legitimate1)] + [y_train2[i] for i in range(number_legitimate2)] + [y_train1[i + number_legitimate1] for i in range(len(y_train1) - number_legitimate1)] + [y_train2[i + number_legitimate2] for i in range(len(y_train2) - number_legitimate2)]
    # concatenate sparse matrices vertically
    #X_train_legit= vstack([np.split(X_train1,number_legitimate1)[0], np.split(X_train2, number_legitimate2)[0]])
    #X_train_phish= vstack([np.split(X_train1,number_legitimate1)[0], np.split(X_train2, number_legitimate2)[0]])
    #X_train=vstack([X_train_legit,X_train_phish])
    print("new legitimate count: {}".format(y_train.count(0)))
    print("new phish count: {}".format(y_train.count(1)))


    joblib.dump(X_train, "Data_Dump/Feature_Combine/X_combined_unprocessed.pkl")
    joblib.dump(y_train, "Data_Dump/Feature_Combine/y_combined_unprocessed.pkl")
    joblib.dump(vectorizer, "Data_Dump/Feature_Combine/vectorizer_combined.pkl")

    

if __name__ == "__main__":
    # execute only if run as a
    main()
