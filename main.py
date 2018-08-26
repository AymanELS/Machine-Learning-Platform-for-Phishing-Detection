import os
import sys
import Features
from Classifiers import classifiers
import Features
import Features_Support
import Feature_Selection
import Imbalanced_Dataset
from sklearn.externals import joblib
#import User_options
import re
#from Classifiers_test import load_dataset
import configparser
#from collections import deque

config=configparser.ConfigParser()
config.read('Config_file.ini')


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def Confirmation():
    print("##### Review of Options:")
    if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
        print("extract_features_emails = {}".format(config["Email or URL feature Extraction"]["extract_features_emails"]))
    elif config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
        print("extract_features_urls = {}".format(config["Email or URL feature Extraction"]["extract_features_urls"]))
    
    print("###Paths to datasets:")
    print("Legitimate Dataset (Training): {}".format(config["Dataset Path"]["path_legitimate_training"]))
    print("Phishing Dataset (Training):: {}".format(config["Dataset Path"]["path_phishing_training"]))
    print("Legitimate Dataset (Testing): {}".format(config["Dataset Path"]["path_legitimate_testing"]))
    print("Phishing Dataset (Testing): {}".format(config["Dataset Path"]["path_phishing_testing"]))

    print("\nRun the Feature Extraction: {}".format(config["Extraction"]["feature extraction"]))
    print("\nRun the classifiers: {}".format(config["Classification"]["Running the classifiers"]))

    answer = query_yes_no("Do you wish to continue?")
    return answer

def main():
    Feature_extraction=False
    if config["Extraction"]["Feature Extraction"]=='True':
        Feature_extraction=True
        if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
            (feature_list_dict_train, y_train, feature_list_dict_test, y_test)=Features.extract_features_emails()
        elif config["Email or URL feature Extraction"]["extract_features_urls"]=="True":
            (feature_list_dict_train, y_train, feature_list_dict_test, y_test)=Features.extract_features_urls()
        
        X_train, X_test=Features_Support.Vectorization(feature_list_dict_train, feature_list_dict_test)
        #
        #TFIDF
        #
        print("Shape y_train: {}".format(len(y_train)))
        print("Shape y_test: {}".format(len(y_test)))
        print("Shape X_train: {}".format(X_train.shape))
        print("Shape X_test: {}".format(X_test.shape))
        X_train, X_test=Features_Support.Preprocessing(X_train, X_test)
        
        if config["Feature Selection"]["select best features"]=="True":
            #k: Number of Best features
            k=int(config["Feature Selection"]["number of best features"])
            X_train, X_test=Feature_Selection.Select_Best_Features(X_train, y_train, X_test, k)
        if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
            X_train, y_train=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_train, y_train)
        #Dumping Results
        joblib.dump(X_train,"X_train.pkl")
        joblib.dump(y_train,"y_train.pkl")
        joblib.dump(X_test,"X_test.pkl")
        joblib.dump(y_test,"y_test.pkl")

    print("Feature Extraction Done!")
    if config["Classification"]["Running the classifiers"]=="True":
        if Feature_extraction==False:
            X_train=joblib.load("X_train.pkl")
            y_train=joblib.load("y_train.pkl")
            X_test=joblib.load("X_test.pkl")
            y_test=joblib.load("y_test.pkl")

        print("Running the Classifiers....")
        classifiers(X_train, y_train, X_test, y_test )
        print("Done running the Classifiers!!")

if __name__ == "__main__":
    # execute only if run as a script
    answer = Confirmation()
    original = sys.stdout
    if answer is True:
        print("Running......")
        #sys.stdout= open("log.txt",'w')
        main()
        #sys.stdout=original
        print("Done!")
    sys.stdout=original