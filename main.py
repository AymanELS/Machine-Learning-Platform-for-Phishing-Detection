import os
import sys
from Classifiers import classifiers
from Classifiers import fit_MNB
import Features
import Features_Support
import Feature_Selection
import Imbalanced_Dataset
from sklearn.externals import joblib
#import User_options
import re
#from Classifiers_test import load_dataset
import configparser
import Tfidf
from scipy.sparse import hstack
#from collections import deque
import logging
import argparse
from sklearn.datasets import dump_svmlight_file


parser = argparse.ArgumentParser(
    description='A test script for http://stackoverflow.com/q/14097061/78845'
)
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

config=configparser.ConfigParser()
config.read('Config_file.ini')

def setup_logger():
    args = parser.parse_args()

    # create formatter
    # formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
    # create console handler and set level to debug
    handler = logging.StreamHandler()
    # add formatter to handler
    handler.setFormatter(formatter)
    # create logger
    logger = logging.getLogger('root')
    if args.verbose:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)

setup_logger()
logger = logging.getLogger('root')

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
    print("\nFeature Extraction for Training Data: {}".format(config["Extraction"]["training dataset"]))
    print("\nFeature Extraction for Testing Data: {}".format(config["Extraction"]["testing dataset"]))
    print("\nRun the classifiers: {}".format(config["Classification"]["Running the classifiers"]))

    answer = query_yes_no("Do you wish to continue?")
    return answer

def main():
    Feature_extraction=False #flag for feature extraction
    flag_training=False
    # Feature dumping and loading methods
    # flag_saving_pickle=config["Features Format"]["Pikle"]
    # flag_saving_svmlight=config["Features Format"]["Svmlight format"]


    if config["Extraction"]["Feature Extraction"]=='True':
        Feature_extraction=True
        if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
            if config["Extraction"]["Training Dataset"] == "True":
                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_train, y_train, corpus_train)=Features.Extract_Features_Emails_Training()
                # Tranform the list of dictionaries into a sparse matrix
                X_train, vectorizer=Features_Support.Vectorization_Training(feature_list_dict_train)
                # Create Dump directory if doesn't exist
                if not os.path.exists("Data_Dump/Emails_Training"):
                    os.makedirs("Data_Dump/Emails_Training")
                # Save model vor vectorization
                joblib.dump(vectorizer,"Data_Dump/Emails_Training/vectorizer.pkl")
                # Add tfidf if the user marked it as True
                if config["Email_Features"]["tfidf_emails"] == "True":
                    logger.info("tfidf_emails_train ######")
                    Tfidf_train, tfidf_vectorizer=Tfidf.tfidf_emails_training(corpus_train)
                    X_train=hstack([X_train, Tfidf_train])
                    # Save tfidf model
                    joblib.dump(tfidf_vectorizer,"Data_Dump/Emails_Training/tfidf_vectorizer.pkl")

                # Use Min_Max_scaling for prepocessing the feature matrix
                X_train=Features_Support.Preprocessing(X_train)
                if config["Feature Selection"]["select best features"]=="True":
                    #k: Number of Best features
                    logger.info("Select Best Features ######")
                    k = int(config["Feature Selection"]["number of best features"])
                    #X_train, selection = Feature_Selection.Select_Best_Features_Training(X_train, y_train, k)
                    X_train, selection = Feature_Selection.Feature_Ranking(X_train, y_train,k)
                    # dump selection model
                    joblib.dump(selection,"Data_Dump/Emails_Training/selection.pkl")
                # Train Classifiers on imbalanced dataset
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_train, y_train=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_train, y_train)

                # Fit classifier MNB
                #fit_MNB(X_train, y_train)
                # Save features for training dataset
                #dump_svmlight_file(X_train,y_train,"Data_Dump/Emails_Training/Feature_Matrix.txt")
                joblib.dump(X_train,"Data_Dump/Emails_Training/X_train.pkl")
                joblib.dump(y_train,"Data_Dump/Emails_Training/y_train.pkl")

                # flag to mark if training was done
                flag_training=True
                logger.info("Feature Extraction for training dataset: Done!")

            if config["Extraction"]["Testing Dataset"] == "True":
                # if training was done in another instance of the plaform then load the necessary files
                if flag_training==False:
                    X_train=joblib.load("Data_Dump/Emails_Training/X_train.pkl")
                    y_train=joblib.load("Data_Dump/Emails_Training/y_train.pkl")
                    vectorizer=joblib.load("Data_Dump/Emails_Training/vectorizer.pkl")
                    tfidf_vectorizer=joblib.load("Data_Dump/Emails_Training/tfidf_vectorizer.pkl")
                    selection=joblib.load("Data_Dump/Emails_Training/selection.pkl")
                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_test, y_test, corpus_test)=Features.Extract_Features_Emails_Testing()
                # Tranform the list of dictionaries into a sparse matrix
                X_test=Features_Support.Vectorization_Testing(feature_list_dict_test, vectorizer)
                if config["Email_Features"]["tfidf_emails"] == "True":
                    logger.info("tfidf_emails_train ######")
                    Tfidf_test=Tfidf.tfidf_emails_testing(corpus_test, tfidf_vectorizer)
                    X_test=hstack([X_test, Tfidf_test])
                # Use Min_Max_scaling for prepocessing the feature matrix
                X_test=Features_Support.Preprocessing(X_test)
                if config["Feature Selection"]["select best features"]=="True":
                    #k: Number of Best features
                    k = int(config["Feature Selection"]["number of best features"])
                    X_test = Feature_Selection.Select_Best_Features_Testing(X_test, selection)
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_test, y_test=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_test, y_test)
                #Dump Testing feature matrix with labels
                if not os.path.exists("Data_Dump/Emails_Testing"):
                    os.makedirs("Data_Dump/Emails_Testing")
                joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test.pkl")
                joblib.dump(y_test,"Data_Dump/Emails_Testing/y_test.pkl")
                logger.info("Feature Extraction for testing dataset: Done!")

######## URL feature extraction
        if config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
            if config["Extraction"]["Training Dataset"] == "True":
                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_train, y_train, corpus_train)=Features.Extract_Features_Urls_Training()
                # Tranform the list of dictionaries into a sparse matrix
                X_train, vectorizer=Features_Support.Vectorization_Training(feature_list_dict_train)
                # Create directory to store dada
                if not os.path.exists("Data_Dump/URLs_Training"):
                    os.makedirs("Data_Dump/URLs_Training")
                # Dump vectorizer
                joblib.dump(vectorizer,"Data_Dump/URLs_Training/vectorizer.pkl")
                # Add tfidf if the user marked it as True
                if config["HTML_Features"]["tfidf_websites"] == "True":
                    logger.info("Extracting TFIDF features for training websites ###### ######")
                    Tfidf_train, tfidf_vectorizer=Tfidf.tfidf_websites_training(corpus_train)
                    X_train=hstack([X_train, Tfidf_train])
                    #dump tfidf vectorizer
                    joblib.dump(tfidf_vectorizer,"Data_Dump/URLs_Training/tfidf_vectorizer.pkl")
                # Use Min_Max_scaling for prepocessing the feature matrix
                X_train=Features_Support.Preprocessing(X_train)
                if config["Feature Selection"]["select best features"]=="True":
                    #k: Number of Best features
                    k = int(config["Feature Selection"]["number of best features"])
                    X_train, selection = Feature_Selection.Feature_Ranking(X_train, y_train, k)
                    #Dump model
                    joblib.dump(selection,"Data_Dump/URLs_Training/selection.pkl")
                # Train Classifiers on imbalanced dataset
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_train, y_train=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_train, y_train)
                # dump features and labels and vectorizers

                joblib.dump(X_train,"Data_Dump/URLs_Training/X_train.pkl")
                joblib.dump(y_train,"Data_Dump/URLs_Training/y_train.pkl")

                # flag to mark if training was done
                flag_training=True
                logger.info("Feature Extraction for training dataset: Done!")

            if config["Extraction"]["Testing Dataset"] == "True":
                # if training was done in another instance of the plaform then load the necessary files
                if flag_training==False:
                    X_train=joblib.load("Data_Dump/URLs_Training/X_train.pkl")
                    y_train=joblib.load("Data_Dump/URLs_Training/y_train.pkl")
                    vectorizer=joblib.load("Data_Dump/URLs_Training/vectorizer.pkl")
                    tfidf_vectorizer=joblib.load("Data_Dump/URLs_Training/tfidf_vectorizer.pkl")
                    selection=joblib.load("Data_Dump/URLs_Training/selection.pkl")
                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_test, y_test, corpus_test)=Features.Extract_Features_Urls_Testing()
                # Tranform the list of dictionaries into a sparse matrix
                X_test=Features_Support.Vectorization_Testing(feature_list_dict_test, vectorizer)
                if config["HTML_Features"]["tfidf_websites"] == "True":
                    logger.info("Extracting TFIDF features for testing websites ######")
                    Tfidf_test=Tfidf.tfidf_emails_testing(corpus_test, tfidf_vectorizer)
                    X_test=hstack([X_test, Tfidf_test])
                # Use Min_Max_scaling for prepocessing the feature matrix
                X_test=Features_Support.Preprocessing(X_test)
                if config["Feature Selection"]["select best features"]=="True":
                    #k: Number of Best features
                    k = int(config["Feature Selection"]["number of best features"])
                    X_test = Feature_Selection.Select_Best_Features_Testing(X_test, selection)
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_test, y_test=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_test, y_test)
                #Dump Testing feature matrix with labels
                if not os.path.exists("Data_Dump/URLs_Testing"):
                    os.makedirs("Data_Dump/URLs_Testing")
                joblib.dump(X_test,"Data_Dump/URLs_Testing/X_test.pkl")
                joblib.dump(y_test,"Data_Dump/URLs_Testing/y_test.pkl")
                logger.info("Feature Extraction for testing dataset: Done!")


    if config["Classification"]["Running the classifiers"]=="True":
        if Feature_extraction==False:
            if config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
                X_train=joblib.load("Data_Dump/URLs_Training/X_train.pkl")
                y_train=joblib.load("Data_Dump/URLs_Training/y_train.pkl")
                X_test=joblib.load("Data_Dump/URLs_Testing/X_test.pkl")
                y_test=joblib.load("Data_Dump/URLs_Testing/y_test.pkl")
            elif config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
                X_train=joblib.load("Data_Dump/Emails_Training/X_train.pkl")
                y_train=joblib.load("Data_Dump/Emails_Training/y_train.pkl")
                X_test=joblib.load("Data_Dump/Emails_Testing/X_test.pkl")
                y_test=joblib.load("Data_Dump/Emails_Testing/y_test.pkl")

        logger.info("Running the Classifiers....")
        classifiers(X_train, y_train, X_test, y_test)
        logger.info("Done running the Classifiers!!")

if __name__ == "__main__":
    # execute only if run as a script
    answer = Confirmation()
    original = sys.stdout
    if answer is True:
        logger.debug("Running......")
        # sys.stdout= open("log.txt",'w')
        main()
        # sys.stdout=original
        logger.debug("Done!")
    sys.stdout=original
