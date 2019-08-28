import os
import sys
#from Classifiers import classifiers, Attack_Features
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
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import sparse

parser = argparse.ArgumentParser(description='Argument parser')
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

    print("\nRun Feature Ranking Only: {}".format(config["Feature Selection"]["Feature Ranking Only"]))
    if config["Extraction"]["feature extraction"]=="True":
        print("\nRun the Feature Extraction: {}".format(config["Extraction"]["feature extraction"]))
        print("\nFeature Extraction for Training Data: {}".format(config["Extraction"]["training dataset"]))
        print("\nFeature Extraction for Testing Data: {}".format(config["Extraction"]["testing dataset"]))
    else:
        print("\nRun the Feature Extraction: {}".format(config["Extraction"]["feature extraction"]))
    print("\nRun the classifiers: {}".format(config["Classification"]["Running the classifiers"]))
    print("\n")
    answer = query_yes_no("Do you wish to continue?")
    return answer

def main():
    Feature_extraction=False #flag for feature extraction
    flag_training=False
    # Feature dumping and loading methods
    # flag_saving_pickle=config["Features Format"]["Pikle"]
    # flag_saving_svmlight=config["Features Format"]["Svmlight format"]


### Feature ranking only/ Features need to be already extracted
    if config["Feature Selection"]["Feature Ranking Only"]=='True':
        if not os.path.exists("Data_Dump/Feature_Ranking"):
                os.makedirs("Data_Dump/Feature_Ranking")        
        if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
            vectorizer= joblib.load("Data_Dump/Emails_Training/vectorizer.pkl")
            if os.path.exists("Data_Dump/Emails_Training/tfidf_vectorizer.pkl"):
                tfidf_vectorizer= joblib.load("Data_Dump/Emails_Training/tfidf_vectorizer.pkl")
            else:
                tfidf_vectorizer=None
            if config["Feature Selection"]["with Tfidf"] == "True":
                X=joblib.load("Data_Dump/Emails_Training/X_train_processed_with_tfidf.pkl")
            else:
                X=joblib.load("Data_Dump/Emails_Training/X_train_processed.pkl")
                print("without tfidf")
            y=joblib.load("Data_Dump/Emails_Training/y_train.pkl")
        elif config["Email or URL feature Extraction"]["extract_features_URLs"] == "True":
            vectorizer= joblib.load("Data_Dump/URLs_Training/vectorizer.pkl")
            X=joblib.load("Data_Dump/URLs_Training/X_train.pkl")
            y=joblib.load("Data_Dump/URLs_Training/y_train.pkl")
        #feature_list_dict_train=None
        logger.info("Select Best Features ######")
        k = int(config["Feature Selection"]["number of best features"])
        X, selection = Feature_Selection.Feature_Ranking(X, y, k, vectorizer, tfidf_vectorizer)


### Email Feature Extraction
    elif config["Extraction"]["Feature Extraction"]=='True':
        Feature_extraction=True
        if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
            #TRAINING
            if config["Extraction"]["Training Dataset"] == "True":
                start_training=time.time()
                # Create Data Dump directory if doesn't exist
                if not os.path.exists("Data_Dump/Emails_Training"):
                    os.makedirs("Data_Dump/Emails_Training")
                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_train, y_train, corpus_train)=Features.Extract_Features_Emails_Training()
                Features_Support.Cleaning(feature_list_dict_train)
                X_train, vectorizer=Features_Support.Vectorization_Training(feature_list_dict_train)
                ## Save model for vectorization
                joblib.dump(vectorizer,"Data_Dump/Emails_Training/vectorizer.pkl")
                joblib.dump(X_train,"Data_Dump/Emails_Training/X_train_unprocessed.pkl")
                joblib.dump(y_train,"Data_Dump/Emails_Training/y_train.pkl")
                # Add tfidf if the user marked it as True
                if config["Email_Body_Features"]["tfidf_emails"] == "True":
                    logger.info("tfidf_emails_train ######")
                    Tfidf_train, tfidf_vectorizer=Tfidf.tfidf_training(corpus_train)
                    # Save tfidf model
                    joblib.dump(tfidf_vectorizer,"Data_Dump/Emails_Training/tfidf_vectorizer.pkl")
                    joblib.dump(Tfidf_train,"Data_Dump/Emails_Training/tfidf.pkl")
                    # join normal features with tfidf
                    X_train=hstack([X_train, Tfidf_train])
                    joblib.dump(X_train,"Data_Dump/Emails_Training/X_train_unprocessed_with_tfidf.pkl")
                    # preprocessing
                    X_train=Features_Support.Preprocessing(X_train)
                    joblib.dump(X_train,"Data_Dump/Emails_Training/X_train_processed_with_tfidf.pkl")
                # Use Min_Max_scaling for prepocessing the feature matrix
                if config["Email_Body_Features"]["tfidf_emails"] == "False":
                    X_train=Features_Support.Preprocessing(X_train)
                    joblib.dump(X_train,"Data_Dump/Emails_Training/X_train_processed.pkl")
                
                # feature ranking
                if config["Feature Selection"]["select best features"]=="True":
                    #k: Number of Best features
                    logger.info("Select Best Features ######")
                    k = int(config["Feature Selection"]["number of best features"])
                    if config["Email_Body_Features"]["tfidf_emails"] == "False":
                        tfidf_vectorizer=None
                    X_train, selection = Feature_Selection.Feature_Ranking(X_train, y_train, k, vectorizer, tfidf_vectorizer)
                    joblib.dump(X_train,"Data_Dump/Emails_Training/X_train_processed_best_features.pkl")
                    joblib.dump(selection,"Data_Dump/Emails_Training/selection.pkl")
                
                # Train Classifiers on imbalanced dataset
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_train, y_train=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_train, y_train)
                    joblib.dump(X_train,"Data_Dump/Emails_Training/X_train_imbalanced.pkl")
                    joblib.dump(y_train,"Data_Dump/Emails_Training/y_train_imbalanced.pkl")

                end_training=time.time()
                training_time=end_training-start_training

                # flag to mark if training was done
                flag_training=True
                logger.info("Feature Extraction for training dataset: Done!")
                logger.info("Feature Extraction time for training dataset: {}".format(training_time))

                with open("Data_Dump/Emails_Training/Feature_Extraction_time.txt",'w') as f:
                    f.write("Feature Extraction Time for Training dataset: {}".format(training_time))

            #TESTING
            if config["Extraction"]["Testing Dataset"] == "True":
                start_testing=time.time()
                if not os.path.exists("Data_Dump/Emails_Testing"):
                    os.makedirs("Data_Dump/Emails_Testing")
                # if training was done in another instance of the plaform then load the necessary files
                if flag_training==False:
                    if config["Email_Body_Features"]["tfidf_emails"] == "True":
                        X_train=joblib.load("Data_Dump/Emails_Training/X_train_processed_with_tfidf.pkl")
                        y_train=joblib.load("Data_Dump/Emails_Training/y_train.pkl")
                    if config["Feature Selection"]["select best features"]=="True":
                        X_train=joblib.load("Data_Dump/Emails_Training/X_train_processed_best_features.pkl")
                        y_train=joblib.load("Data_Dump/Emails_Training/y_train.pkl")
                    if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                        X_train=joblib.load("Data_Dump/Emails_Training/X_train_imbalanced.pkl")
                        y_train=joblib.load("Data_Dump/Emails_Training/y_train_imbalanced.pkl")
                    vectorizer=joblib.load("Data_Dump/Emails_Training/vectorizer.pkl")
                
                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_test, y_test, corpus_test)=Features.Extract_Features_Emails_Testing()
                # Tranform the list of dictionaries into a sparse matrix
                X_test=Features_Support.Vectorization_Testing(feature_list_dict_test, vectorizer)
                
                joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test_unprocessed.pkl")
                joblib.dump(y_test,"Data_Dump/Emails_Testing/y_test.pkl")
                # Add tfidf if the user marked it as True
                if config["Email_Body_Features"]["tfidf_emails"] == "True":
                    tfidf_vectorizer=joblib.load("Data_Dump/Emails_Training/tfidf_vectorizer.pkl")
                    logger.info("tfidf_emails_train ######")
                    Tfidf_test=Tfidf.tfidf_testing(corpus_test, tfidf_vectorizer)
                    joblib.dump(Tfidf_test,"Data_Dump/Emails_Testing/tfidf.pkl")
                    # join normal features with tfidf
                    X_test=hstack([X_test, Tfidf_test])
                    joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test_unprocessed_with_tfidf.pkl")
                    X_test=Features_Support.Preprocessing(X_test)
                    joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test_processed_with_tfidf.pkl") 
                
                # Use Min_Max_scaling for prepocessing the feature matrix
                X_test=Features_Support.Preprocessing(X_test)
                joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test_processed.pkl")

                # feature ranking
                if config["Feature Selection"]["select best features"]=="True":
                    #k: Number of Best features
                    selection=joblib.load("Data_Dump/Emails_Training/selection.pkl")
                    k = int(config["Feature Selection"]["number of best features"])
                    X_test = Feature_Selection.Select_Best_Features_Testing(X_test, selection)
                    logger.info("### Feature Ranking and Selection for Training Done!")
                    joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test_processed_best_features.pkl")

                # Train Classifiers on imbalanced dataset
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_test, y_test=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_test, y_test)
                    joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test_imbalanced.pkl")
                    joblib.dump(y_test,"Data_Dump/Emails_Testing/y_test_imbalanced.pkl")

                end_testing=time.time()
                testing_time=end_testing-start_testing
                #Dump Testing feature matrix with labels
                #joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test.pkl")
                #joblib.dump(y_test,"Data_Dump/Emails_Testing/y_test.pkl")
                logger.info("Feature Extraction for testing dataset: Done!")
                logger.info("Feature Extraction time for testing dataset: {}".format(testing_time))
                with open("Data_Dump/Emails_Testing/Feature_Extraction_time.txt",'w') as f:
                    f.write("Feature Extraction Time for Testing dataset: {}".format(testing_time))

######## URL feature extraction
        elif config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
            if config["Extraction"]["Training Dataset"] == "True":
                # Create directory to store dada
                if not os.path.exists("Data_Dump/URLs_Training"):
                    os.makedirs("Data_Dump/URLs_Training")

                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_train, y_train, corpus_train)=Features.Extract_Features_Urls_Training()
                
                # Tranform the list of dictionaries into a sparse matrix
                X_train, vectorizer=Features_Support.Vectorization_Training(feature_list_dict_train)
                # Dump vectorizer
                joblib.dump(vectorizer,"Data_Dump/URLs_Training/vectorizer.pkl")
                joblib.dump(X_train,"Data_Dump/URLs_Training/X_train_unprocessed.pkl")
                # Add tfidf if the user marked it as True
                if config["HTML_Features"]["tfidf_websites"] == "True":
                    logger.info("Extracting TFIDF features for training websites ###### ######")
                    Tfidf_train, tfidf_vectorizer=Tfidf.tfidf_training(corpus_train)
                    joblib.dump(Tfidf_train, "Data_Dump/URLs_Training/tfidf_features.pkl")
                    X_train=hstack([X_train, Tfidf_train])
                    #dump tfidf vectorizer
                    joblib.dump(tfidf_vectorizer,"Data_Dump/URLs_Training/tfidf_vectorizer.pkl")
                
                joblib.dump(X_train,"Data_Dump/URLs_Training/X_train_unprocessed_with_tfidf.pkl")
                # Use Min_Max_scaling for prepocessing the feature matrix
                X_train=Features_Support.Preprocessing(X_train)
                joblib.dump(X_train,"Data_Dump/URLs_Training/X_train_processed.pkl")

                # Feature Selection
                if config["Feature Selection"]["select best features"]=="True":
                    #k: Number of Best features
                    k = int(config["Feature Selection"]["number of best features"])
                    X_train, selection = Feature_Selection.Feature_Ranking(X_train, y_train,k, feature_list_dict_train)
                    #Dump model
                    joblib.dump(selection,"Data_Dump/URLs_Training/selection.pkl")
                    joblib.dump(X_train,"Data_Dump/URLs_Training/X_train_processed_best_features.pkl")

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
                    
                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_test, y_test, corpus_test)=Features.Extract_Features_Urls_Testing()
                # Tranform the list of dictionaries into a sparse matrix
                X_test=Features_Support.Vectorization_Testing(feature_list_dict_test, vectorizer)
                joblib.dump(X_test,"Data_Dump/URLs_Testing/X_test_unprocessed.pkl")
                # TFIDF
                if config["HTML_Features"]["tfidf_websites"] == "True":
                    if flag_training==False:
                        tfidf_vectorizer=joblib.load("Data_Dump/URLs_Training/tfidf_vectorizer.pkl")
                    logger.info("Extracting TFIDF features for testing websites ######")
                    Tfidf_test=Tfidf.tfidf_testing(corpus_test, tfidf_vectorizer)
                    joblib.dump(Tfidf_test, "Data_Dump/URLs_Testing/tfidf_features.pkl")
                    X_test=hstack([X_test, Tfidf_test])
                
                joblib.dump(X_test,"Data_Dump/URLs_Testing/X_test_unprocessed_with_tfidf.pkl")
                # Use Min_Max_scaling for prepocessing the feature matrix
                X_test=Features_Support.Preprocessing(X_test)
                joblib.dump(X_train,"Data_Dump/URLs_Training/X_test_processed.pkl")
                
                # Feature Selection
                if config["Feature Selection"]["select best features"]=="True":
                    if flag_training==False:
                        selection=joblib.load("Data_Dump/URLs_Training/selection.pkl")
                    #k: Number of Best features
                    k = int(config["Feature Selection"]["number of best features"])
                    X_test = Feature_Selection.Select_Best_Features_Testing(X_test, selection, k, feature_list_dict_test)
                    joblib.dump(X_train,"Data_Dump/URLs_Training/X_test_processed_best_features.pkl")
                
                
                # Test on imbalanced datasets
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_test, y_test=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_test, y_test)
                #Dump Testing feature matrix with labels
                if not os.path.exists("Data_Dump/URLs_Testing"):
                    os.makedirs("Data_Dump/URLs_Testing")
                joblib.dump(X_test,"Data_Dump/URLs_Testing/X_test.pkl")
                joblib.dump(y_test,"Data_Dump/URLs_Testing/y_test.pkl")
                logger.info("Feature Extraction for testing dataset: Done!")


    if config["Classification"]["Running the classifiers"]=="True":
        #if Feature_extraction==False:
        if config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
            X_train=joblib.load("Data_Dump/URLs_Training/X_train.pkl")
            y_train=joblib.load("Data_Dump/URLs_Training/y_train.pkl")
            X_test=joblib.load("Data_Dump/URLs_Testing/X_test.pkl")
            y_test=joblib.load("Data_Dump/URLs_Testing/y_test.pkl")

        if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
            if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                if config["Classification"]["load model"] == "True":
                    X_train=None
                    y_train=None
                else:    
                    X_train=joblib.load("Data_Dump/Emails_Training/X_train_imbalanced.pkl")
                    y_train=joblib.load("Data_Dump/Emails_Training/y_train_imbalanced.pkl")
                X_test=joblib.load("Data_Dump/Emails_Testing/X_test_imbalanced.pkl")
                y_test=joblib.load("Data_Dump/Emails_Testing/y_test_imbalanced.pkl")
            if config["Email_Body_Features"]["tfidf_emails"] == "True":
                if config["Classification"]["load model"] == "True":
                    X_train=None
                    y_train=None
                else:    
                    X_train=joblib.load("Data_Dump/Emails_Training/X_train_processed_with_tfidf.pkl")
                    y_train=joblib.load("Data_Dump/Emails_Training/y_train.pkl")
                X_test=joblib.load("Data_Dump/Emails_Testing/X_test_processed_with_tfidf.pkl")
                y_test=joblib.load("Data_Dump/Emails_Testing/y_test.pkl")
            if config["Feature Selection"]["select best features"]=="True":
                if config["Classification"]["load model"] == "True":
                    X_train=None
                    y_train=None
                else:    
                    X_train=joblib.load("Data_Dump/Emails_Training/X_train_processed_best_features.pkl")
                    y_train=joblib.load("Data_Dump/Emails_Training/y_train.pkl")
                X_test=joblib.load("Data_Dump/Emails_Testing/X_test_processed_best_features.pkl")
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