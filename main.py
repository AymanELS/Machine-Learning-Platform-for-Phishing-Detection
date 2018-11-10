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
import time


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


### Feature ranking only
    if config["Feature Selection"]["Feature Ranking Only"]=='True':
        start=time.time()
        if not os.path.exists("Data_Dump/Feature_Ranking"):
            os.makedirs("Data_Dump/Feature_Ranking")
        if config["Email or URL feature Extraction"]["extract_features_emails"] == "True": 
            # create directory
            if not os.path.exists("Data_Dump/Emails_Training"):
                os.makedirs("Data_Dump/Emails_Training")
            # feature extraction
            (feature_list_dict_train, y, corpus)=Features.Extract_Features_Emails_Training()
            # transforming the feature into a sparse matrix
            X, vectorizer=Features_Support.Vectorization_Training(feature_list_dict_train)
            # preprocessing
            X=Features_Support.Preprocessing(X)
            # dump vectorizer
            joblib.dump(vectorizer,"Data_Dump/Emails_Training/vectorizer.pkl")
        
        elif config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
            if not os.path.exists("Data_Dump/URLs_Training"):
                os.makedirs("Data_Dump/URLs_Training")
            # feature extraction
            (feature_list_dict_train, y, corpus_train)=Features.Extract_Features_Urls_Training()
            # transforming the feature into a sparse matrix
            X, vectorizer=Features_Support.Vectorization_Training(feature_list_dict_train)
            # preprocessing
            X=Features_Support.Preprocessing(X)
            # dump vectorizer
            joblib.dump(vectorizer,"Data_Dump/URLs_Training/vectorizer.pkl")

        logger.info("Select Best Features ######")
        # feature selection
        k = int(config["Feature Selection"]["number of best features"])
        #X, selection = Feature_Selection.Select_Best_Features_Training(X, y, k)
        X, selection = Feature_Selection.Feature_Ranking(X, y,k, feature_list_dict_train)
        #dump selection model
        if config["Email or URL feature Extraction"]["extract_features_emails"] == "True": 
            joblib.dump(selection,"Data_Dump/Emails_Training/selection.pkl")
        elif config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
            joblib.dump(selection,"Data_Dump/URLs_Training/selection.pkl")
        
        end=time.time()
        ex_time=end-start
        
        if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":  
            if not os.path.exists("Data_Dump/Email_Training"):
                os.makedirs("Data_Dump/Email_Training")
            with open("Data_Dump/Email_Training/Feature_Ranking_Extraction_time.txt",'w') as f:
                f.write("Feature Ranking only - Time extraction: {}".format(ex_time))
        if config["Email or URL feature Extraction"]["extract_features_urls"] == "True":  
            if not os.path.exists("Data_Dump/URLs_Training"):
                os.makedirs("Data_Dump/Email_Training")
            with open("Data_Dump/Email_Training/Feature_Ranking_Extraction_time.txt",'w') as f:
                f.write("Feature Ranking only - Time extraction: {}".format(ex_time))



### Email Feature Extraction
    elif config["Extraction"]["Feature Extraction"]=='True':
        Feature_extraction=True
        if config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
            if config["Extraction"]["Training Dataset"] == "True":
                start_training=time.time()
                # Create Data Dump directory if doesn't exist
                if not os.path.exists("Data_Dump/Emails_Training"):
                    os.makedirs("Data_Dump/Emails_Training")
                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_train, y_train, corpus_train)=Features.Extract_Features_Emails_Training()
                # Tranform the list of dictionaries into a sparse matrix
                X_train, vectorizer=Features_Support.Vectorization_Training(feature_list_dict_train)
                # Save model for vectorization
                joblib.dump(vectorizer,"Data_Dump/Emails_Training/vectorizer.pkl")
                joblib.dump(X_train,"Data_Dump/Emails_Training/X_train_unprocessed.pkl")
                # Add tfidf if the user marked it as True
                if config["Email_Body_Features"]["tfidf_emails"] == "True":
                    logger.info("tfidf_emails_train ######")
                    Tfidf_train, tfidf_vectorizer=Tfidf.tfidf_training(corpus_train)
                    X_train=hstack([X_train, Tfidf_train])
                    # Save tfidf model
                    joblib.dump(tfidf_vectorizer,"Data_Dump/Emails_Training/tfidf_vectorizer.pkl")
                    joblib.dump(X_train,"Data_Dump/Emails_Training/X_train_unprocessed_with_tfidf.pkl")
                
                # Use Min_Max_scaling for prepocessing the feature matrix
                X_train=Features_Support.Preprocessing(X_train)

                # feature ranking
                if config["Feature Selection"]["select best features"]=="True":
                    #k: Number of Best features
                    logger.info("Select Best Features ######")
                    k = int(config["Feature Selection"]["number of best features"])
                    #X_train, selection = Feature_Selection.Select_Best_Features_Training(X_train, y_train, k)
                    X_train, selection = Feature_Selection.Feature_Ranking(X_train, y_train,k, feature_list_dict_train)
                    # dump selection model
                    joblib.dump(selection,"Data_Dump/Emails_Training/selection.pkl")
                    logger.info("### Feature Ranking and Selection for Training Done!")
                    joblib.dump(X_train,"Data_Dump/Emails_Training/X_train_processed_best_features.pkl")
                
                
                # Train Classifiers on imbalanced dataset
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_train, y_train=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_train, y_train)

                # Fit classifier MNB
                #fit_MNB(X_train, y_train)
                # Save features for training dataset
                #dump_svmlight_file(X_train,y_train,"Data_Dump/Emails_Training/Feature_Matrix.txt")
                joblib.dump(X_train,"Data_Dump/Emails_Training/X_train.pkl")
                joblib.dump(y_train,"Data_Dump/Emails_Training/y_train.pkl")
                end_training=time.time()
                training_time=end_training-start_training

                # flag to mark if training was done
                flag_training=True
                logger.info("Feature Extraction for training dataset: Done!")
                logger.info("Feature Extraction time for training dataset: {}".format(training_time))

            
                if not os.path.exists("Data_Dump/Email_Training"):
                    os.makedirs("Data_Dump/Email_Training")
                with open("Data_Dump/Email_Training/Feature_Extraction_time.txt",'w') as f:
                    f.write("Feature Extraction Time for Training dataset: {}".format(training_time))

            if config["Extraction"]["Testing Dataset"] == "True":
                start_testing=time.time()
                # if training was done in another instance of the plaform then load the necessary files
                if flag_training==False:
                    X_train=joblib.load("Data_Dump/Emails_Training/X_train.pkl")
                    y_train=joblib.load("Data_Dump/Emails_Training/y_train.pkl")
                    vectorizer=joblib.load("Data_Dump/Emails_Training/vectorizer.pkl")
                
                # Extract features in a dictionnary for each email. return a list of dictionaries
                (feature_list_dict_test, y_test, corpus_test)=Features.Extract_Features_Emails_Testing()
                # Tranform the list of dictionaries into a sparse matrix
                X_test=Features_Support.Vectorization_Testing(feature_list_dict_test, vectorizer)
                
                joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test_unprocessed.pkl")
                # Add tfidf if the user marked it as True
                if config["Email_Body_Features"]["tfidf_emails"] == "True":
                    tfidf_vectorizer=joblib.load("Data_Dump/Emails_Training/tfidf_vectorizer.pkl")
                    logger.info("tfidf_emails_train ######")
                    Tfidf_test=Tfidf.tfidf_testing(corpus_test, tfidf_vectorizer)
                    X_test=hstack([X_test, Tfidf_test])
                    joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test_unprocessed_with_tfidf.pkl")
                
                # Use Min_Max_scaling for prepocessing the feature matrix
                X_test=Features_Support.Preprocessing(X_test)

                joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test_processed.pkl")
                # feature ranking
                if config["Feature Selection"]["select best features"]=="True":
                    #k: Number of Best features
                    selection=joblib.load("Data_Dump/Emails_Training/selection.pkl")
                    k = int(config["Feature Selection"]["number of best features"])
                    X_test = Feature_Selection.Select_Best_Features_Testing(X_test, selection, k, feature_list_dict_test)
                    logger.info("### Feature Ranking and Selection for Training Done!")
                
                joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test_processed_best_features.pkl")
                # Train Classifiers on imbalanced dataset
                if config["Imbalanced Datasets"]["Load_imbalanced_dataset"]=="True":
                    X_test, y_test=Imbalanced_Dataset.Make_Imbalanced_Dataset(X_test, y_test)
                end_testing=time.time()
                testing_time=end_testing-start_testing
                #Dump Testing feature matrix with labels
                if not os.path.exists("Data_Dump/Emails_Testing"):
                    os.makedirs("Data_Dump/Emails_Testing")
                joblib.dump(X_test,"Data_Dump/Emails_Testing/X_test.pkl")
                joblib.dump(y_test,"Data_Dump/Emails_Testing/y_test.pkl")
                logger.info("Feature Extraction for testing dataset: Done!")
                logger.info("Feature Extraction time for testing dataset: {}".format(testing_time))
                

                if not os.path.exists("Data_Dump/Email_Testing"):
                    os.makedirs("Data_Dump/Email_Testing")
                with open("Data_Dump/Email_Testing/Feature_Extraction_time.txt",'w') as f:
                    f.write("Feature Extraction Time for Testing dataset: {}".format(testing_time))

######## URL feature extraction
        elif config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
            if config["Extraction"]["Training Dataset"] == "True":
                start_training=time.time()
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
                end_training=time.time()
                training_time=end_training-start_training
                logger.info("Feature Extraction and processing time for training dataset: {}".format(training_time))

                with open("Data_Dump/URLs_Training/Feature_Extraction_time.txt",'w') as f:
                    f.write("Feature Extraction and processing Time for Training dataset: {}".format(training_time))

            if config["Extraction"]["Testing Dataset"] == "True":
                start_testing=time.time()
                
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
                end_testing=time.time()
                testing_time=end_testing-start_testing
                #Dump Testing feature matrix with labels
                if not os.path.exists("Data_Dump/URLs_Testing"):
                    os.makedirs("Data_Dump/URLs_Testing")
                joblib.dump(X_test,"Data_Dump/URLs_Testing/X_test.pkl")
                joblib.dump(y_test,"Data_Dump/URLs_Testing/y_test.pkl")

                with open("Data_Dump/URLs_Testing/Feature_Extraction_time.txt",'w') as f:
                    f.write("Feature Extraction Time for Testing dataset: {}".format(testing_time))

                logger.info("Feature Extraction for testing dataset: Done!")
                logger.info("Feature Extraction and processing time for testing dataset: {}".format(testing_time))

    if config["Classification"]["Running the classifiers"]=="True":
        if Feature_extraction==False:
            start_classification=time.time()
            if config["Email or URL feature Extraction"]["extract_features_urls"] == "True":
                X_train=joblib.load("Data_Dump/URLs_Training/X_train.pkl")
                y_train=joblib.load("Data_Dump/URLs_Training/y_train.pkl")
                X_test=joblib.load("Data_Dump/URLs_Testing/X_test.pkl")
                y_test=joblib.load("Data_Dump/URLs_Testing/y_test.pkl")
                vectorizer=joblib.load("Data_Dump/URLs_Training/vectorizer.pkl")
                features_extracted=vectorizer.get_feature_names()
                logger.info(features_extracted)
                Features_training=vectorizer.inverse_transform(X_train)
                Features_testing=vectorizer.inverse_transform(X_test)
                mask=[]
                #mask.append(0)
                list_restricted_features=[]
                        #logger.info("Section: {} ".format(section))
                for feature in features_extracted:
                    feature_name=feature
                    if "=" in feature:
                        feature_name=feature.split("=")[0]
                    if "url_char_distance_" in feature:
                        feature_name="char_distance"
                    for section in ["HTML_Features", "URL_Features", "Network_Features", "Javascript_Features"]:
                        try:
                            if config[section][feature_name]=="True":
                                if config[section][section.lower()]=="True":
                                    logger.info("Feature: {}".format(feature))
                                    list_restricted_features.append(feature)
                                    mask.append(1)
                                else:
                                    mask.append(0)
                        except KeyError as e:
                            pass
                vectorizer.restrict(mask)
                logger.info((vectorizer.get_feature_names()))
                X_train=vectorizer.transform(Features_training)
                X_test=vectorizer.transform(Features_testing)
                end_classification=time.time()
                classification_time=end_classification-start_classification
                if not os.path.exists("Data_Dump/URLs_Classification"):
                    os.makedirs("Data_Dump/URLs_Classification")
                joblib.dump(vectorizer, "Data_Dump/URLs_Classification/vectorizer_restricted.pkl")
                joblib.dump(X_train,"Data_Dump/URLs_Classification/X_train_restricted.pkl")
                joblib.dump(X_test,"Data_Dump/URLs_Classification/X_test_restricted.pkl")
                logger.info("Running the Classifiers....")
                classifiers(X_train, y_train, X_test, y_test)
                with open("Data_Dump/URLs_Classification/Classification_time.txt",'w') as f:
                    f.write("Classification time: {}".format(classification_time))

                #logger.info(len(vectorizer.get_feature_names()))
            elif config["Email or URL feature Extraction"]["extract_features_emails"] == "True":
                X_train=joblib.load("Data_Dump/Emails_Training/X_train.pkl")
                y_train=joblib.load("Data_Dump/Emails_Training/y_train.pkl")
                X_test=joblib.load("Data_Dump/Emails_Testing/X_test.pkl")
                y_test=joblib.load("Data_Dump/Emails_Testing/y_test.pkl")
                vectorizer=joblib.load("Data_Dump/Emails_Training/vectorizer.pkl")
                features_extracted = vectorizer.get_feature_names()
                logger.info(len(features_extracted))
                mask= []
                for feature_name in features_extracted:
                    if "=" in feature_name:
                        feature_name=feature_name.split("=")[0]
                    if "count_in_body" in feature_name:
                        if config["Email_Features"]["blacklisted_words_body"] == "True":
                            mask.append(1)
                        else:
                            mask.append(0)
                    elif "count_in_subject" in feature_name:
                        if config["Email_Features"]["blacklisted_words_subject"] == "True":
                            mask.append(1)
                        else:
                            mask.append(0)
                    else:
                        if config["Email_Features"][feature_name]=="True":
                            mask.append(1)
                        else:
                            mask.append(0)
                logger.info(mask)
                vectorizer=vectorizer.restrict(mask)
                logger.info(len(vectorizer.get_feature_names()))

                #with open("Data_Dump/URLs_Classification/Classification_time.txt",'w') as f:
                #    f.write("Classification time: {}".format(classification_time))
                #X_train=vectorizer.transform(X_train)
                logger.info("Running the Classifiers....")
                classifiers(X_train, y_train, X_test, y_test)
                with open("Data_Dump/Emails_Classification/Classification_time.txt",'w') as f:
                    f.write("Classification time: {}".format(classification_time))
                
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
