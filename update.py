import os
import sys
import Features
import Classifiers
import Imbalanced_Dataset
import Evaluation_Metrics
import inspect
import configparser
#from Classifiers_test import load_dataset

def config(list_Features, list_Classifiers, list_Imbalanced_dataset, list_Evaluation_metrics):
	config = configparser.ConfigParser()

	config['Features'] = {}
	config['Email_Features']={}
	#C_Features=config['Features']
	C_Email_Features=config['Email_Features']
	C_Email_Features["extract header features"]="True"
	C_Email_Features["extract body features"]="True"
	C_Email_Features["extract url features"]="True"
	C_Email_Features["extract external features"]="False"


	config['Email_Body_Features']={}
	C_Email_Features=config['Email_Body_Features']
	for feature in list_Features:
		if feature.startswith("Email_Body"):
			C_Email_Features[feature.replace('Email_Body_','')]="True"

	config['Email_Header_Features']={}
	C_Email_Features=config['Email_Header_Features']
	for feature in list_Features:
		if feature.startswith("Email_Header"):
			C_Email_Features[feature.replace('Email_Header_','')]="True"

	config['Email_URL_Features']={}
	C_Email_Features=config['Email_URL_Features']
	for feature in list_Features:
		if feature.startswith("Email_URL"):
			C_Email_Features[feature.replace('Email_URL_','')]="True"

	config['Email_External_Features']={}
	C_Email_Features=config['Email_External_Features']
	for feature in list_Features:
		if feature.startswith("Email_External"):
			C_Email_Features[feature.replace('Email_External_','')]="False"

	config['HTML_Features']={}
	C_HTML_Features=config['HTML_Features']
	for feature in list_Features:
		if feature.startswith("HTML_"):
			C_HTML_Features[feature.replace('HTML_','')]="True"

	config['URL_Features']={}
	C_URL_Features=config['URL_Features']	
	for feature in list_Features:
		if feature.startswith("URL_"):
			C_URL_Features[feature.replace('URL_','')]="True"


	config['Network_Features']={}
	C_Network_Features=config['Network_Features']
	for feature in list_Features:
		if feature.startswith("Network_"):
			C_Network_Features[feature.replace('Network_','')]="True"


	config['Javascript_Features']={}
	C_Javascript_Features=config['Javascript_Features']
	for feature in list_Features:
		if feature.startswith("Javascript_"):
			C_Javascript_Features[feature.replace('Javascript_','')]="True"

	config['Classifiers']={}
	C_Classifiers=config['Classifiers']
	C_Classifiers["weighted"]="False"
	C_Classifiers["autosklearn"]="False"
	C_Classifiers["bagging"]="True"
	C_Classifiers["boosting"]="True"
	C_Classifiers["dnn"]="True"
	C_Classifiers["decisiontree"]="True"
	C_Classifiers["elm"]="True"
	C_Classifiers["gaussiannaivebayes"]="True"
	#C_Classifiers["hddt"]="False"
	C_Classifiers["kmeans"]="False"
	C_Classifiers["logisticregression"]="True"
	C_Classifiers["multinomialnaivebayes"]="True"
	C_Classifiers["randomforest"]="True"
	C_Classifiers["svm"]="True"
	C_Classifiers["tpot"]="False"
	C_Classifiers["knearestneighbor"]="True"


	config['Imbalanced Datasets'] = {}
	C_Imbalanced=config['Imbalanced Datasets']
	for imbalanced in list_Imbalanced_dataset:
		C_Imbalanced[imbalanced]="True"
	C_Imbalanced["load_imbalanced_dataset"]="False"

	config['Evaluation Metrics']={}
	C_Metrics=config['Evaluation Metrics']
	C_Metrics["accuracy"]="True"
	C_Metrics["balanced_accuracy_score"]="True"
	C_Metrics["completeness"]="False"
	C_Metrics["confusion_matrix"]="True"
	C_Metrics["Cross_validation"]="False"
	C_Metrics["f1_score"]="True"
	C_Metrics["geomteric_mean_score"]="True"
	C_Metrics["homogenity"]="False"
	C_Metrics["matthews_corrcoef"]="True"
	C_Metrics["precision"]="True"
	C_Metrics["roc_auc"]="True"
	C_Metrics["recall"]="True"
	C_Metrics["v_measure"]="False"
	C_Metrics["geometric_mean_score"]="True"
	C_Metrics["parameter_search"]="False"

	

	#for metric in list_Evaluation_metrics:
	#	C_Metrics[metric]="True"

	config['Preprocessing']={}
	C_Preprocessing=config['Preprocessing']
	#C_Preprocessing['mean_scaling']= "True"
	C_Preprocessing['min_max_scaling']= "True"
	#C_Preprocessing['abs_scaler']= "True"
	#C_Preprocessing['normalize']= "True"
	

	config["Feature Selection"]={}
	C_selection=config["Feature Selection"]
	C_selection["Select Best Features"]="True"
	C_selection["Feature Ranking Only"]="False"
	C_selection["with Tfidf"]="True"
	C_selection["Number of Best Features"]="10"
	C_selection["Recursive Feature Elimination"]="False"
	C_selection["Information Gain"]="True"
	C_selection["Gini"]="False"
	C_selection["Chi-2"]="False"
	C_selection["LSA"]="False"
	C_selection["PCA"]="False"

	config['Dataset Path']={}
	C_Dataset=config['Dataset Path']
	C_Dataset["path_legitimate_training"]="Training_legit_Emails"
	C_Dataset["path_phishing_training"]="Training_phish_Emails"
	C_Dataset["path_legitimate_testing"]="Testing_legit_Emails"
	C_Dataset["path_phishing_testing"]="Testing_phish_Emails"

	config['Email or URL feature Extraction']={}
	C_email_url=config['Email or URL feature Extraction']
	C_email_url["extract_features_emails"]="True"
	C_email_url["extract_features_urls"]="False"

	config['Extraction']={}
	C_extraction=config['Extraction']
	C_extraction["Feature Extraction"]="True"
	C_extraction["Training Dataset"]="True"
	C_extraction["Testing Dataset"]="True"

	config['Features Format']={}
	C_features_format=config['Features Format']
	C_features_format["Pikle"]="False"
	C_features_format["Svmlight format"]="False"


	config['Classification']={}
	C_classification=config['Classification']
	C_classification["Running the Classifiers"]="True"
	C_classification["Save Model"]="True"
	C_classification["load model"]="False"
	C_classification["rank classifiers"]="False"
	C_classification["rank on metric"]="False"

	#config["Summary"]={}
	#C_summary=config["Summary"]
	#C_summary["Path"]="summary.txt"

	with open('Config_file.ini', 'w') as configfile:
		config.write(configfile)


def update_list():
	list_Features=[]
	list_Classifiers=[]
	list_Evaluation_metrics=[]
	list_Imbalanced_dataset=[]
	for a in dir(Features):
		element=getattr(Features, a)
		if inspect.isfunction(element):
			list_Features.append(a)

	for a in dir(Classifiers):
		element=getattr(Classifiers, a)
		if inspect.isfunction(element):
			list_Classifiers.append(a)

	for a in dir(Imbalanced_Dataset):
		element=getattr(Imbalanced_Dataset, a)
		if inspect.isfunction(element):
			list_Imbalanced_dataset.append(a)

	for a in dir(Evaluation_Metrics):
		element=getattr(Evaluation_Metrics, a)
		if inspect.isfunction(element):
			list_Evaluation_metrics.append(a)

	return list_Features, list_Classifiers, list_Imbalanced_dataset, list_Evaluation_metrics


if __name__ == "__main__":
    # execute only if run as a script
    list_Features, list_Classifiers, list_Imbalanced_dataset, list_Evaluation_metrics = update_list()
    #update_file(list_Features, list_Classifiers, list_Imbalanced_dataset, list_Evaluation_metrics)
    config(list_Features, list_Classifiers, list_Imbalanced_dataset, list_Evaluation_metrics)
