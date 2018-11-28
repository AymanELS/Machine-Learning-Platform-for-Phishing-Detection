import os
import sys
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import pickle
import argparse
import re
import Features_Support
from scipy.sparse import hstack

# prog = re.compile("('[a-zA-Z0-9_\-\. ]*':\"'[a-zA-Z0-9_\-\. ]*'\")|('[a-zA-Z0-9_\-\. ]*':\"[a-zA-Z0-9_\-\. ]*\")|('[a-zA-Z0-9_\-\. ]*':[0-9\.[0-9]*)|('[a-zA-Z0-9_\-\. ]*':*)")


parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('--html_content', type=str, required=True,
                    help='path to the html pkl file.')
parser.add_argument('--features', type=str, required=False,
                    help='path to the feature sparse matrix unprocessed file.')
parser.add_argument('--dataset_name', type=str, required=False,
                    help=' name of dataset.')
parser.add_argument('--output_dir', type=str, required=False,
                    help='directory to store the features.')

args = parser.parse_args()


def convert_from_pkl_to_text(file):
	text=''
	first_line=1
	corpus=[]
	with open(file, 'rb') as f:
		try:
			while(True):
				data=joblib.load(f)
				if data.startswith("URL: ")==False:
					corpus.append(data)
		except (EOFError):
			pass
	#with open(os.path.join(args.output_dir,args.dataset_name+"_html_content.txt"),'w', errors='ignore') as g:
	#	g.write(str(corpus))
	return corpus


def Tfidf_Vectorizer(corpus):
	vectorizer=TfidfVectorizer(analyzer='word', ngram_range=(1,1),
                     min_df = 5, stop_words = 'english', sublinear_tf=True)
	tfidf_matrix=vectorizer.fit_transform(corpus)
	joblib.dump(tfidf_matrix, os.path.join(args.output_dir,args.dataset_name+'_tfidf_matrix_combined.pkl'))
	return tfidf_matrix

def Combine_Matrix(m1, m2):
	print(m1.shape, m2.shape)
	X=hstack([m1, m2])
	X=Features_Support.Preprocessing(X)
	joblib.dump(X, os.path.join(args.output_dir,args.dataset_name+"_Features_with_Tfidf_processed.pkl"))

if __name__ == '__main__':
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	corpus=convert_from_pkl_to_text(args.html_content[0])
	print("length of list of html content (rows in tfidf matrix): {}".format(len(corpus)))
	#with open("list_html_content.txt",'w', errors='ignore') as f:
	#	#for i in html_dict:
	#	f.write(str(html_dict))
	tfidf_matrix=Tfidf_Vectorizer(corpus)
	if args.features:
		X_features = joblib.load(args.features)
		Combine_Matrix(X_features,tfidf_matrix)


	#dict_feature_vectors_openphish = convert_from_text_todict(args.features)
	#convert_to_vectorizer(dict_feature_vectors_openphish, args.dataset_name)
