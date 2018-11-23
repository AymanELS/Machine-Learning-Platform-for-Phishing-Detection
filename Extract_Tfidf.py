import os
import sys
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import pickle
import argparse
import re
import Features_Support


# prog = re.compile("('[a-zA-Z0-9_\-\. ]*':\"'[a-zA-Z0-9_\-\. ]*'\")|('[a-zA-Z0-9_\-\. ]*':\"[a-zA-Z0-9_\-\. ]*\")|('[a-zA-Z0-9_\-\. ]*':[0-9\.[0-9]*)|('[a-zA-Z0-9_\-\. ]*':*)")

prog = re.compile ("""('[a-zA-Z0-9_\-\. ]*':"'[a-zA-Z0-9_\-\. ]*'")|('[a-zA-Z0-9_\-\. ]*':'[a-z0-9\.\s\/\-0-9]*')|('[a-zA-Z0-9_\-\. ]*':[0-9\.0-9]*)""")
prog2= re.compile(r"(>url: .*?<)|(^URL: .*?<)", flags=re.IGNORECASE)
prog3= re.compile("^url: .*?\n", flags=re.IGNORECASE|re.MULTILINE)
prog4= re.compile("\nurl: .*?\n", flags=re.IGNORECASE)

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('--html_content', type=str, required=True,
                    help='path to the human-readable feature file.')
parser.add_argument('--features', type=str, required=False,
                    help='path to the human-readable feature file.')

args = parser.parse_args()


def convert_from_pkl_to_text(file):
	text=''
	first_line=1
	with open(file, 'rb') as f:
		try:
			while(True):
				data=joblib.load(f)
				#print(data)
				#print(re.findall(prog2,str(data)))
				#	url=re.findall(prog2,data)[0]
				#	print(url)
				#	url= ">"+url+"\n<"
				#	data=re.sub(prog2,url,data)
				# 
				if data.startswith("URL: "):
					if first_line==1:
						data=data+"\n"
						first_line=0
					else:
						data="\n"+data+"\n"
					text=text+data
				else:
					text=text+data
		except (EOFError):
			pass
	with open("html_content.txt",'w', errors='ignore') as g:
		g.write(text)
	return text

def convert_from_text_todict(text):
	html_dict=[]
	#f=open("html_content.txt",'r',errors='ignore')
	#text=f.read()
	#result2=str(re.split(prog3,str(text))[0])
	#html_dict.append(result1)
	html_dict= re.split(prog4,text)
	html_dict[0]=(re.split(prog3,html_dict[0]))[1]
	#print(html_dict[0])
	print("Length of dictionary of features: {}".format(len(html_dict)))
	return html_dict

def Tfidf_Vectorizer(html_dict):
	vectorizer=TfidfVectorizer()
	tfidf_matrix=vectorizer.fit_transform(html_dict)
	joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
	return tfidf_matrix

def Combine_Matrix(m1, m2):
	X=hstack([m1, m2])
	X=Features_Support.Preprocessing(X)
	joblib.dump(X,"Features_with_Tfidf_processed.pkl")

if __name__ == '__main__':
	text=convert_from_pkl_to_text(args.html_content)
	html_dict=convert_from_text_todict(text)
	with open("html_dict.txt",'w', errors='ignore') as f:
		#for i in html_dict:
		f.write(str(html_dict))
	tfidf_matrix=Tfidf_Vectorizer(html_dict)
	if args.features:
		Combine_Matrix(args.features,tfidf_matrix)


	#dict_feature_vectors_openphish = convert_from_text_todict(args.features)
	#convert_to_vectorizer(dict_feature_vectors_openphish, args.dataset_name)
