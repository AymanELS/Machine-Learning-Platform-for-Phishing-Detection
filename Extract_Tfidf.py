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


parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('--html_content', type=str, required=True,
                    help='path to the html pkl file.')
parser.add_argument('--features', type=str, required=False,
                    help='path to the feature sparse matrix unprocessed file.')
parser.add_argument('--labels', type=str, required=True,
                    help='path to the label file.')
parser.add_argument('--dataset_name', type=str, required=False,
                    help=' name of dataset.')
parser.add_argument('--output_dir', type=str, required=False,
                    help='directory to store the features.')

args = parser.parse_args()

def binormal_separation(corpus):
    y_train=joblib.load(args.labels)
    vocab = []
    vectorizer=CountVectorizer(analyzer='word', ngram_range=(1,1), min_df = 5, stop_words = 'english')
    analyzer = vectorizer.build_analyzer()
    new_corpus= []
    input_dict = {}
    input_dict['phish'] = []
    input_dict['legit'] = []
    for i, document in enumerate(corpus):
        if y_train[i] == 0:
            input_dict['legit'].append(analyzer(document))
        if y_train[i] == 1:
            input_dict['phish'].append(analyzer(document))

    from DocumentFeatureSelection import interface
    rankings = interface.run_feature_selection(input_dict, method='bns', use_cython=True).convert_score_matrix2score_record()
    for i, item in enumerate(rankings):
        if i< 100:
            vocab.append(item['feature'])
    print(vocab)
    return vocab

def website_tfidf(binormal_separation=False):
        corpus=convert_from_pkl_to_text(args.html_content)
        print("length of list of html content (rows in tfidf matrix): {}".format(len(corpus)))
        vocab = None
        if binormal_separation:
            vocab = binormal_separation(corpus)
        tfidf_matrix=Tfidf_Vectorizer(corpus, vocabulary=vocab)
        if args.features:
                X_features = joblib.load(args.features)
                return Combine_Matrix(X_features,tfidf_matrix)

def url_tokenizer(input):
    return re.split('[^a-zA-Z]', input) 

def url_tfidf(word=True, X_input=None):
    corpus=convert_from_pkl_to_text(args.html_content)
    print("length of list of URLs (rows in tfidf matrix): {}".format(len(corpus)))
    if word:
        tfidf_matrix=Tfidf_Vectorizer(corpus, analyzer='word', tokenizer=url_tokenizer, idf=False)
    else:
        tfidf_matrix=Tfidf_Vectorizer(corpus, analyzer='char', idf=False)
    if X_input:
        return Combine_Matrix(X_input,tfidf_matrix)


def convert_from_pkl_to_text(input_file, url=False):
    text=''
    first_line=1
    corpus=[]
    with open(input_file, 'rb') as f:
        try:
            while(True):
                data=joblib.load(f)
                if url and data.startswith("URL: "):
                    corpus.append(data.split(":", 1)[1])
                elif not url and not data.startswith("URL: "):
                    corpus.append(data)
        except (EOFError):
            pass
    return corpus


def Tfidf_Vectorizer(corpus, analyzer='word', tokenizer=None, idf=True, vocabulary=None):
    if idf:
        vectorizer=TfidfVectorizer(analyzer=analyzer, ngram_range=(1,1), tokenizer=tokenizer,
                                    min_df = 5, stop_words = 'english', sublinear_tf=True, vocabulary=vocabulary)
    else:
                vectorizer=CountVectorizer(analyzer=analyzer, ngram_range=(1,1), tokenizer=tokenizer,
                        min_df = 5, stop_words = 'english', vocabulary=vocabulary)
    tfidf_matrix=vectorizer.fit_transform(corpus)
    joblib.dump(tfidf_matrix, os.path.join(args.output_dir,args.dataset_name+'_tfidf_matrix_combined.pkl'))
    return tfidf_matrix

def Combine_Matrix(m1, m2):
    print(m1.shape, m2.shape)
    X=hstack([m1, m2])
    X=Features_Support.Preprocessing(X)
    return X

if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #X = url_tfidf(False, url_tfidf(True, website_tfidf()))

    X = website_tfidf(False)
    if args.features:
        joblib.dump(X, os.path.join(args.output_dir,args.dataset_name+"_Features_with_Tfidf_processed.pkl"))

