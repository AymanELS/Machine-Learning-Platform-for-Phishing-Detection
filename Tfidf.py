from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import Features_Support
#import User_options
import Download_url
import configparser
#from collections import deque
import logging

logger = logging.getLogger('root')

config=configparser.ConfigParser()
config.read('Config_file.ini')

## Build the corpus from both the datasets
def build_corpus():
	data=list()
	path=config["Dataset Path"]["path_legit_email"]
	corpus_data_legit = Features_Support.read_corpus(path)
	logger.info("Corpus Data legit: >>>>>>>>>>>>>>> " + str(len(corpus_data_legit)))
	data.extend(corpus_data_legit)
	#for path in config["Dataset Path"][""]path_phish_email:
	path = config["Dataset Path"]["path_phish_email"]
	corpus_data_phish = Features_Support.read_corpus(path)
	logger.info("Corpus Data phish: >>>>>>>>>>>>>>> " + str(len(corpus_data_phish)))
	data.extend(corpus_data_phish)
	return data


def tfidf_emails(corpus):
	#corpus=[]
	#data=build_corpus()
	data=corpus
	#for filepath in data:
	#	try:
	#		print(filepath)
	#		with open(filepath,'r', encoding = "ISO-8859-1") as f:
	#			email=f.read()
#
#	#			body_text, body_html, text_Html, test_text, num_attachment, content_disposition_list, content_type_list, Content_Transfer_Encoding_list, file_extension_list, charset_list, size_in_Bytes =Features_Support.extract_body(email)
#	#			corpus.append(body_text)
#	#	except Exception as e:
	#		print("exception: " + str(e))
	tf= TfidfVectorizer(analyzer='word', ngram_range=(1,1),
                     min_df = 0, stop_words = 'english', sublinear_tf=True)		
	tfidf_matrix = tf.fit_transform(corpus)
	return tfidf_matrix



def Header_Tokenizer(corpus):
	# corpus=[]
	# data=build_corpus()
	data=corpus
	#for filepath in data:
	#	try:
	#		print(filepath)
	#		with open(filepath,'r', encoding = "ISO-8859-1") as f:
	#			email=f.read()
	#			header=Features_Support.extract_header(email)
	#			corpus.append(header)
	#	except Exception as e:
	#		print("exception: " + str(e))
	cv= CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words='english',
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)
	header_tokenizer = cv.fit_transform(corpus)
	return header_tokenizer	


def tfidf_websites(corpus):
	# data=list()
	# corpus=[]
	# corpus_data = Features_Support.read_corpus(config["Dataset Path"]["path_legit_urls"])
	# data.extend(corpus_data)
	# corpus_data = Features_Support.read_corpus(config["Dataset Path"]["path_phish_urls"])
	# data.extend(corpus_data)
	#data=corpus
	#for filepath in data:
	#		print("===================")
	#		print(filepath)
	#		try:
	#			with open(filepath,'r', encoding = "ISO-8859-1") as f:
	#				for rawurl in f:
	#					print("URL >>>>>>>>> {}".format(rawurl))
	#					if rawurl in Bad_URLs_list:
	#						print("This URL will not be considered for further processing because It's registred in out list of dead URLs")
	#					else:
	#						content=Download_url.download_url_content(rawurl)
	#						#print('%s' % ', '.join(map(str, content)))
	#						#print(''.join(content))
	#						corpus.append(''.join(content))
	#		except Exception as e:
	#			print("exception: " + str(e))
	tf= TfidfVectorizer(analyzer='word', ngram_range=(5,5),
                     min_df = 0, stop_words = 'english', sublinear_tf=True)
	tfidf_matrix = tf.fit_transform(corpus)
	return tfidf_matrix
if __name__ == '__main__':
	matrix=tfidf_website()
	logger.info(matrix)
