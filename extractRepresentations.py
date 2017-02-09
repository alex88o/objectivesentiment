import sqlite3 as lite
import json
import pprint
from scipy.io import loadmat
from utils import commuteTag, get_normalized_text, extract_single_senti_scores, mat_dataset_to_dict
import sys

from sklearn.feature_extraction.text import CountVectorizer

project_root	=	'/home/aortis/SentimentAnalysis/'
db_path		=	project_root + 'flickrCrawling/flickrCrossSentiment.db'

tokens_dir	=	'preprocessedTokens/'


# Textual representation
fnameV = 'textVoc.json'
fV = open(fnameV,'r')
V = json.loads(fV.read())
fV.close()

# Sentiment representation
fnameV = 'sentiVoc.json'
fV = open(fnameV,'r')
sentiV = json.loads(fV.read())
fV.close()

#Revisited representation (frequencies computed with dataset_split = 1 )
fname = 'revisitedScores.json'
f = open(fname,'r')
revisitedScores = json.loads(f.read())
f.close()


# We are splitting the data to train and test files for memory issues
dataset_split = 1
split_path = 'Dataset/' + 'shuffleDataset_' + str(dataset_split) + '.mat'

dataset = loadmat(file_name = split_path, chars_as_strings = True)
# Take the list of training images' FlckrIds
dataset = mat_dataset_to_dict(dataset)
train_ids = dataset['train_set']
test_ids = dataset['test_set']


set_con = lite.connect(db_path)
set_cur = set_con.cursor()
set_cur.execute("SELECT FlickrId FROM Image")

rows = set_cur.fetchall()


TEXTUAL_TRAIN 	=	[]
TEXTUAL_TEST 		=	[]
SENTIMENT_TRAIN	=	[]
SENTIMENT_TEST	=	[]
REVISITED_TRAIN 	= 	[]
REVISITED_TEST 	= 	[]


vectorizer_1 = CountVectorizer(input = 'content', vocabulary = V)
vectorizer_2 = CountVectorizer(input = 'content', vocabulary = sentiV)
"""
	print len(vectorizer_1.vocabulary)
	print vectorizer_1.vocabulary[1:10]
	print vectorizer_1.vocabulary[-10:]


	print V[0:10]
	print V[-10:]
	print len(V)
	print len(vectorizer_1.vocabulary)

	print vectorizer_1.vocabulary.index('abacus')
	print vectorizer_1.vocabulary.index('zoo')
	print vectorizer_1.vocabulary.index('zucchini')


	sys.exit(0)
"""

IMG = {}
for img_idx, id in enumerate(rows):
	print "\n\nProcessing image\t" + str(id[0]) + "\t" + str(img_idx) +  "/" + str(len(rows))


	norm_text = get_normalized_text(id[0], tokens_dir)
	document = " ".join(norm_text)
 	# Textual representation (without SVD)
	X = vectorizer_1.transform([document])
	BoW_repr = X.toarray()
	BoW_repr = BoW_repr[0]
	text_repr = BoW_repr[:]  #needed for the revisited representation

	IMG['id'] = id[0]
	IMG['textBoW'] = BoW_repr.tolist()	# np array to list for json serialization

	if id[0] in train_ids:
		TEXTUAL_TRAIN.append(IMG)
	elif id[0] in test_ids:
		TEXTUAL_TEST.append(IMG)
	IMG = {}
	
	#print sum( x > 0 for x in BoW_repr)
	
	# Sentiment representation (without SVD)
	X = vectorizer_2.transform([document])
	BoW_repr = []
	BoW_repr = X.toarray()
	BoW_repr = BoW_repr[0]
	
	IMG['id'] = id[0]
	IMG['sentiBoW'] = BoW_repr.tolist()
	if id[0] in train_ids:
		SENTIMENT_TRAIN.append(IMG)
	elif id[0] in test_ids:
		SENTIMENT_TEST.append(IMG)
	IMG = {}
	
	#print sum( x > 0 for x in BoW_repr)
	
	
	# Revisited representation
	W = [0] * len(text_repr)	#list of zeros
	for i, wordFreq in enumerate(text_repr):
		
		if wordFreq==0:	# wordfFreq := [freqInDataset, freqInPosSent, freqInNegSent]
			W[i] = 0.0
			continue
			
		scores = revisitedScores[i]		

		#print scores

		sortedScores = scores[:]	#copy by value
		sortedScores.sort()
		
		#print sortedScores
		#_ = raw_input()
		

#		print sortedScores
#		print wordFreq

		M = sortedScores[-1]  #take the max
		M = float(M)
		if M == sortedScores[-2] or M == scores[1]:	# if the max is not unique (the word is not positive or negative), or its equal to the objective score, the feature element is zero
			W[i] = 0					#objective word
		
		elif M == scores[0]:
			W[i] = M*wordFreq			#positive word
		
		else:
			W[i] = -1*M*wordFreq		#negative word
		
#		print W[i]
		
		#if  W[i]>0.0:
		#	print W[i]
		#	_ = raw_input()

#	print W[:10]
#	print W[-10:]
	
	
	IMG['id'] = id[0]
	IMG['revisitedBoW'] = W
	if id[0] in train_ids:
		REVISITED_TRAIN.append(IMG)
	elif id[0] in test_ids:
		REVISITED_TEST.append(IMG)
	IMG = {}


	
print len(TEXTUAL_TRAIN)
print len(TEXTUAL_TEST)
print len(SENTIMENT_TRAIN)
print len(SENTIMENT_TEST)
print len(REVISITED_TRAIN)
print len(REVISITED_TEST)


fname = 'DS' + str(dataset_split) + '_TEXTUAL_TRAIN.json'
f = open(fname,'w')
f.write(json.dumps(TEXTUAL_TRAIN))
f.close()

fname = 'DS' + str(dataset_split) + '_TEXTUAL_TEST.json'
f = open(fname,'w')
f.write(json.dumps(TEXTUAL_TEST))
f.close()

fname = 'DS' + str(dataset_split) + '_SENTIMENT_TRAIN.json'
f = open(fname,'w')
f.write(json.dumps(SENTIMENT_TRAIN))
f.close()

fname = 'DS' + str(dataset_split) + '_SENTIMENT_TEST.json'
f = open(fname,'w')
f.write(json.dumps(SENTIMENT_TEST))
f.close()

fname = 'DS' + str(dataset_split) + '_REVISITED_TRAIN.json'
f = open(fname,'w')
f.write(json.dumps(REVISITED_TRAIN))
f.close()

fname = 'DS' + str(dataset_split) + '_REVISITED_TEST.json'
f = open(fname,'w')
f.write(json.dumps(REVISITED_TEST))
f.close()

#dataset_split = 1
#split_path = 'Dataset/' + 'shuffleDataset_' + str(dataset_split) + '.mat'
#dataset = loadmat(file_name = split_path, chars_as_strings = True)
# Take the list of training images' FlckrIds
#dataset = mat_dataset_to_dict(dataset)
#train_ids = dataset['train_set']
