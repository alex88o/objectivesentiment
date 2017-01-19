# This script computes the frequences for the words of the voc.
# For each word, it computes the occurences of when the word is used in either a positive and a negative sentence.


import sqlite3 as lite
import json
from utils import commuteTag, get_normalized_text, extract_single_senti_scores
import nltk, sys
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, sentiwordnet as swn
stop_words = set(stopwords.words('english'))



lemmatizer = WordNetLemmatizer()

project_root	=	'/home/aortis/SentimentAnalysis/'
db_path		=	project_root + 'flickrCrawling/flickrCrossSentiment.db'

google_data	=	project_root + 'scripts/cnn_tools/GoogleNetcnnData.db'
mvso_data	=	project_root + 'scripts/cnn_tools/MVSOcnnData.db'
places_data	=	project_root + 'scripts/cnn_tools/Places205_cnnData.db'
descr_data	=	project_root + 'scripts/description/results/'

tokens_dir	=	'preprocessedTokens/'

th_sentence = 0.6

data_source_pool = [google_data, mvso_data, places_data]


def extract_raw_words(flickrId):
	out = { 'googlenet': [], 'mvsoenglish': [], 'places205': [], 'description' : [] }

	myout = [] # all the words will be stacked here
	for idx, db_source in enumerate(data_source_pool):

#		print "\nConnecting to db source:\t" + db_source
		con = lite.connect(db_source)
		cur = con.cursor()
		
		cur.execute("SELECT cnn, top_3_labels FROM Cnndata WHERE FlickrId=?",(flickrId))
		res = cur.fetchall()
		res = res[0]
		labels = json.loads(res[1])  # top_3_labels
#		myout = []
		print res[0] + ":\t" + res[1]
#		print "Original:\t" + res[1]

		for idx, l in enumerate(labels):
			l = l.replace(",","")
			l = l.replace("-","")	# e.g., seacoast  vs. sea-coast  4331117232
			ls = l.split(' ')
			if res[0] == 'googlenet':
				ls = ls[1:]
			for x in ls:
				xs = x.split("_")
				# TOFIX: passare a lemmatize() un secondo parametro (pos='v' [|n|a|r])
				#xs = [lemmatizer.lemmatize(xs_el) for xs_el in xs]
				#xs = [xs_el for xs_el in xs if not xs_el in stop_words]

				#In questo punto estrai la part of speech considerando gli elementi di xs
				#pos_tagging = nltk.pos_tag(xs)
      #				print pos_tagging

				for xs_el in xs:
					myout.append(xs_el)
			
#		print myout
		#pp, nn, oo = extract_senti_scores(myout)

#		out[res[0]] = myout
		con.close()

	#Get image description

	desc_file_path = descr_data + flickrId[0] + ".json"
#	print "\nOpening file:\t" + desc_file_path
	desc_file = open(desc_file_path,'r')
	desc	  = desc_file.read()

	desc_obj = json.loads(desc)
	caption = desc_obj[0]['caption']
	print "NeuralTalk2:\t" + caption
#	print  caption

	caption = nltk.word_tokenize(caption)

#	desc_words = [lemmatizer.lemmatize(el) for el in caption]
#	desc_words = [el for el in desc_words if not el in stop_words]
#	pos_tagging = nltk.pos_tag(desc_words)
#	myout = []

#	for el in pos_tagging:
	for el in caption:
		myout.append(el)
#	print myout
#	out['description'] = myout
	
	#pp, nn, oo = extract_senti_scores(out['description'])
	print "\nExtracted text:"
	print myout

#	print "pos:\t"+ str(p) + "\tobj:\t" + str(o)+ "\tneg:\t" + str(n)

#	return out
	return myout

def negation_processing(FlickrID):

	# Extract non-processed words form each image
	ass_words = extract_raw_words(FlickrID)
	
	NEG = len(set(ass_words) & set(['no', 'not', 'but', 'however'])) >0
	return NEG
	
	
	
def sentenceSentiment(sentence):

	P_sum = 0
	N_sum = 0
	for w in sentence:
		pos, neg, obj = extract_single_senti_scores(w)
		P_sum += pos
		N_sum += neg
		
	return P_sum, N_sum
	
	
	
set_con = lite.connect(db_path)
set_cur = set_con.cursor()
set_cur.execute("SELECT FlickrId FROM Image")

rows = set_cur.fetchall()
for img_idx, id in enumerate(rows):
	print "\n\nProcessing image\t" + str(id[0]) + "\t" + str(img_idx) +  "/" + str(len(rows))

	# Step 1: negation detection, returns True if the extracted text contains a negative word
	NEG = negation_processing(id)
	
	# Step 3: extract and normalize associated text (i.e., lemmatize, etc.)
	norm_text = get_normalized_text(id[0], tokens_dir)
	
	print norm_text
	
	# Step 4: determine the sentence sentiment and apply negation
	norm_text = list(set(norm_text)) #remove duplicates
	posS, negS = sentenceSentiment(norm_text)
	
	# True if the sentence sentiment is positive ( ^ defines the XOR)
	is_positive = (posS>negS) ^ NEG

	print "Sentence polarity:\t"+ str(is_positive)
	_ = raw_input()
	
	# Step 5: if the threshold condition is satisified,  increment the occurrences of each word in the vocabulary 
	

# Step 6: store the frequences for the word in the vocabulary

	