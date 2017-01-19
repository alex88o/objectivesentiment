# How many images have a flat zero (i.e., all zero values) in their senti representation with sentiVoc.json?

import sqlite3 as lite
import sys, json, pprint
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

project_root	=	'/home/aortis/SentimentAnalysis/'
tokens_dir	=	'preprocessedTokens/'
db_path		=	project_root + 'flickrCrawling/flickrCrossSentiment.db'
#corpus_name	=	'extractedWords.txt'

#  Debug options
DEBUG = False

# EDIT: move this function to utils.py
def commuteTag(treebank_tag):

	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return 'n'
		
con = lite.connect(db_path)
cur = con.cursor()
cur.execute("SELECT FlickrId FROM Image")
rows = cur.fetchall()

lemmatizer = WordNetLemmatizer()
corpus_file = open(corpus_name,'w')

first_word = True
for img_idx, id in enumerate(rows):
	if DEBUG and D_ID:
		id = rows[D_ID]

	print "\n\nProcessing text from image\t" + str(id[0]) + "\t" + str(img_idx) + "/" + str(len(rows))

	fname = tokens_dir + str(id[0]) + ".json"
	f = open(fname,'r')

	obj = f.read()
	obj = json.loads(obj)
	obj = obj[0]
	if DEBUG:
		pprint.pprint(obj)
	f.close()

	#NB: applico lemmatizing perche' nel preprocessing mancava un parametro
	google_data	=	obj['googlenet']
	mvso_data	=	obj['mvsoenglish']
	places_data	=	obj['places205']
	neuraltalk	=	obj['description']

	pool = [google_data, mvso_data, places_data, neuraltalk]
	
	for s_idx,source in enumerate(pool):
		print "\n"
		for t_idx,token in enumerate(source):
			word	=	token[0].lower()
			tag	=	token[1]

			if "'" in word:			# e.g.,  jeweller's
				word = word[:-2]


			# these if statements correct ANP tagging when the adjective is tagged as preposition
			if s_idx == 1:	#mvsoenglish
				if t_idx % 2 == 0:
					tag = 'JJ'
				else:
					tag = 'NN'

		#	if word in ['hat','window']:  #special cases (snow, kite := FW)
		#		tag = 'NN'

#			if word in ['wooden']:
		#		tag = 'JJ'			

			if DEBUG:
				print tag

			if tag == 'CD':
				continue	#CD: cardinal number

			if tag in ['IN','WP','CC','DT','PRP','FW','WRB']:
#				continue	#ignoring prepositions, Wh-pronouns, coordinating conjunctions
				print "Ignoring tag " + tag +" for the word\t" + word + "\t(press to continue)"
#				_ = raw_input()
				pos_tag = 'n'   # as default
			else:
				pos_tag = commuteTag(tag)
			lemma = lemmatizer.lemmatize(word, pos_tag)

			if len(word)<4:
				print word + "("+tag+")\t\t->\t" + lemma + "("+pos_tag+")"
			else:
				print word + "("+tag+")\t->\t" + lemma + "("+pos_tag+")"
