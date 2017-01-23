# How many images have a flat zero (i.e., all zero values) in their senti representation with sentiVoc.json?
# You can change the behaviour of this script by the parameter n

# Usage:	python 	sentiWordStatistics.py	n
#		n:	max number of words to have a flat representation 


import sqlite3 as lite
import sys, json, pprint
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

project_root	=	'/home/aortis/SentimentAnalysis/'
tokens_dir	=	'preprocessedTokens/'
db_path		=	project_root + 'flickrCrawling/flickrCrossSentiment.db'

n = int(sys.argv[1])

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

fV = open('sentiVoc.json')
txt_V = fV.read()
fV.close()

senti_V = json.loads(txt_V)
senti_Set = set(senti_V)

count = 0
idx_single_words =[]
for img_idx, id in enumerate(rows):

#	print "\n\nProcessing text from image\t" + str(id[0]) + "\t" + str(img_idx) + "/" + str(len(rows))

	ass_words = [] # list of image's associated words

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
#		print "\n"
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
#				print "Ignoring tag " + tag +" for the word\t" + word + "\t(press to continue)"
#				_ = raw_input()
				pos_tag = 'n'   # as default
			else:
				pos_tag = commuteTag(tag)
			lemma = lemmatizer.lemmatize(word, pos_tag)

			ass_words.append(lemma)

	#Compare ass_words with senti vocabulary
	ass_Set = set(ass_words)
	
	#Intersection between vocabulary and words associated to the image
	intersect = ass_Set & senti_Set
	if len(intersect) < n:	#counts the number of images with a number of associated words lower than n
		count +=1
		if n == 2:	#insert into idx_single_words the words which are used as a 1-hot like representation
			for w_idx, w in enumerate(senti_V):
				if w in intersect:
					idx_single_words.append(w_idx)
					val = ass_words.count(w)	# count value of that bin in the BoW representation
					print "Found\t"+w+"\tin\t"+str(w_idx)+"\tvalue:\t"+str(val)

# if n == 2 the number of non unique 'single words' give us the amount of images with similar 1-hot like representation, hence the discriminative lacks of if
if n == 2:
	single_Set = set(idx_single_words)
	print "Images with similar representations:\t"+ str(len(idx_single_words)-len(single_Set))
if n == 1:
	print "Null representation images (images with all zeros representation):\t" + str(count)
else:
	print "Poor representation images (images with less than "+str(n)+" voc. words):\t" + str(count)
