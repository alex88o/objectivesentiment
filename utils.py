from sklearn.feature_extraction.text import CountVectorizer 

import json
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords, sentiwordnet as swn

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def mat_dataset_to_dict(dataset):
	
	dataset = dataset['RND_DATA']
	shuffle_no = dataset['shuffle_no'][0][0][0][0]
	train_set = dataset['train_set'][0][0][0]
	train_set = [str(int(x)) for x in train_set]
	train_label = dataset['train_label'][0][0][0]

	test_set = dataset['test_set'][0][0][0]
	test_set = [str(int(x)) for x in test_set]
	test_label = dataset['test_label'][0][0][0]
	
	D = {}
	D['shuffle_no'] = shuffle_no
	D['train_set'] = train_set
	D['train_label'] = train_label
	D['test_set'] = test_set
	D['test_label'] = test_label
	
	return D


def get_normalized_text(FlickrID, tokens_dir):

	fname = tokens_dir + str(FlickrID) + ".json"
	f = open(fname,'r')

	obj = f.read()
	obj = json.loads(obj)
	obj = obj[0]
	f.close()

	#NB: applico lemmatizing perche' nel preprocessing mancava un parametro
	google_data	=	obj['googlenet']
	mvso_data	=	obj['mvsoenglish']
	places_data	=	obj['places205']
	neuraltalk		=	obj['description']

	pool = [google_data, mvso_data, places_data, neuraltalk]
	
	associated_norm_words = []
	
	for s_idx,source in enumerate(pool):
	#	print "\n"
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

			if tag == 'CD':
				continue	#CD: cardinal number

#			if tag in ['IN','WP','CC','DT','PRP','FW','WRB']:
#				pos_tag = 'n'   # as default
#			else:
			#	pos_tag = commuteTag(tag)
			
			#commented if statement handled by the commuteTag function
			pos_tag = commuteTag(tag)
			lemma = lemmatizer.lemmatize(word, pos_tag)
			
			# Now the word "lemma" is in the same form as in the vocabulary
			associated_norm_words.append(lemma)

	return associated_norm_words



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
		

def extract_single_senti_scores(word):
	pscore = 0
	nscore = 0
	objscore = 0
	
	if len(swn.senti_synsets(word))>0:
		s_scores = list(swn.senti_synsets(word))[0]
		pscore	= s_scores.pos_score()
		objscore= s_scores.obj_score()
		nscore	= s_scores.neg_score()

	return pscore, nscore, objscore


def is_objective(w):
	p,n,o = extract_single_senti_scores(w)
	if o>p and o>n:
		return True
	else:
		return False
# takes a list of [word, tag]
def extract_senti_scores(tagged):
	pscore = 0
	nscore = 0
	objscore = 0

	for i in range(0,len(tagged)):
		print "evaluating\t" +tagged[i][0]
		p = o = n = 0
		if 'NN' in tagged[i][1] and len(swn.senti_synsets(tagged[i][0],'n'))>0:
			p = list(swn.senti_synsets(tagged[i][0],'n'))[0].pos_score()
			o = list(swn.senti_synsets(tagged[i][0],'n'))[0].obj_score()
			n = list(swn.senti_synsets(tagged[i][0],'n'))[0].neg_score()
		
			pscore+=(list(swn.senti_synsets(tagged[i][0],'n'))[0]).pos_score() #positive score of a word
			nscore+=(list(swn.senti_synsets(tagged[i][0],'n'))[0]).neg_score()  #negative score of a word
			objscore+=(list(swn.senti_synsets(tagged[i][0],'n'))[0]).obj_score()  
		elif 'VB' in tagged[i][1] and len(swn.senti_synsets(tagged[i][0],'v'))>0:
			p = list(swn.senti_synsets(tagged[i][0],'v'))[0].pos_score()
			o = list(swn.senti_synsets(tagged[i][0],'v'))[0].obj_score()
			n = list(swn.senti_synsets(tagged[i][0],'v'))[0].neg_score()
		
			pscore+=(list(swn.senti_synsets(tagged[i][0],'v'))[0]).pos_score()
			nscore+=(list(swn.senti_synsets(tagged[i][0],'v'))[0]).neg_score()
			objscore+=(list(swn.senti_synsets(tagged[i][0],'v'))[0]).obj_score()  
		elif 'JJ' in tagged[i][1] and len(swn.senti_synsets(tagged[i][0],'a'))>0:
			p = list(swn.senti_synsets(tagged[i][0],'a'))[0].pos_score()
			o = list(swn.senti_synsets(tagged[i][0],'a'))[0].obj_score()
			n = list(swn.senti_synsets(tagged[i][0],'a'))[0].neg_score()
		
			pscore+=(list(swn.senti_synsets(tagged[i][0],'a'))[0]).pos_score()
			nscore+=(list(swn.senti_synsets(tagged[i][0],'a'))[0]).neg_score()
			objscore+=(list(swn.senti_synsets(tagged[i][0],'a'))[0]).obj_score()  
		elif 'RB' in tagged[i][1] and len(swn.senti_synsets(tagged[i][0],'r'))>0:
			p = list(swn.senti_synsets(tagged[i][0],'r'))[0].pos_score()
			o = list(swn.senti_synsets(tagged[i][0],'r'))[0].obj_score()
			n = list(swn.senti_synsets(tagged[i][0],'r'))[0].neg_score()

		
			pscore+=(list(swn.senti_synsets(tagged[i][0],'r'))[0]).pos_score()
			nscore+=(list(swn.senti_synsets(tagged[i][0],'r'))[0]).neg_score()
			objscore+=(list(swn.senti_synsets(tagged[i][0],'r'))[0]).obj_score()  

		print "pos:\t"+ str(p) + "\tobj:\t" + str(o)+ "\tneg:\t" + str(n)

#		print "pos:\t" +str(pscore)+"\tobj:\t" + str(objscore)+"\tneg:\t" + str(nscore)
	return pscore, nscore, objscore
