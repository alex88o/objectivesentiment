from sklearn.feature_extraction.text import CountVectorizer  #  <<----- usato per BoW

# Per ogni immagine nel db: estrae tutti i dati testuali, applica una procedura di preprocessing (lemmatizing, stop word removal, positional tagging) e salva in 'preprocessedTokens' in formato JSON.

import sqlite3 as lite
import json
import nltk
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
result_dir		=	project_root + 'scripts/textprocessing/preprocessedTokens/'

data_source_pool = [google_data, mvso_data, places_data]

def extract_words(flickrId):
	out = { 'googlenet': [], 'mvsoenglish': [], 'places205': [], 'description' : [] }
	p = pp =  0
	n = nn = 0
	o = oo = 0
	for idx, db_source in enumerate(data_source_pool):

#		print "\nConnecting to db source:\t" + db_source
		con = lite.connect(db_source)
		cur = con.cursor()
		
		cur.execute("SELECT cnn, top_3_labels FROM Cnndata WHERE FlickrId=?",(flickrId))
		res = cur.fetchall()
		res = res[0]
		labels = json.loads(res[1])
		myout = []
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
				xs = [lemmatizer.lemmatize(xs_el) for xs_el in xs]
				xs = [xs_el for xs_el in xs if not xs_el in stop_words]
				#In questo punto estrai la part of speech considerando gli elementi di xs
				pos_tagging = nltk.pos_tag(xs)
      #				print pos_tagging
#				for xs_el in xs:
				for xs_el in pos_tagging:
					myout.append(xs_el)
			
#		print myout
		#pp, nn, oo = extract_senti_scores(myout)
		p += pp
		n +=nn
		o +=oo

		out[res[0]] = myout
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
	desc_words = [lemmatizer.lemmatize(el) for el in caption]
	desc_words = [el for el in desc_words if not el in stop_words]
	pos_tagging = nltk.pos_tag(desc_words)
	myout = []
	for el in pos_tagging:
		myout.append(el)
#	print myout
	out['description'] = myout
	
	#pp, nn, oo = extract_senti_scores(out['description'])
	p  += pp
	n  += nn
	o  += oo
	print "\nExtracted text:"
	print out

#	print "pos:\t"+ str(p) + "\tobj:\t" + str(o)+ "\tneg:\t" + str(n)

	return out,



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
			n =  list(swn.senti_synsets(tagged[i][0],'n'))[0].neg_score()
		
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

set_con = lite.connect(db_path)
set_cur = set_con.cursor()
set_cur.execute("SELECT FlickrId FROM Image")

rows = set_cur.fetchall()
SKIP = True
for img_idx, id in enumerate(rows):
	print "\n\nProcessing image\t" + str(id[0]) + "\t" + str(img_idx) +  "/" + str(len(rows))
#	id = (u'4331117232',)
#	id = (u'3412282394',)
	print id[0]
	if not id[0] == '3412282394' and SKIP:
		print "skip image"
		continue
	SKIP = False

	ass_words = extract_words(id)
	
	
	js = json.dumps(ass_words)
	
	fname = result_dir + id[0]+".json"
	f = open(fname, 'w')
	f.write(js)
	f.close()
	








