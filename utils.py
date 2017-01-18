from sklearn.feature_extraction.text import CountVectorizer 


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, sentiwordnet as swn
stop_words = set(stopwords.words('english'))

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
