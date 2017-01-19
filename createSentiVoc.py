# Create sentiment vocabulary considering only words with either positive or negative sentiment higher than 0.15 (threshold adopted by Katsurai et al.)

import json
import pprint
from nltk.corpus import sentiwordnet as swn
from utils import extract_single_senti_scores

text_voc_path	=	'textVoc.json'
senti_voc_path	=	'sentiVoc.json'
senti_th	=	0.15	# adopted by Katsurai et al.

f = open(text_voc_path)
V = json.loads(f.read())
f.close()

original_N = len(V)
senti_V = []

DEBUG = True
count = 0
for idx, w in enumerate(V):
	pos, neg, obj = extract_single_senti_scores(w)

	if pos >= senti_th or neg> senti_th:
		count +=1
		senti_V.append(w)
		if DEBUG:
			print w + "\t+"+str(pos) +"\t-" +str(neg) + "\t" + str(obj)
		
f = open(senti_voc_path,'w')
f.write(json.dumps(senti_V))

print str(count) + " of " + str(len(V))

#pprint.pprint(V)

