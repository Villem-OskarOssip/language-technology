#!/usr/bin/env python
# coding: utf-8

# LTAT.01.002 Keeletehnoloogia (2019 kevad)
# Kodutöö nr 2. Tekstide liigitamine

import random
import os
import nltk
from sklearn.naive_bayes import MultinomialNB
import estnltk
from nltk.probability import *
from estnltk import Text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.model_selection import KFold

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

labels = ["www.advent.ee", "www.bioneer.ee", "www.lapsemure.ee", "www.naisteleht.ee", "arvamus.postimees.ee"]

# Korpuse sisselugemine (korpus asub programmifaili kausta alamkaustas)
def loaddata(corpus):
	columns = ['label', 'text']
	data = pd.DataFrame(columns = columns)
	for f in os.listdir(os.getcwd() + "/" + corpus):
		df = pd.read_csv(os.getcwd() + "/" + corpus + "/" + f, delimiter="\t", index_col=None, header=None, names=columns)
		data = data.append(df)
	return data

def bagOfWordsprinter(tegevus):
	#Paremaks visualiseerimiseks loodud abimeetod
	print("###########################")
	print("BagOfWords")
	if tegevus == 0:
		print("(ilma lisa parameetriteta)\n")
	if tegevus == 1:
		print("(+ lemmad)\n")
	if tegevus == 2:
		print("(+ stoppsõnad)\n")
	if tegevus == 3:
		print("(+ lemmad + stoppsõnad)\n")

def doc2VecPrinter():
	# Paremaks visualiseerimiseks loodud abimeetod
	print("###########################")
	print("Doc2Vec \n")

def stopWords(limit, tegevus):
	#Kui tahame kasutada stoppsõnu (2 või 3)
	if tegevus == 2 or tegevus == 3:
		columns = ['arv','sona','NaN']
		data = pd.DataFrame(columns = columns)
		df = pd.read_csv(os.getcwd() + "/" +  "sagedussonastik_lemmad_kahanev.txt", delimiter=" ", index_col=None, header=None, names=columns)
		data = data.append(df)
		arv = 0
		sonad = []
		for i, rida in data.iterrows():
			kat = rida['arv']
			sisu = rida['sona']
			if arv < limit:
				sonad.append(sisu)
				arv += 1
		#print(sonad)
	else:
		sonad = []
	return sonad

# Treenimine
def bagOfWords(tekstid, stopWordsLimit, tegevus):
	global globalLemmad
	if tegevus == 1 or tegevus == 3:
		globalLemmad = True
	else:
		globalLemmad = False
	bagOfWordsprinter(tegevus)
	allcounts = {}
	count_vects = {}
	#Stopp sõnade list
	stopwords = stopWords(stopWordsLimit, tegevus)
	#Tulemuste raamatukogu
	results = {'all': []}
	for i, row in tekstid.iterrows():
		text = row["text"]
		label = row["label"]
		#Kui tahame lemmatiseerida(1 või 3)
		if tegevus == 1 or tegevus == 3:
			#print("Kontroll 1")
			text = " ".join(Text(text).lemmas)
		results['all'].append(text)
		if label in results:
			results[label].append(text)
		else:
			results[label] = [text]

	#Käime üle kõikide tulemuste ja teeme neist vektorid uude raamatukokku
	for i in results.keys():
		count_vect = CountVectorizer(stop_words=stopwords)
		count_vects[i] = count_vect
		allcounts[i] = count_vect.fit_transform(results[i])

	#Arvestame iga sõna esinemise osakaalu
	tfidf_transformer = TfidfTransformer()
	count_tfidf = tfidf_transformer.fit_transform(allcounts['all'])
	#Õpetame mudelit
	model = MultinomialNB().fit(count_tfidf, tekstid["label"])
	return [model, count_vects['all'], tfidf_transformer]


def doc2Vec(tekstid):
	doc2VecPrinter()
	#Loome gensin jaoks listi
	documents = []
	for i, rida in tekstid.iterrows():
		#Loeme sisse tekstid mudeli
		documents.append(gensim.models.doc2vec.TaggedDocument(Text(rida['text']).word_texts, [rida['label']]))
	#Treenime doc2vec mudelit (parameetrid votsin nii nagu tunni materjalides olid)
	model = gensim.models.doc2vec.Doc2Vec(documents, vector_size=100, window=8, min_count=5, workers=4)
	return model

#def kfold(tekstid):
	#kfold = KFold(n_splits=2, random_state=None, shuffle=False)
	#for a, b in kfold.split(tekstid):

# Ennustamine
def predict(mudel, sample):
	#Siin on erinevate mudelite ennustamised
	# predictDoc2Vec(model, text)
	# predictBagOfWords(model, sample)
	return predictBagOfWords(mudel, sample)

def predictBagOfWords(mudel, sample):
	#Praktikumi BagOfWords saadud ennustamine
	count_vect = mudel[1]
	tfidf_transformer = mudel[2]
	models = mudel[0]
	global globalLemmad
	if globalLemmad:
		sample = " ".join(Text(sample).lemmas)
	#Õpetame mudelit test hulgaga
	X_test_counts = count_vect.transform([sample])
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)
	# Ennustame mudeli põhjal
	ennustused = models.predict(X_test_tfidf)
	return ennustused[0]

def predictDoc2Vec(mudel, text):
	#Praktikumi gensim materjalist saadud ennustamine
	inferred_docvec = mudel.infer_vector(Text(text).word_texts)
	#[0][0] mõtte sain kursusekaaslaselt
	return mudel.docvecs.most_similar([inferred_docvec], topn=2)[0][0]

#Learn
def learn(corpus):
	# doc2Vec(corpus)
	# bagOfWords(corpus, stopWordsLimit, tegevus) tegevus:
	# N: bagOfWords(corpus, 20, 3)
	# 0 (ära kasuta midagi),
	# 1 (kasuta lemmasid),
	# 2 (kasuta stopp sõnu),
	# 3 (kasuta mõlemat)
	return bagOfWords(corpus, 50, 0)


# Hindamine (testhulgal)
# Sisend: treenimisfunktsiooni väljundist saadud mudel ja testkorpuses olev info DataFrame'ina
def evaluate(model, testset):
	correct = 0
	for i, row in testset.iterrows():
		rightAnswer = row['label']
		text = row['text']
		prediction = predict(model, text)
		#print(rightAnswer, prediction)
		if rightAnswer == prediction:
			correct += 1
	
	print("Täpsus: {0:}%".format(100.0 * correct/len(testset)))


def evaluateDoc2Vec(model, testset):
	correct = 0
	for i, row in testset.iterrows():
		rightAnswer = row['label']
		text = row['text']
		prediction = predictDoc2Vec(model, text)
		# print(rightAnswer, prediction)
		if rightAnswer == prediction:
			correct += 1

	print("Täpsus: {0:}%".format(100.0 * correct / len(testset)))

def doEverything():
	#Käime kõik mudelid läbi
	#Sedasi ei pea mitukorda lugema sisse train ja test andmeid
	trainSet = loaddata("a_train")
	testSet = loaddata("a_test")

	bagOfWordsmodel1 = bagOfWords(trainSet, 50, 0)
	evaluate(bagOfWordsmodel1, testSet)

	bagOfWordsmodel2 = bagOfWords(trainSet, 50, 1)
	evaluate(bagOfWordsmodel2, testSet)

	bagOfWordsmodel3 = bagOfWords(trainSet, 50, 2)
	evaluate(bagOfWordsmodel3, testSet)

	bagOfWordsmodel4 = bagOfWords(trainSet, 50, 3)
	evaluate(bagOfWordsmodel4, testSet)

	doc2Vecmodel = doc2Vec(trainSet)
	evaluateDoc2Vec(doc2Vecmodel, testSet)

#Globaalne muutuja, mis aitab lemmade puhul paremini ennustada meeotdis predictBagOfWords()
globalLemmad = False

doEverything()
#BagOfWords  => 58.3%
#BagOfWords + lemmad => 57.3%
#BagOfWords + stoppsõnad => 63%
#BagOfWords + stoppsõnad + lemmad => 64.3%
#Word2Vec => 91%

#trainSet = loaddata("a_train")
#model = learn(trainSet)
#testSet = loaddata("a_test")
#evaluate(model, testSet
