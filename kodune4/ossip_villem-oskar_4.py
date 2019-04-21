import estnltk
from estnltk import Text
import os
import sys
import nltk
from nltk.tokenize.regexp import WhitespaceTokenizer
import html
import re
import pandas as pd

# 1.1 Korpuse töötlus
def get_tagged_words(sisend, nr):
    margendiga_laused = []
    f = open(sisend)
    read = f.readlines()
    uuedRead = makeNewLines(read)
    li = []

    for rida in uuedRead:
        if rida.strip() == "</s>":
            if len(li) != 0:
                margendiga_laused.append(li)
                li = []
        else:
            rida = html.unescape(rida)
            sona = rida.split(None, 1)[0]
            if nr == 1:
                margend = re.search(r"//.*_(.*)_", rida).group(1).upper()
                li.append((sona, margend))
            if nr == 2:
                li.append(sona)
    return margendiga_laused

def makeNewLines(read):
    uuedRead = []
    for rida in read:
        if (rida.strip() not in ["<s>", "<p>", "</p>"] and len(rida.strip()) != 0):
            uuedRead.append(rida.strip().lower())
    return uuedRead

def loopFiles(path):
    path = os.getcwd() + '/' + path
    folder = os.listdir(path)
    tagged = []
    for file in folder:
        tagged.extend(get_tagged_words(path + '/' + file, 1))
    return tagged

# prindin välja erinevate märgendajate evalutaionid
def print(test_sents):
    result = bigram_tagger_backoff.evaluate(test_sents)
    print('{0:.4f} BigramTagger'.format(result))
    result = trigram_tagger_backoff.evaluate(test_sents)
    print("{0:.4f} TrigramTagger".format(result))
    result = hmm_tagger.evaluate(test_sents)
    print("{0:.4f} HiddenMarkovModelTagger".format(result))

def createDataFrame():
    df = pd.DataFrame()
    df['word'] = [w for s in result for w in s]
    df['bi_tag'] = [w[1] for s in bi_tagged for w in s]
    df['tri_tag'] = [w[1] for s in tri_tagged for w in s]
    df['hmm_tag'] = [w[1] for s in hmm_tagged for w in s]
    return df


tagged_texts = loopFiles(sys.argv[1]) # loen sisse treeninghulga
test_texts = loopFiles(sys.argv[2]) # loen sisse teshulga andmed'

train_sents = tagged_texts
default_tagger = nltk.DefaultTagger("S") #S(nimisona) on koige sagedasem
unigram_tagger_backoff = nltk.UnigramTagger(train_sents, backoff = default_tagger)
bigram_tagger_backoff = nltk.BigramTagger(train_sents, backoff = unigram_tagger_backoff)
trigram_tagger_backoff = nltk.TrigramTagger(train_sents, backoff = bigram_tagger_backoff)
hmm_tagger = nltk.HiddenMarkovModelTagger.train(train_sents)

result = get_tagged_words(os.getcwd() + '/' + sys.argv[3], 2)

bi_tagged = bigram_tagger_backoff.tag_sents(result)
tri_tagged = trigram_tagger_backoff.tag_sents(result)
hmm_tagged = hmm_tagger.tag_sents(result)

#Loome DataFrame'i
df = createDataFrame()
#Kirjutame faili
df.to_csv("ossip_villem-oskar_4.csv", header=False)