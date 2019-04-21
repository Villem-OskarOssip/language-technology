"""
An example of running a gaps test.

Usage:

    python example_test.py <gap file> <candidates' file> <path to corpus directory>

For each sentence in the gap file, outputs the rank of the gap word.

"""
import sys
import estnltk
from estnltk import teicorpus
import nltk.data
from nltk.probability import *
from nltk.util import ngrams
from nltk.util import bigrams
import codecs

#Kodutöö lahendamisel sain abi kursusekaaslaselt

def truecase(model, word):
    try:
        return model[word].max()
    except ValueError:
        return word

def inter(w1, w2, w3, trigram, bigram, unigram, len):
    trigram_f = trigram[(w1, w2, w3)]
    bigram_f = bigram[(w1,w2)]
    if bigram_f == 0 or trigram_f == 0:
        tri = 0
    else:
        tri =  0.85 * (trigram_f/bigram_f)
    bigram_f = bigram[(w2, w3)]
    word_f = unigram[w2]
    if bigram_f == 0 or word_f == 0:
        bi = 0
    else:
        bi = 0.1 * (bigram_f / word_f)
    uni = 0.04 * (unigram[w3]/len)
    return tri + bi + uni + 0.01

if __name__ == '__main__':
    # korpusefailide asukoht
    corp_loc = sys.argv[3]

    xml_files = estnltk.teicorpus.parse_tei_corpora(corp_loc, prefix='', suffix='.xml', target=['artikkel'], encoding = 'utf-8')
    Trigrams = []
    Bigrams = []
    Unigrams = []
    corpus = []

    for file in xml_files:
        corpus += [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(file['text'])]

    labelled = []  #kõik lause sees olevad sõnad (ehk mitte esimesed)
    for sent in corpus:
        for i, w in enumerate(sent):
            if (i != 0):
                labelled.append((w.lower(), w))
    model = nltk.ConditionalProbDist(nltk.ConditionalFreqDist(labelled), nltk.MLEProbDist)

    for sent in corpus:
        sent[0] = truecase(model, sent[0].lower())
        Unigrams.extend(sent)
        Bigrams.extend(bigrams(['<s>'] + sent + ['</s>']))
        Trigrams.extend(ngrams(['<s>', '<s>'] + sent + ['</s>', '</s>'], 3))

    TriFreq = FreqDist(Trigrams)
    BiFreq = FreqDist(Bigrams)
    UniFreq = FreqDist(Unigrams)

    len = sum(UniFreq.values())

    # lünktekst
    gap_fnm = sys.argv[1]
    # vastavate kandidaatide nimistu
    cnd_fnm = sys.argv[2]
    gap_file = codecs.open(gap_fnm, 'r', 'utf-8')
    cnd_file = codecs.open(cnd_fnm, 'r', 'utf-8')

    while 1:
        # read a line from the gap test file
        gap_ln = gap_file.readline().rstrip()
        if not gap_ln:
            break
        items = gap_ln.split()
        word_offset = int(items[0])
        sentence = items[1:]
        gap_word = sentence[word_offset]

        padded_sentence = ['<s>', '<s>'] + sentence + ['</s>', '</s>']
        sentence_gap = padded_sentence[word_offset:word_offset + 5]
        #padded_sentence[2] on puuduolev sõna

        # read a line from the candidates' file
        cnd_ln = cnd_file.readline().rstrip()
        candidates = cnd_ln.split()
        #print(candidates)

        # Score the candidates in a smart way here. As for now, just shuffle randomly.
        candidates.append(gap_word)
        #random.shuffle(candidates)
        ###
        # Ja siia tuleb siis see 'smart way' paremusjärjestuseks
        ###

        h = {}

        for cand in candidates:
            sentence_gap[2] = cand

            p = inter(sentence_gap[0], sentence_gap[1], sentence_gap[2], TriFreq, BiFreq, UniFreq, len) * \
                inter(sentence_gap[1], sentence_gap[2], sentence_gap[3], TriFreq, BiFreq, UniFreq, len) * \
                inter(sentence_gap[2], sentence_gap[3], sentence_gap[4], TriFreq, BiFreq, UniFreq, len)

            h[cand] = p

        variants = sorted(h.items(), key=lambda s: s[1], reverse=True)

        possible = []
        for el in variants:
            possible.append(el[0])

        # output the rank of the original gap word
        for i, c in enumerate(possible):
            if c == gap_word:
                print(i)
                break