import os
import random
from nltk.tokenize import sent_tokenize
from keras_preprocessing import sequence
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

random.seed(42)
toke = Tokenizer(char_level=False)
autorid = []
WINDOW_SIZE = 10

def train(file1, file2, file3):
    fails = [file1,file2,file3]

    for file in fails:
        autorid.append(os.path.splitext(os.path.basename(file))[0])
    print(autorid)

    korpused = []
    for file in fails:
        korpused.append(sent_tokenize(loadFile(file)))

    train, test = makeTrainTest(korpused)
    X_train, y_train, tok = preprocess(train)
    X_test, y_test, _ = preprocess(test)
    return doMagic(X_train, y_train, tok)

def doMagic(X_train, y_train, tok):
    vocabulary_size = len(tok.word_index) + 1
    mudel = Sequential()
    mudel.add(Embedding(vocabulary_size, 100, input_length=WINDOW_SIZE))
    mudel.add(Dropout(0.2))
    mudel.add(LSTM(35, return_sequences=True))
    mudel.add(LSTM(35, return_sequences=False))
    mudel.add(Dense(len(autorid), activation='softmax'))
    mudel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    try:
        mudel.fit(X_train, y_train, epochs=2, batch_size=64)
    except KeyboardInterrupt:
        return mudel
    return mudel

def makeTrainTest(korpused):
    train = []
    test = []
    for korpus in korpused:
        suurus = round(len(korpus) * 0.2)
        train.append(korpus[suurus:])
        test.append(korpus[:suurus])
    return train, test

def predict(model, sent):
    andmed, autor, tok = preprocess([sent], toke)
    prediction = []
    all = model.predict(andmed)[0]
    for c in all:
        prediction.append(float(c))
    koefitsient = max(prediction)
    return prediction.index(koefitsient) + 1

def preprocess(korpused, tok=None):
    sentences = []
    y = []
    i = 0
    for korpus in korpused:
        sentences += korpus
        y.extend([i] * len(korpus))
        i += 1

    if tok is None:
        tok = toke
        tok.fit_on_texts(sentences)

    sent_seqs = tok.texts_to_sequences(sentences)
    X = sequence.pad_sequences(sent_seqs, maxlen=WINDOW_SIZE)
    return X, to_categorical(y, num_classes=len(autorid)), tok

def loadFile(file):
    data = open(file, 'r+', encoding='utf-8').read()
    return data.replace("\n", " ")

m = train('shakespeare.txt', 'poe.txt', 'janeausten.txt')
lause = ""
print(round(predict(m, lause)))
