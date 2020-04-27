import pandas as pd
import csv
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
import numpy as np
from bs4 import BeautifulSoup
from spacy.lang.en import English
nltk.download('stopwords')

'''
The data preprocessing class. 
This class has modules for reading the data, cleaning the data, encoding inputs and outputs, building ranking systems using Cosine Simmilarity scores.
'''

class Preprocess():

    def __init__(self, filepath, columns):
        self.filepath = filepath
        self.columns = columns
        self.columns.append('answerText')
        self.columns.append('upvotes')
        self.contraction_mapping = {}
        self.stopWords = set(stopwords.words('english'))
        self.word_embeddings = {}
        f = open('glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.word_embeddings[word] = coefs
        f.close()

    def readData(self):
        self.data = pd.read_csv(self.filepath, usecols=self.columns)
        self.data.dropna(axis=0, inplace=True)
        with open('ContractMapping') as csv_file:
            self.contractMapping = csv.reader(csv_file, delimiter = ',')
            for row in self.contractMapping:
                self.contraction_mapping[row[0].split(':')[0]] = row[0].split(':')[1]
        self.journal = self.data['questionTitle']


    def cleanJournal(self, text):
        newString = text.lower()
        newString = BeautifulSoup(newString, 'lxml').text
        newString = re.sub(r'\([^)]*\)', '', newString)
        newString = re.sub('"', '', newString)
        newString = ' '.join([self.contraction_mapping[t] if t in self.contraction_mapping else t for t in newString.split(" ")])
        newString = re.sub(r"'s\b", "", newString)
        newString = re.sub("[^a-zA-Z]", " ", newString)
        tokens = [w for w in newString.split() if not w in self.stopWords]
        long_words = []
        for i in tokens:
            if len(i) >= 3:  # removing short word
                long_words.append(i)
        return (" ".join(long_words)).strip()

    def cleanReflection(self, text):
        newString = re.sub('"', '', text)
        newString = ' '.join([self.contraction_mapping[t] if t in self.contraction_mapping else t for t in newString.split(" ")])
        newString = re.sub(r"'s\b", "", newString)
        newString = re.sub("[^a-zA-Z]", " ", newString)
        newString = newString.lower()
        tokens = newString.split()
        newString = ''
        for i in tokens:
            newString = newString + i + ' '
        return newString

    def cleanText(self):
        self.cleanedJournal = []
        for journal in self.data[self.columns[0]]:
            self.cleanedJournal.append(self.cleanJournal(journal))

        self.cleanedReflection = []
        for reflection in self.data[self.columns[1]]:
            self.cleanedReflection.append(self.cleanReflection(reflection))


        self.data['cleanedJournal'] = self.cleanedJournal
        self.data['cleanedReflection'] = self.cleanedReflection


    def preprocessMaps(self):
        questionTitle = {}
        for title in self.cleanedReflection:
            if len(title) != 0:
                v = sum([self.word_embeddings.get(w, np.zeros((100,))) for w in title.split()]) / (len(title.split()) + 0.001)
            else:
                v = np.zeros((100,))
            questionTitle[title] = v
        np.save('title.npy', questionTitle)


        sentences = {}
        self.dataProcess = self.data.groupby('questionTitle').max()
        for i in range(len(self.dataProcess)):
            sentences[self.dataProcess.iloc[i]['cleanedReflection']] = self.dataProcess.iloc[i]['answerText']
        np.save('answer.npy', sentences)


    def preprocess(self, testSize):
        self.data.sample(frac=1)
        x = self.data['cleanedJournal']
        y = self.data['cleanedReflection']
        spacyEn = English()
        self.en_words = Counter()
        self.de_words = Counter()
        self.en_inputs = []
        self.de_inputs = []
        for i in range(len(x)):
            en_tokens = spacyEn(x.iloc[i])
            de_tokens = spacyEn(y.iloc[i])

            if len(en_tokens) == 0 or len(de_tokens) == 0:
                continue

            for token in en_tokens:
                self.en_words.update([token.text.lower()])
            self.en_inputs.append([token.text.lower() for token in en_tokens] + ['_EOS'])
            for token in de_tokens:
                self.de_words.update([token.text.lower()])
            self.de_inputs.append([token.text.lower() for token in de_tokens] + ['_EOS'])

        self.en_words = ['_SOS', '_EOS', '_UNK'] + sorted(self.en_words,
                                                        key=self.en_words.get,
                                                        reverse=True)
        self.en_w2i = {o:i for i,o in enumerate(self.en_words)}
        self.en_i2w = {i:o for i,o in enumerate(self.en_words)}

        self.de_words = ['_SOS', '_EOS', '_UNK'] + sorted(self.de_words,
                                                        key=self.de_words.get,
                                                        reverse=True)
        self.de_w2i = {o:i for i,o in enumerate(self.de_words)}
        self.de_i2w = {i:o for i,o in enumerate(self.de_words)}

        trainSize = len(x) - int(len(x)*testSize)
        self.en_tr_inputs = self.en_inputs[:trainSize]
        self.en_ts_inputs = self.en_inputs[trainSize:]

        self.de_tr_inputs = self.de_inputs[:trainSize]
        self.de_ts_inputs = self.de_inputs[trainSize:]

        for i in range(len(self.en_tr_inputs)):
            en_sentence = self.en_tr_inputs[i]
            de_sentence = self.de_tr_inputs[i]
            self.en_tr_inputs[i] = [self.en_w2i[word] for word in en_sentence]
            self.de_tr_inputs[i] = [self.de_w2i[word] for word in de_sentence]

        for i in range(len(self.en_ts_inputs)):
            en_sentence = self.en_ts_inputs[i]
            de_sentence = self.de_ts_inputs[i]
            self.en_ts_inputs[i] = [self.en_w2i[word] for word in en_sentence]
            self.de_ts_inputs[i] = [self.de_w2i[word] for word in de_sentence]

    def encodeText(self, text):
        text = self.cleanJournal(text)
        words = text.split()
        self.textInput = []
        for i in range(len(words)):
            if words[i].lower() in self.en_w2i.keys():
                self.textInput.append(self.en_w2i[words[i].lower()])
            else:
                self.textInput.append(self.en_w2i['_UNK'])

    def fetchReply(self, text):
        sentence_vectors = []
        text = text.lower()
        for i in text:
            if len(i) != 0:
                v = sum([self.word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors = v
        title = np.load('title.npy', allow_pickle='TRUE').item()
        scores = {}
        for sent in title.keys():
            c = 0
            vec = title[sent]
            for i in range(len(sentence_vectors)):
                c+= sentence_vectors[i]*vec[i]
            cosine = c/float((sum(sentence_vectors**2)*sum(vec**2))**0.5)
            scores[cosine] = sent
        answers = np.load('answer.npy', allow_pickle='TRUE').item()
        reply = ''
        for i in sorted(scores.keys(), reverse=True):
            reply = answers[scores[i]]
            break
        return reply











