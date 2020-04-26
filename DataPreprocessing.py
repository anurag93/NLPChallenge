import pandas as pd
import csv
import nltk
import re
#from gensim.models import Word2Vec
from collections import Counter
from nltk.corpus import stopwords
#import numpy as np
#import pickle
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
#import spacy
from spacy.lang.en import English
nltk.download('stopwords')
class Preprocess():

    def __init__(self, filepath, columns):
        self.filepath = filepath
        self.columns = columns
        self.contraction_mapping = {}
        self.stopWords = set(stopwords.words('english'))

    def readData(self):
        self.data = pd.read_csv(self.filepath, usecols=self.columns)
        self.data.dropna(axis=0, inplace=True)
        with open('ContractMapping') as csv_file:
            self.contractMapping = csv.reader(csv_file, delimiter = ',')
            for row in self.contractMapping:
                self.contraction_mapping[row[0].split(':')[0]] = row[0].split(':')[1]

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
            if len(i) >= 3:
                newString = newString + i + ' '
        return newString

    def cleanText(self):
        cleanedJournal = []
        for journal in self.data[self.columns[0]]:
            cleanedJournal.append(self.cleanJournal(journal))

        cleanedReflection = []
        for reflection in self.data[self.columns[1]]:
            cleanedReflection.append(self.cleanReflection(reflection))

        self.data['cleanedJournal'] = cleanedJournal
        self.data['cleanedReflection'] = cleanedReflection


    def preprocess(self, testSize):

        x_tr, x_val, y_tr, y_val = train_test_split(self.data['cleanedJournal'], self.data['cleanedReflection'], test_size=testSize,
                                                    random_state=0, shuffle=True)
        spacyEn = English()
        self.en_tr_words = Counter()
        self.de_tr_words = Counter()
        self.en_tr_inputs = []
        self.de_tr_inputs = []
        for i in range(len(x_tr)):
            en_tokens = spacyEn(x_tr.iloc[i])
            de_tokens = spacyEn(y_tr.iloc[i])

            if len(en_tokens) == 0 or len(de_tokens) == 0:
                continue

            for token in en_tokens:
                self.en_tr_words.update([token.text.lower()])
            self.en_tr_inputs.append([token.text.lower() for token in en_tokens] + ['_EOS'])
            for token in de_tokens:
                self.de_tr_words.update([token.text.lower()])
            self.de_tr_inputs.append([token.text.lower() for token in de_tokens] + ['_EOS'])

        self. en_tr_words = ['_SOS', '_EOS', '_UNK'] + sorted(self.en_tr_words,
                                                        key=self.en_tr_words.get,
                                                        reverse=True)
        self.en_tr_w2i = {o:i for i,o in enumerate(self.en_tr_words)}
        self.en_tr_i2w = {i:o for i,o in enumerate(self.en_tr_words)}

        self.de_tr_words = ['_SOS', '_EOS', '_UNK'] + sorted(self.de_tr_words,
                                                        key=self.de_tr_words.get,
                                                        reverse=True)
        self.de_tr_w2i = {o:i for i,o in enumerate(self.de_tr_words)}
        self.de_tr_i2w = {i:o for i,o in enumerate(self.de_tr_words)}

        for i in range(len(self.en_tr_inputs)):
            en_sentence = self.en_tr_inputs[i]
            de_sentence = self.de_tr_inputs[i]
            self.en_tr_inputs[i] = [self.en_tr_w2i[word] for word in en_sentence]
            self.de_tr_inputs[i] = [self.de_tr_w2i[word] for word in de_sentence]

        self.en_ts_words = Counter()
        self.de_ts_words = Counter()
        self.en_ts_inputs = []
        self.de_ts_inputs = []
        for i in range(len(x_val)):
            en_tokens = spacyEn(x_val.iloc[i])
            de_tokens = spacyEn(y_val.iloc[i])

            if len(en_tokens) == 0 or len(de_tokens) == 0:
                continue
            for token in en_tokens:
                self.en_ts_words.update([token.text.lower()])
            self.en_ts_inputs.append([token.text.lower() for token in en_tokens] + ['_EOS'])
            for token in de_tokens:
                self.de_ts_words.update([token.text.lower()])
            self.de_ts_inputs.append([token.text.lower() for token in de_tokens] + ['_EOS'])

        self.en_ts_words = ['_SOS', '_EOS', '_UNK'] + sorted(self.en_ts_words,
                                                        key=self.en_ts_words.get,
                                                        reverse=True)
        self.en_ts_w2i = {o:i for i,o in enumerate(self.en_ts_words)}
        self.en_ts_i2w = {i:o for i,o in enumerate(self.en_ts_words)}

        self.de_ts_words = ['_SOS', '_EOS', '_UNK'] + sorted(self.de_ts_words,
                                                        key=self.de_ts_words.get,
                                                        reverse=True)
        self.de_ts_w2i = {o:i for i,o in enumerate(self.de_ts_words)}
        self.de_ts_i2w = {i:o for i,o in enumerate(self.de_ts_words)}
        for i in range(len(self.en_ts_inputs)):
            en_sentence = self.en_ts_inputs[i]
            de_sentence = self.de_ts_inputs[i]
            self.en_ts_inputs[i] = [self.en_ts_w2i[word] for word in en_sentence]
            self.de_ts_inputs[i] = [self.de_ts_w2i[word] for word in de_sentence]



