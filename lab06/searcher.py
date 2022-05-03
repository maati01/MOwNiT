import json

from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np


class Searcher:
    def __init__(self):
        self.train = load_dataset("wikipedia", "20220301.simple")['train']
        self.dictionary = {}
        self.words = []
        self.size = len(self.train['title'])
        self.size = 1000
        self.words_frequency_in_article = [{} for _ in range(self.size)]
        self.stop_words = set(stopwords.words('english'))

        self.preprocessing()

        self.words_frequency = self.find_frequency()

        self.matrix = self.create_matrix()
        self.idf()
        self.find_articles()

        # nltk.download('stopwords')
        # nltk.download('wordnet')
        # nltk.download('omw-1.4')

    def prepare_list(self, data):
        text = data.lower()
        text = text.replace("_", "")
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub('  +', ' ', text)
        return text.split()

    def preprocessing(self):
        lemmatizer = WordNetLemmatizer()
        idx = 0
        for data in self.train['text']:
            # text = data.lower()
            # text = text.replace("_", "")
            # text = re.sub(r'[^\w\s]', '', text)
            # text = re.sub('  +', ' ', text)
            words_list = self.prepare_list(data)

            for word in words_list:
                word = lemmatizer.lemmatize(word, pos="v")
                if word in self.stop_words: continue

                if self.dictionary.__contains__(word):
                    self.dictionary[word] += 1
                else:
                    self.dictionary[word] = 1
                    self.words.append(word)  # to jest zbedne

                if self.words_frequency_in_article[idx].__contains__(word):
                    self.words_frequency_in_article[idx][word] += 1
                else:
                    self.words_frequency_in_article[idx][word] = 1
            idx += 1
            if idx == 1000:
                break

    def create_matrix(self):
        matrix = np.zeros((len(self.words), self.size))
        indexes = {self.words[i]: i for i in range(len(self.words))}

        idx = 0
        for column in self.words_frequency_in_article:
            sum_ = sum(column.values())
            for word in column.keys():
                matrix[indexes[word], idx] = self.words_frequency_in_article[idx][word] / sum_

            idx += 1

        return matrix

    def idf(self):
        indexes = {i: self.words[i] for i in range(len(self.words))}
        for i in range(len(self.words)):
            self.matrix[i, :] *= np.log(self.size / self.words_frequency[indexes[i]])

    def find_frequency(self):
        freq = {self.words[i]: 0 for i in range(len(self.words))}

        for word in self.words:
            for set_freq in self.words_frequency_in_article:
                if set_freq.__contains__(word):
                    freq[word] += 1

        return freq

    def find_articles(self):
        # np.savetxt('matrix.txt', self.matrix, fmt='%d')
        indexes = {self.words[i]: i for i in range(len(self.words))}
        similarity = []
        word = input()

        # text = word.lower()
        # text = text.replace("_", "")
        # text = re.sub(r'[^\w\s]', '', text)
        # text = re.sub('  +', ' ', text)
        # words_list = text.split()
        words_list = self.prepare_list(word)

        vector = np.zeros(len(self.words))

        for word in words_list:
            vector[indexes[word]] = 1 / len(words_list)

        for i in range(self.size):
            cos = np.matmul(vector.T, self.matrix[:, i]) / (np.linalg.norm(vector) * np.linalg.norm(self.matrix[:, i]))
            similarity.append(cos)

        ind = np.array(similarity)
        ind = ind.argsort()[-9:][::-1]

        for idx in ind:
            print(self.train['title'][idx])
