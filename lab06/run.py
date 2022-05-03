import numpy as np
import re
from datasets import load_dataset


def prepare_list(data):
    text = data.lower()
    text = text.replace("_", "")
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('  +', ' ', text)
    return text.split()


def find_articles(matrix, words, train):
    np.save('data.npy', matrix)
    words_number = len(words)
    word_to_index = {words[i]: i for i in range(words_number)}
    similarity = []
    word = input()

    words_list = prepare_list(word)

    vector = np.zeros(words_number)

    for word in words_list:
        vector[word_to_index[word]] = 1 / len(words_list)

    for i in range(len(matrix[0])):
        cos = np.matmul(vector.T, matrix[:, i]) / (np.linalg.norm(vector) * np.linalg.norm(matrix[:, i]))
        similarity.append(cos)

    ind = np.array(similarity)
    ind = ind.argsort()[-9:][::-1]

    i = 0
    for idx in ind:
        print(f"{i}: ", train['title'][idx])
        i += 1


if __name__ == "__main__":
    matrix = np.load('matrix.npy')
    words = np.load('words.npy')
    find_articles(matrix, words, load_dataset("wikipedia", "20220301.simple")['train'])
