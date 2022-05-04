from flask import Flask, render_template, request, url_for, flash, redirect
import numpy as np
import re
from datasets import load_dataset

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ee8ed6c253804765bb8b2d9a7f49609c379257022eb4fd3d'

messages = []

matrix = np.load('matrix.npy')
words = np.load('words.npy')
data_set = load_dataset("wikipedia", "20220301.simple")['train']
word_to_index = {words[i]: i for i in range(len(words))}


def prepare_list(data):
    text = data.lower()
    text = text.replace("_", "")
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('  +', ' ', text)
    return text.split()


def find_articles(matrix, words, message):
    words_number = len(words)

    similarity = []

    words_list = prepare_list(message)

    vector = np.zeros(words_number)

    for word in words_list:
        if word_to_index.__contains__(word):
            vector[word_to_index[word]] = 1 / len(words_list)

    for i in range(len(matrix[0])):
        cos = np.matmul(vector.T, matrix[:, i]) / (np.linalg.norm(vector) * np.linalg.norm(matrix[:, i]))
        similarity.append(cos)

    ind = np.array(similarity)
    ind = ind.argsort()[-9:][::-1]

    return [(data_set['title'][idx], data_set['url'][idx]) for idx in ind]


@app.route('/', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        message = request.form['message']

        if not message:
            flash('Write something!')

        else:
            messages.append({'message': message})
            return redirect(url_for('list'))

    return render_template('create.html')


@app.route('/list/', methods=('GET', 'POST'))
def list():
    if request.method == "POST":
        messages.pop()
        return redirect(url_for('create'))

    result = find_articles(matrix, words, messages[0]['message'])
    return render_template('list.html', list=result)


if __name__ == '__main__':
    app.run()
