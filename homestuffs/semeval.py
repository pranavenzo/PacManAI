import csv
import time

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import re
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, Flatten, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras import backend as K

class_mapping = {"AGAINST": 0, "NONE": 1, "FAVOR": 2}


def sentence_cleaner(raw):
    clean = re.sub("[^a-zA-Z]", " ", raw)
    words = clean.split()
    return words


def read_data(file_names):
    X, y = [], []
    text = ''
    for file_name in file_names:
        reader = csv.reader(open(file_name))
        next(reader, None)  # skip the headers
        for row in reader:
            X.append(row[2])  # ignoring the target for now lmao
            y.append(class_mapping[row[-1]])
            text += row[2]
    return X, y, text


X, y, data_text = read_data(["semeval2016-task6-trialdata.csv", "semeval2016-task6-trainingdata.csv"])
data_list = sentence_cleaner(data_text)
num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data_list)
x_train_tokens = tokenizer.texts_to_sequences(X)
print(tokenizer.word_index)
print(np.array(x_train_tokens[1]))
num_tokens = [len(tokens) for tokens in x_train_tokens]
num_tokens = np.array(num_tokens)
print(np.mean(num_tokens))
print(np.max(num_tokens))
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
print(max_tokens)
print(np.sum(num_tokens < max_tokens) / len(num_tokens))
pad = 'pre'
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)
print(x_train_pad.shape)
idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))


def tokens_to_string(tokens):
    # Map from tokens back to words.
    words = [inverse_map[token] for token in tokens if token != 0]

    # Concatenate all words.
    text = " ".join(words)

    return text


print(X[1])
print(tokens_to_string(x_train_tokens[1]))
embedding_size = 4
# model = Sequential()
#
# model.add(Embedding(input_dim=num_words,
#                     output_dim=embedding_size,
#                     input_length=max_tokens,
#                     name='layer_embedding'))
# model.add(GRU(units=16, return_sequences=True))
# model.add(GRU(units=8, return_sequences=True))
# model.add(GRU(units=16))
#
# model.add(Dense(1, activation='elu'))
# optimizer = Adam(lr=1e-3)
# model.compile(loss='mean_absolute_error',
#               optimizer=optimizer,
#               metrics=['accuracy'])
# print(model.summary())
# start = time.time()
# model.fit(x_train_pad, y,
#           validation_split=0.1, epochs=10, batch_size=64)
# print('Time taken : %f' % (time.time() - start))
#
y_train_onehot = np_utils.to_categorical(y)
# plt.hist(y, label='Y_train')
# plt.show()
vals = {}
for y_i in y:
    vals[y_i] = vals.get(y_i, 0) + 1.0
for v in vals.keys():
    vals[v] = vals[v] / len(y)
print(vals)
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores_avg = []
cvscores_for = []
cvscores_agnst = []
tri = []
for train, test in kfold.split(x_train_pad, y):
    x_train = x_train_pad[train]
    y_train = y_train_onehot[train]
    x_test = x_train_pad[test]
    y_test = y_train_onehot[test]
    model = Sequential()
    model.add(Embedding(input_dim=num_words,
                        output_dim=embedding_size,
                        input_length=max_tokens,
                        name='layer_embedding'))
    # model.add(Flatten())
    # model.add(Dense(25, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(GRU(units=16))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['acc'])

    model.fit(x_train, y_train,
              epochs=20, batch_size=128, verbose=0)
    y_pred = model.predict(x_test).argmax(axis=-1)
    cm = classification_report(y_test.argmax(axis=-1), y_pred)
    dl = f1_score(y_true=y_test.argmax(axis=-1), y_pred=y_pred, average=None)
    print(cm, dl)
    score = model.evaluate(x_test, y_test, batch_size=128)
    score2 = model.evaluate(x_train, y_train, batch_size=128)
    print(score, score2)
    cvscores_avg.append((dl[0] + dl[1]) / 2.0)
    cvscores_for.append(dl[0])
    cvscores_agnst.append(dl[1])
    nyp=np.array([0]*len(y_test.argmax(axis=-1)))
    trial = f1_score(y_true=y_test.argmax(axis=-1), y_pred=nyp, average=None)[0]
    tri.append(trial)
print('Avg', np.mean(cvscores_avg))
print('For', np.mean(cvscores_for))
print('Agnst', np.mean(cvscores_agnst))
print('BL',  np.mean(tri))
# print(model.summary())
