import keras
from keras.layers import Input, Dense, Embedding, GRU, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
import re
import csv

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

def make_model():
    embedding_size = 16
    I1 = Input(shape=(max_tokens,), name='I1')
    embedder = Embedding(input_dim=num_words,
                         output_dim=embedding_size,
                         name='EMB')
    A1 = embedder(I1)
    A2 = GRU(units=8, name='A2')(A1)
    A3 = Dropout(rate=0.5)(A2)
    I2 = Input(shape=(max_tokens2,), name='I2')
    B1 = embedder(I2)
    # B2 = GRU(units=8, name='B2')(B1)
    B2 = Flatten(name='B2')(B1)
    B3 = Dense(30, activation='elu', name='B3')(B2)
    B4 = Dropout(rate=0.5)(B3)
    C1 = keras.layers.concatenate([B4, A3])
    C2 = Dense(30, activation='elu', name='C2')(C1)
    C3 = Dropout(rate=0.5)(C2)
    C4 = Dense(3, activation='softmax', name='C3')(C3)
    merged = Model(inputs=[I1, I2], outputs=[C4])
    print(merged.summary())
    # plot_model(merged, to_file='demo.png', show_shapes=True)
    return merged

class_mapping = {"AGAINST": 0, "NONE": 1, "FAVOR": 2}

def sentence_cleaner(raw):
    clean = re.sub("[^a-zA-Z]", " ", raw)
    words = clean.split()
    return words

def read_data(file_names):
    X_0, X_1, y = [], [], []
    text = ''
    for file_name in file_names:
        reader = csv.reader(open(file_name))
        next(reader, None)  # skip the headers
        for row in reader:
            X_0.append(row[2])
            X_1.append(row[1])
            y.append(class_mapping[row[-1]])
            text += row[2]
    return X_0, X_1, y, text

X_0, X_1, y, data_text = read_data(["semeval2016-task6-trialdata.csv", "semeval2016-task6-trainingdata.csv"])
data_list = sentence_cleaner(data_text)
num_words = 100
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data_list)
x_train_tokens_0 = tokenizer.texts_to_sequences(X_0)
x_train_tokens_1 = tokenizer.texts_to_sequences(X_1)

def max_calc(x_train_tokens):
    num_tokens = [len(tokens) for tokens in x_train_tokens]
    num_tokens = np.array(num_tokens)
    print(np.mean(num_tokens))
    print(np.max(num_tokens))
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens=np.max(num_tokens)
    max_tokens = int(max_tokens)
    print(max_tokens)
    print(np.sum(num_tokens < max_tokens) / len(num_tokens))
    return max_tokens

max_tokens = max_calc(x_train_tokens_0)
pad = 'pre'
x_train_pad_0 = pad_sequences(x_train_tokens_0, maxlen=max_tokens,
                              padding=pad, truncating=pad)
max_tokens2 = max_calc(x_train_tokens_1)
pad = 'pre'
x_train_pad_1 = pad_sequences(x_train_tokens_0, maxlen=max_tokens2,
                              padding=pad, truncating=pad)
print(x_train_pad_0.shape)
print(x_train_pad_1.shape)
y_onehot = np_utils.to_categorical(y)

def print_input_data_stats():
    vals = {}
    for y_i in y:
        vals[y_i] = vals.get(y_i, 0) + 1.0
    for v in vals.keys():
        vals[v] = vals[v] / len(y)
    print(vals)

print_input_data_stats()

kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores_avg = []
cvscores_against = []
cvscores_favor = []
tri = []
for train, test in kfold.split(x_train_pad_0, y):
    x_train_0 = x_train_pad_0[train]
    x_train_1 = x_train_pad_1[train]
    y_train = y_onehot[train]
    x_test_0 = x_train_pad_0[test]
    x_test_1 = x_train_pad_1[test]
    y_test = y_onehot[test]
    model = make_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['acc'])
    print(model.input)
    print(x_train_0.shape)
    print(x_train_1.shape)
    print(x_train_1.shape)
    print(y_train.shape)
    # for x_train_0_i, x_train_1_i, y_train_i in zip(x_train_0, x_train_1, y_train):
    model.fit([x_train_0, x_train_1], y_train, epochs=20, batch_size=64, verbose=0)
    y_pred = model.predict([x_test_0, x_test_1]).argmax(axis=-1)
    scores = f1_score(y_true=y_test.argmax(axis=-1), y_pred=y_pred, average=None)
    cvscores_avg.append((scores[0] + scores[1]) / 2.0)
    cvscores_against.append(scores[0])
    cvscores_favor.append(scores[1])
    score = model.evaluate([x_test_0, x_test_1], y_test, batch_size=64)
    score2 = model.evaluate([x_train_0, x_train_1], y_train, batch_size=64)
    print(score, score2)
print('For', np.mean(cvscores_favor))
print('Against', np.mean(cvscores_against))
print('Avg', np.mean(cvscores_avg))
