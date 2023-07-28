import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import os
import numpy as np

from app.arch import get_model


VOCAB_SIZE = 88584
MAX_LEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

# make each review the same length which is MAX_LEN=250
train_data = sequence.pad_sequences(train_data, MAX_LEN)
test_data = sequence.pad_sequences(test_data, MAX_LEN)

model = get_model(VOCAB_SIZE)
model.trainable = True
# print(model.summary())

# training
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)
model.save('models/movie_review_rnn.h5')

