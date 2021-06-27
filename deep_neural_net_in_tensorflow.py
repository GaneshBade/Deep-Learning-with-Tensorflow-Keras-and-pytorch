#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

X_train = X_train.reshape(60000, 784).astype('float32')
X_valid = X_valid.reshape(10000, 784).astype('float32')

n_classes = 10
to_categorical(y_train, n_classes)
y_train = to_categorical(y_train, n_classes)
y_valid = to_categorical(y_valid, n_classes)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(BatchNormalization)
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer='nadam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_valid, y_valid))
X_valid[0].reshape(1, 784)

valid_0 = X_valid[0].reshape(1, 784)

import numpy as np
np.argmax(model.predict(valid_0))
model.predict_proba(valid_0)
dir(model.predict_proba(valid_0))
model.predict_proba(valid_0).argmax
model.predict_proba(valid_0).argmax()