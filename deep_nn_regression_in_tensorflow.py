#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import os

(X_train, y_train), (X_valid, y_valid) = boston_housing.load_data()

model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, batch_size=8, epochs=32, verbose=1, validation_data=(X_valid, y_valid))

output_dir = 'model_output/'
run_name = 'regression_baseline'
output_path = output_dir + run_name
output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)

modelcheckpoint = ModelCheckpoint(output_path + '/weights.{epoch:02d}.hdf5', save_weights_only=True)

model.load_weights(output_path + '/weights.20.hdf5')

model.predict(np.reshape(X_valid[42], [1, 13]))