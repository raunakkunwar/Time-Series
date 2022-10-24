


import numpy as np
import keras
import pandas as pd
import math


def data_normalization(raw_data, train_start, train_end):
    (number, dimensions) = raw_data.shape
    normalized_data = np.zeros((number, dimensions))

    for d in range(0, dimensions):
        feature_values = raw_data[0:train_end, d]
        m = np.mean(feature_values)
        s = np.std(feature_values)
        normalized_data[:, d] = (raw_data[:, d]-m)/s
    return normalized_data


def make_inputs_and_targets(data, months, size, sampling):
    (ts_length, dimensions) = data.shape
    input_length = math.ceil((24*6*14)/sampling)
    inputs = np.zeros((size, input_length, dimensions))
    targets = np.zeros((size))

    for i in range(0, size):
        max_start = ts_length - input_length*sampling
        start = np.random.randint(0,max_start)
        end = start + input_length*sampling
        inputs[i] = data[start:end:sampling, :]
        targets[i] = months[start+(24*6*7)]
    return inputs, targets


def build_and_train_dense(train_inputs, train_targets, val_inputs, val_targets, filename):
    input_shape = train_inputs[0].shape

    model = keras.Sequential([
        keras.Input(input_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='tanh'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(12, activation='softmax')
        ])
    print(model.summary())
    model.compile(loss='SparseCategoricalCrossentropy', optimizer='adam', metrics=['accuracy'])
    callbacks = [keras.callbacks.ModelCheckpoint(filename, save_best_only=True)]
    history = model.fit(train_inputs, train_targets, epochs=10, validation_data=(val_inputs, val_targets), callbacks = callbacks)
    model.save(filename)
    return history


def test_model(filename, test_inputs, test_targets):
    model = keras.models.load_model(filename)
    return model.evaluate(test_inputs, test_targets)[1]


def confusion_matrix(filename, test_inputs, test_targets):
    model = keras.models.load_model(filename)
    y_pred = model.predict(test_inputs)
    x = test_targets.size
    confMtrx = np.zeros((12,12), dtype = int)
    for i in range(0, x):
    	max_prediction = np.argmax(y_pred[i,:])
    	targets = int(test_targets[i])
    	confMtrx[targets, max_prediction] += 1
    	
    return confMtrx
    #   return metrics.confusion_matrix(test_targets,y_pred)


def build_and_train_rnn(train_inputs, train_targets, val_inputs, val_targets, filename):
    input_shape = train_inputs[0].shape
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dense(12, activation='softmax')
    ])

    model.compile(loss='SparseCategoricalCrossentropy', optimizer='adam', metrics=['accuracy'])
    callbacks = [keras.callbacks.ModelCheckpoint(filename, save_best_only=True)]
    history = model.fit(train_inputs, train_targets, epochs=10, validation_data=(val_inputs, val_targets), callbacks = callbacks)
    model.save(filename)
    return history
