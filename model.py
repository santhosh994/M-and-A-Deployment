#!/usr/bin/env python
# coding: utf-8

# ## Artificial Neural Network

import tensorflow.keras as keras
from keras.models import Sequential

from keras.layers import Dense
# import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

classifier = Sequential()
sc = StandardScaler()


def init_model():
    df_ann = pd.read_csv("df_knn.csv")

    df_ann.head()

    df_ann.columns

    x = df_ann.drop(['success', 'Unnamed: 0'], axis=1)
    print(x)
    x.columns
    y = df_ann['success']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_train.columns

    # Feature Scaling
    X_train_scaled = sc.fit_transform(x_train)
    X_test_scaled = sc.transform(x_test)

    classifier.add(Dense(
        units=6,
        kernel_initializer="uniform",
        activation="relu",
        input_dim=11
    ))
    # adding second hidden layer
    classifier.add(Dense(
        units=6,
        kernel_initializer="uniform",
        activation="relu",
    ))

    # adding output layer
    classifier.add(Dense(
        units=1,
        kernel_initializer="uniform",
        activation="sigmoid",
    ))

    # compile ANN
    classifier.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=['accuracy']
    )

    classifier.fit(
        X_train_scaled,
        y_train,
        batch_size=5,
        epochs=100
    )

    # saving model to disk
    pickle.dump(classifier, open('model.pkl', 'wb'))


def run_model(values):
    # loading model
    model = pickle.load(open('model.pkl', 'rb'))

    # Predicting result for Single Observation
    return model.predict(sc.fit_transform(
        values)) > 0.5


# Init Train Model
# init_model()

# Test Model
result = run_model([[8, 10000, 104, 449, 2, 2014, 2015, 2017, 2, 1, 1]])
print("Result: ", result)
