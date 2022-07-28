# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


#scaling the data(excluding target variable)
sc = StandardScaler()

def init_model():
    #load data
    data_nn = pd.read_csv("C:/Users/santy/df_knn.csv")
    
    data_nn.columns 
    x = data_nn.drop(['success', 'Unnamed: 0'], axis=1)
    print(x)
    y = data_nn['success']
    #scaling the data(excluding target variable)
    X = sc.fit_transform(x)

    
    #splitting the data to train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    #creating deep learning ann model
    classifier = Sequential()
    classifier.add(Dense(units=10, input_dim=11, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the model with best hyperparamters
    classifier.fit(x_train,y_train, batch_size=5 , epochs=125)

    #save model to disk
    pickle.dump(classifier, open('model.pkl', 'wb'))
    
def run_model(values):
    #load model
    model = pickle.load(open('model.pkl', 'rb'))
    
    # Predicting result for Single Observation
    return model.predict(sc.fit_transform(
        values)) > 0.5

# Init Train Model
#init_model()
    
#test model
result = run_model([[8, 10000, 104, 449, 2, 2014, 2015, 2017, 2, 1, 1]])
print("Result:", result)








