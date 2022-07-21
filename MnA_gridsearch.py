


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#loading data
data_ann = pd.read_csv("C:/Users/santy/df_knn.csv")

data_ann.columns 
x = data_ann.drop(['success', 'Unnamed: 0'], axis=1)
print(x)
y = data_ann['success']

#scaling the data(excluding target variable)
sc = StandardScaler()
X = sc.fit_transform(x)

#splitting the data to train and test datasets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Function to generate Deep ANN model 
def make_classification_ann(Optimizer_Trial, Neurons_Trial):
    from keras.models import Sequential
    from keras.layers import Dense
    
    # Creating the classifier ANN model
    classifier = Sequential()
    classifier.add(Dense(units=Neurons_Trial, input_dim=11, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=Neurons_Trial, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=Optimizer_Trial, loss='binary_crossentropy', metrics=['accuracy'])
            
    return classifier


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
 
 
Parameter_Trials={'batch_size':[5,10,20],
                      'epochs':[100,150],
                    'Optimizer_Trial':['adam', 'rmsprop'],
                  'Neurons_Trial': [10,12]
                 }
 
# Creating the classifier ANN
classifierModel=KerasClassifier(make_classification_ann, verbose=0)

# Creating the Grid search space
# See different scoring methods by using sklearn.metrics.SCORERS.keys()
grid_search=GridSearchCV(estimator=classifierModel, param_grid=Parameter_Trials, scoring='f1', cv=5)

# Measuring how much time it took to find the best params
import time
StartTime=time.time()
 
# Running Grid Search for different paramenters
grid_search.fit(x_train,y_train, verbose=1)
 
EndTime=time.time()
print("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')

# printing the best parameters
print('\n Best hyperparamters')
grid_search.best_params_


#Training the model with best hyperparamters
classifier = Sequential()
classifier.add(Dense(units=12, input_dim=11, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train,y_train, batch_size=10 , epochs=150, verbose=1)
# accuracy obtained 70.76






























