import numpy as np
import pandas as pd
from tensorflow import keras
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense

#Loading the data with pandas and selecting our features
weatherData = pd.read_csv('seattle-weather.csv')
X = weatherData.values[:, 1:5] #Features (precipitation, tempMax, tempMin, wind)
Y = weatherData.values[:, 5] #Labels (in this case: weather type)

#Reshaping the labels to a single column, so they all display individually
reshapedY = Y.reshape(-1, 1)

#Represent the labels as binary vectors (numerical variables) with OneHotEncoder instead of nominal variables
encoder = OneHotEncoder(sparse=False)
Y = encoder.fit_transform(reshapedY)

#Split the data in 3 different categories: trainining, validation and testing
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size = 0.20)
X_train, x_val, Y_train, y_val = train_test_split(X_train, Y_train, test_size=0.25)

#Specify a structure/model of a Neural Network
model = Sequential()

#Adding hidden layer(s) with relu activation function
model.add(Dense(10, input_shape=(4,), activation='relu', name='HiddenLayer')) #Any negative numbers are set to zero, positive numbers remain unchanged in the output -- f(x) = max(0, x)
#model.add(Dense(10, activation='relu', name='HiddenLayer2'))
model.add(Dense(5, activation='softmax', name='output')) #Five-class classification task for the output layer

#Compile keras model and optimize with (place_holder)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['Accuracy', 'Precision', 'Recall'])

print('Neural Network Model Summary: ')
print(model.summary())