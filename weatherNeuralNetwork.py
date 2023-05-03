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
x = weatherData.values[:, 1:5] #Features (precipitation, tempMax, tempMin, wind)

#Convert the data from a list to a float32 type so that TensorFlow can read it
x_array = np.asarray(x).astype('float32')
y = weatherData.values[:, 5] #Labels (in this case: weather type)

#Reshaping the labels to a single column, so they all display individually
reshapedY = y.reshape(-1, 1)

#Represent the labels as binary vectors (numerical variables) with OneHotEncoder instead of nominal variables
encoder = OneHotEncoder(sparse=False)
Y = encoder.fit_transform(reshapedY)

#Split the data in 3 different categories: trainining, validation and testing
#We use sklearn.model_selection.train_test_split twice
#First to split to 'train and test', then split train again into 'validation and train'
train_x, test_x, train_y, test_y = train_test_split(x_array, Y, test_size = 0.20)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25) #0.25 * 0.80 = 0.20

#Specify a structure/model of a Neural Network, feedforward
model = Sequential()

#Adding hidden layer(s) with relu activation function
model.add(Dense(10, input_shape=(4,), activation='relu', name='HiddenLayer')) #Any negative numbers are set to zero, positive numbers remain unchanged in the output -- f(x) = max(0, x)
model.add(Dense(10, activation='relu', name='HiddenLayer2'))
model.add(Dense(5, activation='softmax', name='output')) #Five-class classification task for the output layer

#Compile keras model and optimize with (place_holder)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['Accuracy', 'Precision', 'Recall'])

print('Neural Network Model Summary: ')
print(model.summary())

#Train the model on the fitted data
#Batch_size = number of training examples to feedforward and propagate before the model weights are updated
#Epochs = How many times we iterate over all training examples
model.fit(train_x, train_y, verbose=1, batch_size=32, epochs=200)

#Test on unseen data
results = model.evaluate(test_x, test_y)

#Print the final results for the model's: set loss, accuracy, precision and recall
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))
print('Final test set precision: {:4f}'.format(results[2]))
print('Final test set recall: {:4f}'.format(results[3]))