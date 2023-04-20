import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt




#Load the CSV file to read the weather data
weatherData = pd.read_csv('seattle-weather.csv')

#Categorize weather conditions with numbers
"""
 Drizzle = 0
    Rain = 1
     Sun = 2
    Snow = 3
     Fog = 4
"""

weatherData = weatherData.replace({'drizzle': '0', 'rain': '1', 'sun': '2', 'snow': '3', 'fog': '4'})


knn = KNeighborsClassifier()
print(weatherData.groupby('weather').size())
X = weatherData.values[:, 1:5]  # variables
Y = weatherData.values[:, 5] #Weather

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=7, shuffle=True)

knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print("KNN Results", "\n")
print("Accuracy:", accuracy_score(Y_test, predictions))
print("Confusion matrix:", "\n", confusion_matrix(Y_test, predictions))
print("Classification report:", "\n", classification_report(Y_test, predictions), "\n")
print(weatherData.groupby('weather').size())

print("Trying the Kfold cross validation")
kfold = model_selection.KFold(n_splits=10, random_state=None, shuffle=False)
cv_results = model_selection.cross_val_score(knn, X_train, Y_train, cv=kfold, scoring='accuracy')
print("Generalizability Results", "\n")
print("KNN average accuracy: %f" % (cv_results.mean()))

#Check and drop if the weather data has duplicate entries
#weatherData.drop_duplicates()
#print(weatherData)

if __name__ == '__main__':
    print("hI")