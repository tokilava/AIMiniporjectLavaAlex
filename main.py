import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

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
print(weatherData)

#Check and drop if the weather data has duplicate entries
#weatherData.drop_duplicates()
#print(weatherData)

#Make a boxplot of the different features
#weatherData.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False)
#plt.show()

#Group the data together into weather categories
print(weatherData.groupby('weather').size())

knn = KNeighborsClassifier()
X = weatherData.values[:, 1:5]  # features
Y = weatherData.values[:, 5]  # labels

#Set aside 20% of train and test data for evaluation
#Do the same for validation data
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=32, shuffle=True)
X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_train, Y_train, test_size=0.25, random_state=32, shuffle=True)

#Create an instance of the KNN model then fit this to our training data
#After the model is trained, make predictions on the dataset
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)

#Evaluate the model by using accuracy
#Check the predictions against the actual values in the test set
#Count of many the model predicted correctly
print("KNN Results", "\n")
print("Accuracy:", accuracy_score(Y_test, predictions))
print("Confusion matrix:", "\n", confusion_matrix(Y_test, predictions))
print("Classification report:", "\n", classification_report(Y_test, predictions), "\n")

#Create a display window for the confusion matrix with labels
cm = confusion_matrix(Y_test, predictions, labels=knn.classes_)
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=knn.classes_)
disp.plot()
plt.show()

#K-fold cross-validation
#Print results to the terminal
knn = KNeighborsClassifier()
kfold = model_selection.KFold(n_splits=10, random_state=None, shuffle=False)
cv_results = model_selection.cross_val_score(knn, X_train, Y_train, cv=kfold, scoring='accuracy')
print("Generalizability Results", "\n")
print("KNN average accuracy: %f" % (cv_results.mean()))

#Select a range of values for "k" and store them in an empty array
k_values = [i for i in range (1, 60)]
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_results = model_selection.cross_val_score(knn, X_train, Y_train, cv=kfold)
    scores.append(np.mean(cv_results))

#Visualize the accuracy for all k-values
sns.lineplot(x = k_values, y = scores, markers='o')
plt.xlabel("K-values")
plt.ylabel("Accuracy for cross-validation")
plt.show()