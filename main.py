import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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
weatherData.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False)
plt.show()

#Group the data together into weather categories
print(weatherData.groupby('weather').size())

knn = KNeighborsClassifier()
X = weatherData.values[:, 1:5]  # features
Y = weatherData.values[:, 5]  # labels

#Set aside 20% of train and test data for evaluation
#Do the same for validation data
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=32, shuffle=True)
X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_train, Y_train, test_size=0.25, random_state=32, shuffle=True)

knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)

print("KNN Results", "\n")
print("Accuracy:", accuracy_score(Y_test, predictions))
print("Confusion matrix:", "\n", confusion_matrix(Y_test, predictions))
print("Classification report:", "\n", classification_report(Y_test, predictions), "\n")

#K-fold cross-validation
knn = KNeighborsClassifier()
kfold = model_selection.KFold(n_splits=10, random_state=None, shuffle=False)
cv_results = model_selection.cross_val_score(knn, X_train, Y_train, cv=kfold, scoring='accuracy')
print("Generalizability Results", "\n")
print("KNN average accuracy: %f" % (cv_results.mean()))

#SVM validation
models = [('KNN', KNeighborsClassifier()), ('SVM', SVC())]
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    print("Generalizability Results", "\n")
    print("%s average accuracy: %f" % (name, cv_results.mean()))