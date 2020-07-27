import os
import numpy as np
import pandas as pd
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
data =pd.read_csv('C:\\Users\\hp\\Desktop\\weatherHistory.csv')
data.head()
data.info()
data['Formatted Date'],FormattedDate_names = pd.factorize(data['Formatted Date'])
print(FormattedDate_names)
print(data['Formatted Date'].unique())
data['Summary'],Summary_names = pd.factorize(data['Summary'])
data['Precip Type'],PrecipType_names = pd.factorize(data['Precip Type'])
data['Temperature (C)'],Temperature_names = pd.factorize(data['Temperature (C)'])
data['Apparent Temperature (C)'],ApparentTemperaturenames = pd.factorize(data['Apparent Temperature (C)'])
data['Humidity'],Humidity_names = pd.factorize(data['Humidity'])
data['Wind Speed (km/h)'],WindSpeed_names = pd.factorize(data['Wind Speed (km/h)'])
data['Wind Bearing (degrees)'],WindBearing_names = pd.factorize(data['Wind Bearing (degrees)'])
data['Visibility (km)'],Visibility_names = pd.factorize(data['Visibility (km)'])
data['Loud Cover'],LoudCover_names = pd.factorize(data['Loud Cover'])
data['Pressure (millibars)'],Pressure_names = pd.factorize(data['Pressure (millibars)'])
data['Daily Summary'],DailySummary_names = pd.factorize(data['Daily Summary'])
data.head()
data.info()
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
 
dtree = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
import graphviz
feature_names = X.columns

dot_data = tree.export_graphviz(dtree, out_file=None, filled=True, rounded=True,
                                feature_names=feature_names,  
                                class_names=DailySummary_names)
graph = graphviz.Source(dot_data)  
graph
 
