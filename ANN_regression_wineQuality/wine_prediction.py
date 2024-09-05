import csv 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

# load data
data = pd.read_csv("winequalityN.csv")
print(data.head())
#data = data.replace(0, np.nan, regex=True)
print(data.isnull().sum())
#data.replace("",0) 

data['fixed acidity'] = data['fixed acidity'].fillna(0)
data['volatile acidity'] = data['volatile acidity'].fillna(0)
data['citric acid'] = data['citric acid'].fillna(0)
data['residual sugar'] = data['residual sugar'].fillna(0)
data['chlorides'] = data['chlorides'].fillna(0)
data['free sulfer dioxide'] = data['free sulfur dioxide'].fillna(0)
data['total sulfur dioxide'] = data['total sulfur dioxide'].fillna(0)
data['density'] = data['density'].fillna(0)
data['pH'] = data['pH'].fillna(0)
data['sulphates'] = data['sulphates'].fillna(0)
data['alcohol'] = data['alcohol'].fillna(0)
data['quality'] = data['quality'].fillna(0)
print(data.isnull().sum())

y = data['quality']
print(y.head())
x = data.drop(['quality','type'],axis = 1)
print(x.head())
print(data.shape)


data['quality'] = data['quality'].astype(float)

# visualization data
data.hist(bins = 30, figsize=(5,5))
plt.show()

#plt.figure(bins = 5,figsize=[10,6])
data.plot.bar(x = 'quality', y = 'residual sugar',rot = 10)
plt.ylabel('residual sugar')
plt.xlabel('quality')
plt.show()

##### split data 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=40)
print("The shape of x for training is: ", x_train.shape,"\n","The shape of y for training is: ", y_train.shape)
x_train.to_csv("x_train.csv",index = True)

########### neural network classifier 
# create model 
model = MLPClassifier()
# train the model
model.fit(x_train,y_train)
# test the model -> get the loss and accuracy 
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print("The accuracy of neural network is: ", accuracy)
# save model


########## Random Forest classifier
model_rfc = RandomForestClassifier()
# train the model
model_rfc.fit(x_train,y_train)
# test the model -> get the loss and accuracy 
predictions_rfc = model_rfc.predict(x_test)
accuracy_rfc = accuracy_score(y_test, predictions_rfc)
print("The accuracy of random forest model is: ", accuracy_rfc)


"""with open('winequality.csv', 'r') as f:
    reader = csv.reader(f,delimiter = ",")
    for row in reader:
        print(row[1])"""
