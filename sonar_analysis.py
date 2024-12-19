import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load dataset to dataframe
sonar_data = pd.read_csv('./data/sonar_data.csv', header = None)

print("Mines dataset:")
print(sonar_data.head())
print("---------------------------")

print("Dimensions of the data:")
print(sonar_data.shape)
print("---------------------------")

print("Check information (like summarize):")
print(sonar_data.describe())
print("---------------------------")

print("Number of mines, and number of rocks:")
print(sonar_data[60].value_counts(sort = True, ascending = True))

print("Mean grouped by mines/rocks")
print(sonar_data.groupby(60).mean())

print("Cleaning data by index to mine/rock")
X = sonar_data.drop(columns = 60, axis = 1)
Y = sonar_data[60]

print(X)
print(Y)

print("Seperating data out into training and test")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 1)
print(X.shape, X_train.shape, X_test.shape)

print(X_train)
print(Y_train)

print("Performing logistic analysis")
model = LogisticRegression()
model.fit(X_train, Y_train)

print("Getting accuracy of analysis")
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Achieved accuracy of ", training_data_accuracy)

for i, row in X_test.iterrows():
    input_data = row.values.reshape(1, -1)

    prediction = model.predict(input_data)

    if (Y[i] != prediction[0]) :
        print("Incorrect print out for index: ", i)
        print("Predicted ", prediction[0], ", but was ", Y[i])