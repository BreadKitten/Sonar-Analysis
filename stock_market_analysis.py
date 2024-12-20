import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

stock_market_data = pd.read_csv("./data/GOOG.csv")


def printSpace(x) :
    print()
    print("------------------------------------------------------------------------------------------------------------------")
    print()
    print(x)

# first 5 entries
printSpace(stock_market_data.head())

# summary
printSpace(stock_market_data.describe())

# remove header
smd_no_header = stock_market_data.tail(-1)
printSpace(smd_no_header)

# make new 
THRESHOLD = 0.01
Y = []
for i, row in smd_no_header.iterrows() :
    start = float(row.iloc[3])
    end = float(row.iloc[5])
    diff = end - start
    Y.append("N" if np.abs(diff) < start * THRESHOLD else "U" if diff > 0 else "D")

Y = np.array(Y)
printSpace(pd.Series(Y).value_counts(sort = True, ascending = True))

printSpace(Y)
print(Y.shape)

X = smd_no_header.drop(columns = ["Price", "Adj Close", "Close", "High", "Low"], axis = 1)
printSpace(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 1)

# verify dimensions
printSpace(X.shape)
print(X_train.shape, X_test.shape)

model = LogisticRegression()
model.fit(X_train, Y_train)

print("Getting accuracy of analysis")
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Achieved accuracy of ", training_data_accuracy)

incorrect = []

for i, row in X_test.iterrows():
    input_data = pd.DataFrame(row.values.reshape(1, -1), columns = X_train.columns)
    prediction = model.predict(input_data)

    if (Y[i] != prediction[0]) :
        incorrect.append([i, Y[i], prediction[0]])

incorrect = pd.DataFrame(incorrect)

incorrect.sort_values(by = incorrect.columns[0], ascending = True, inplace= True)
incorrect.columns = ["Index", "Expected", "Actual"]

printSpace(incorrect.to_string(index = False))
printSpace("")