import pandas as pd
import numpy as np
data = pd.read_csv("credit_card.csv")
print(data.head())
print("---------------------------------------------------------------------------------------------------------------")
print(data.isnull().sum())
print("---------------------------------------------------------------------------------------------------------------")
print(data.type.value_counts())
print("---------------------------------------------------------------------------------------------------------------")
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())
print("---------------------------------------------------------------------------------------------------------------")
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])
print(x)
print(y)
print("---------------------------------------------------------------------------------------------------------------")
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
print("---------------------------------------------------------------------------------------------------------------")
features = np.array([[3, 9000.6, 9000, 1000000.0]])
print(model.predict(features))
