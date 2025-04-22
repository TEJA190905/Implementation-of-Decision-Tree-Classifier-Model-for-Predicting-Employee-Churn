# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: M THEJESWARAN
RegisterNumber: 212223240168
*/

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv("/content/Employee.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())

le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

x = data[["satisfaction_level", "last_evaluation", "number_project", 
          "average_montly_hours", "time_spend_company"]]

y = data["left"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Sample Prediction:", dt.predict([[0.5, 0.8, 9, 260, 6]]))

plt.figure(figsize=(16,10))
plot_tree(dt, feature_names=x.columns, class_names=['Stay', 'Left'], filled=True)
plt.show()

```

## Output:

![Screenshot (97)](https://github.com/user-attachments/assets/b96a2c1b-c7b9-4ef9-9b4d-a2e33ead1e2a)

![Screenshot (99)](https://github.com/user-attachments/assets/68e083b4-afb9-40e3-8a7d-ba1ba2701b94)

![image](https://github.com/user-attachments/assets/4c40060d-3815-4b58-93fe-0a308bbb5c3c)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
