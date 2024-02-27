# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by:GOKUL J 
RegisterNumber:212222230038
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df = pd.read_csv('/content/ml.csv')
df.head(10)

plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')

x=df.iloc[:,0:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)

x_train
y_train

lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='orange')
lr.coef_
lr.intercept_



```

## Output:
## df.head:
![Screenshot 2024-02-23 213354](https://github.com/Gokul0117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165938/b850e3d6-3111-4303-b479-fc2f5e8e9680)

## Graph of plotted data:
![Screenshot 2024-02-23 212850](https://github.com/Gokul0117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165938/a2efb952-3a9c-4dae-9425-9c038e5189c1)

## Performing Linear Regression:
![Screenshot 2024-02-23 213127](https://github.com/Gokul0117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165938/e5ac47f7-ee0b-4ffa-b141-992f2506ea8d)

## Trained data:
![Screenshot 2024-02-23 213151](https://github.com/Gokul0117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165938/1efa4b20-abc7-4fc1-86ab-b5b94036d129)

## Predicting the line of Regression:
![Screenshot 2024-02-23 213259](https://github.com/Gokul0117/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165938/a05c9580-798d-49a6-b5f9-4e986838d00d)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
