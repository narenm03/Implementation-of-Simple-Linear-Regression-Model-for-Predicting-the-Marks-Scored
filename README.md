# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Start
2.Import the standard Libraries.
3.Set variables for assigning dataset values.
4.Import linear regression from sklearn.
5.Assign the points for representing in the graph.
6.Predict the regression for marks by using the representation of the graph.
7.Compare the graphs and hence we obtained the linear regression for the given datas.
8.End.

## Program:
```
Program to implement the simple linear regression model
for predicting the marks scored.
Developed by: NARENDHARAN.M
RegisterNumber: 212223230134

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('Desktop\student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 
```
## Output:
## Predicted Values of X and Y:
```
[17.04289179 33.51695377 74.21757747 26.73351648
59.68164043 39.33132858 20.91914167 78.09382734 69.37226512]
[20 27 69 30 62 35 24 86 76]
```
## MSE , MAE and RMSE:
```
MSE =  4.691397441397438
MAE =  4.691397441397438
RMSE=  2.1659633979819324
```
## Training set
![Screenshot 2024-08-30 112651](https://github.com/user-attachments/assets/d6b0d378-b157-4670-acc1-0c3a08c02404)
## Testing Set
![Screenshot 2024-08-30 112806](https://github.com/user-attachments/assets/fc0cd86b-f5dd-4805-b50e-e01d61e20ac8)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
