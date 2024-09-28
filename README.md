# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1:START


STEP 2:Import the standard Libraries.


STEP 3:Set variables for assigning dataset values.


STEP 4:Import linear regression from sklearn.


STEP 5:Assign the points for representing in the graph.


STEP 6:Predict the regression for marks by using the representation of the graph.


STEP 7:Compare the graphs and hence we obtained the linear regression for the given datas.


STEP 8:END
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NARENDHARAN.M
RegisterNumber:  212223230134
*/

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("/content/student_scores.csv")

print(df.tail())
print(df.head())
df.info()

x = df.iloc[:, :-1].values  # Hours
y = df.iloc[:,:-1].values   # Scores

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

print("X_Training:", x_train)
print("X_Test:", x_test)
print("Y_Training:", y_train)
print("Y_Test:", y_test)

reg = LinearRegression()
reg.fit(x_train, y_train)

Y_pred = reg.predict(x_test)

print("Predicted Scores:", Y_pred)
print("Actual Scores:", y_test)

a = Y_pred - y_test
print("Difference (Predicted - Actual):", a)

plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, reg.predict(x_train), color="red")
plt.title('Training set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, reg.predict(x_test), color="green")
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(y_test, Y_pred)
mse = mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```

## Output:
![Screenshot 2024-08-29 194702](https://github.com/user-attachments/assets/1b416b49-683a-41df-85cf-af0ad4fc6d66)



![Screenshot 2024-08-29 194714](https://github.com/user-attachments/assets/76dd3e8c-4413-4fa2-bf0b-118b25b73360)



![Screenshot 2024-08-29 194722](https://github.com/user-attachments/assets/9a43d47c-975f-4470-853a-1920f77e9515)



![Screenshot 2024-08-29 194731](https://github.com/user-attachments/assets/d8b6cfc7-36bd-4814-8d69-6d44a7652fcc)



![Screenshot 2024-08-29 194739](https://github.com/user-attachments/assets/4982ca03-01e2-47cd-b853-488028700564)


![Screenshot 2024-08-29 194755](https://github.com/user-attachments/assets/496ed807-1353-4910-ab79-aca7acb96ef3)


![Screenshot 2024-08-29 194802](https://github.com/user-attachments/assets/6b00ec72-fab1-4d2d-9e28-8110225f1588)


![Screenshot 2024-08-29 194811](https://github.com/user-attachments/assets/95befbee-90b2-4e1e-a24d-baa0158906df)


![Screenshot 2024-08-29 194817](https://github.com/user-attachments/assets/4357dad0-0ee0-4f67-ac98-4e48406c39a3)


![Screenshot 2024-08-29 194824](https://github.com/user-attachments/assets/56d7f72e-306f-45c2-add8-2d66aacd63cb)


![Screenshot 2024-08-29 194832](https://github.com/user-attachments/assets/f2a01ead-ee45-4b91-91dc-55660f6c16aa)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
