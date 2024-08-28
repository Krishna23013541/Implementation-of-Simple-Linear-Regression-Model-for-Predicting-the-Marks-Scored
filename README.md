# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: KRISHNA KKUMAR R

RegisterNumber:  212223230107

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
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
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
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

### Dataset
![Screenshot 2024-08-28 085433](https://github.com/user-attachments/assets/8c11c9a1-3830-4994-84cd-3d9c9865f428)

### Head Values
![Screenshot 2024-08-28 085443](https://github.com/user-attachments/assets/021c1fce-f2a6-4f0d-a1b2-e057cbbdf6b4)

### Tail Values
![Screenshot 2024-08-28 085451](https://github.com/user-attachments/assets/35541119-9fcd-405b-b2b0-ac1f2b548308)

### X and Y values
![Screenshot 2024-08-28 085509](https://github.com/user-attachments/assets/a9c9e87e-d3ce-41c6-9c3f-b6b8669318c3)

### Predication values of X and Y
![Screenshot 2024-08-28 085527](https://github.com/user-attachments/assets/e45731c2-a964-4c29-a094-718fb2fbf5e9)

### MSE,MAE and RMSE
![Screenshot 2024-08-28 085558](https://github.com/user-attachments/assets/d6ad2820-0780-497f-a9df-c5d66dda615a)

### Training Set
![Screenshot 2024-08-28 085539](https://github.com/user-attachments/assets/44998365-771c-4a6d-8195-7c1ab21c1fd5)

### Testing Set
![Screenshot 2024-08-28 085550](https://github.com/user-attachments/assets/70e91e47-9760-40de-9a5f-a9e8e084893b)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
