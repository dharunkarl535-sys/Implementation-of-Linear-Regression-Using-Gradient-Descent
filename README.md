# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
*/
```

## Output:
<img width="558" height="222" alt="490484723-e16619ae-fe7f-4005-88eb-43e4b35424a3" src="https://github.com/user-attachments/assets/7645fbf4-171e-4323-9710-815c83894c2f" />

<img width="225" height="713" alt="490485055-4e1cb9f8-f325-42cc-9f4e-d8c0dd00c671" src="https://github.com/user-attachments/assets/25ec791e-38e7-4795-9af6-0f0b937e2a76" />

<img width="343" height="707" alt="490485637-7562b9ba-a041-429a-801a-10b01f3748d5" src="https://github.com/user-attachments/assets/f3b57d77-aa1a-4c5f-b9c7-d72affef62cc" />

<img width="247" height="46" alt="490485838-0c521fab-41b9-44b4-8a26-80883d1a5bec" src="https://github.com/user-attachments/assets/8b3cde33-f2d8-4296-a258-6b1b78de626e" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
