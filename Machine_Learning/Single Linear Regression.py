from sklearn import model_selection, linear_model
import matplotlib.pyplot as plt
import numpy as np
import random

x = list(range(0,40)) 
# y = [1.8*F +32 for F in x]
 
y = [1.8*F +32 + random.randint(-10,10) for F in x] 

x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x,y,test_size=0.2)
model = linear_model.LinearRegression()
model.fit(xTrain,yTrain)
accuracy = model.score(xTest,yTest)


print(f'Cofficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
print(f'Accuracy: {round(accuracy*100,2)}')


plt.plot(x,y,'-*r')
plt.show()
