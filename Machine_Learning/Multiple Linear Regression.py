import pandas as pd
from sklearn import preprocessing,linear_model
import numpy as np
import sklearn

data = pd.read_csv('Machine_Learning/Multiple Linear Regression Dataset/Student_Performance.csv')
data = data [['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced','Performance Index']]
nomalize = preprocessing.LabelEncoder()
data['Extracurricular Activities'] = nomalize.fit_transform((data['Extracurricular Activities']))

# print(data.isnull().sum())
data = data.dropna()
# print(data.head())

x = np.array(data.drop(['Performance Index'], axis=1))
y = np.array(data['Performance Index'])
# print('X',x.shape)
# print('Y',y.shape)
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x,y,test_size=0.2,random_state=10)
# print('XTrain',xTrain.shape)
# print('XTest',xTest.shape)

model = linear_model.LinearRegression()
model.fit(xTrain,yTrain)
accuracy = model.score(xTest,yTest)

testVals = model.predict(xTest)

error = []
for i,testVal in enumerate(testVals) :
        error.append(yTest[i]-testVal)
        print(f'Acutal:{yTest[i]} Prediction:{testVals[i]} Error: {error[i]}')
        if i > 20 : break

# print('Cofficients: ',model.coef_)
# print('Intercept: ', model.intercept_)
print('Accuracy:',round(accuracy*100,2),'%')