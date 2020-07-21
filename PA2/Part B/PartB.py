import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix as cnf
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df = pd.read_csv('Dataset_2_Team_8.csv')
val = df.values

parser = argparse.ArgumentParser()
parser.add_argument('--C',type=float,default = 1.0)
parser.add_argument('--k_factor',type=float,default = 1.0)
args = parser.parse_args()
C = args.C
k_factor = args.k_factor
X = val[:,0:2]
y = val[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=7)
clf  =SVC(C=C,class_weight = {1.0:1,0.0:1*k_factor},kernel='rbf')
clf.fit(X_train,y_train)

pred_test = clf.predict(X_test)
pred_train = clf.predict(X_train)
print('Training Accuracy with C =',C,'is  ',clf.score(X_train,y_train))
print('Testing Accuracy with C =',C,'is  ',clf.score(X_test,y_test))
print("Confusion Matrix")
print(cnf(y_test,pred_test))

gridx = np.linspace(-2.5,2.5,100)
gridy = np.linspace(-2.5,2.5,100)
xv,yv = np.meshgrid(gridx,gridy)

decision_function = clf.predict(np.array(list(zip(xv.flatten(),yv.flatten()))))

X1 = X[np.where(y==0)]
X2 = X[np.where(y==1)]



plt.plot(X1[:,0],X1[:,1],'ro')
plt.plot(X2[:,0],X2[:,1],'bo')

plt.contourf(xv,yv,decision_function.reshape(100,100))
plt.show()