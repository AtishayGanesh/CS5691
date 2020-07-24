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
inv_k_factor = 1
if k_factor<1:
    inv_k_factor = 1/k_factor
    k_factor = 1
clf  =SVC(C=C,class_weight = {1.0:1*inv_k_factor,0.0:1*k_factor},kernel='linear')
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
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5,10)
yy = a * xx - (clf.intercept_[0]) / w[1]

yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin


plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'c--')
plt.xlabel('x_1')
plt.xlabel('x_2')
a = np.where(abs(clf.dual_coef_[0])<1)
q = clf.support_vectors_[a]
print("Margin is",margin)

print("Number of support vectors on marginal hyperplanes",len(q))
sv = plt.plot(q[:,0],q[:,1],'go')
plt.legend(["Class 0","Class 1","Decision Boundary","Class 1 Margin","Class 0 Margin","Support vectors"])
#plt.show()
