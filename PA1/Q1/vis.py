import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loadData(dataset):
    df = pd.read_csv('Dataset_{}_Team_6.csv'.format(dataset))
    val = df.values
    print(val.shape)
    return val

val = loadData(1)
#np.where(val[2]=0)
print()

C1 = val[:,0:2][np.where(val[:,2]==0)]
C2 = val[:,0:2][np.where(val[:,2]==1)]
C3 = val[:,0:2][np.where(val[:,2]==2)]
X1 = np.split(C1,2,axis=1)
X2 = np.split(C2,2,axis=1)
X3 = np.split(C3,2,axis=1)
plt.plot(X1[0],X1[1],'ro')
plt.plot(X2[0],X2[1],'bx')
plt.plot(X3[0],X3[1],'gd')
plt.xlabel("Dimension X1")
plt.ylabel("Dimesnion X2")
plt.legend(("Class 0","Class 1","Class 2"))
plt.title("Visualisation of Dataset 1")
plt.show()
