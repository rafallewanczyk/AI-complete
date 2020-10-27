

print(format('Image classification using RandomForest: An example in Python using CIFAR10 Dataset','*^88'))

import warnings
warnings.filterwarnings("ignore")    

from keras.datasets import cifar10
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score    
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random
import time
import copy 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll

start_time = time.time()

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

RESHAPED = 3072

X_train = X_train.reshape(50000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.flatten()
y_test = y_test.flatten()

y_10_train = copy.deepcopy(y_train)
y_20_train = copy.deepcopy(y_train)
y_30_train = copy.deepcopy(y_train)
y_40_train = copy.deepcopy(y_train)
y_50_train = copy.deepcopy(y_train)

for i in range(5000):
 y_10_train[i] = random.randint(0, 9)
for i in range(10000):
 y_20_train[i] = random.randint(0, 9)
for i in range(15000):
 y_30_train[i] = random.randint(0, 9)
for i in range(20000):
 y_40_train[i] = random.randint(0, 9)
for i in range(25000):
 y_50_train[i] = random.randint(0, 9)

X_train /= 255.
X_test /= 255.

baseline_model = RandomForestClassifier(n_estimators = 10)
model_10 = RandomForestClassifier(n_estimators = 10)
model_20 = RandomForestClassifier(n_estimators = 10)
model_30 = RandomForestClassifier(n_estimators = 10)
model_40 = RandomForestClassifier(n_estimators = 10)
model_50 = RandomForestClassifier(n_estimators = 10)

baseline_model.fit(X_train, y_train)
model_10.fit(X_train,y_10_train)
model_20.fit(X_train,y_20_train)
model_30.fit(X_train,y_30_train)
model_40.fit(X_train,y_40_train)
model_50.fit(X_train,y_50_train)

expected_y  = y_test
predicted_y = baseline_model.predict(X_test)
predicted_10_y = model_10.predict(X_test)
predicted_20_y = model_20.predict(X_test)
predicted_30_y = model_30.predict(X_test)
predicted_40_y = model_40.predict(X_test)
predicted_50_y = model_50.predict(X_test)

print(accuracy_score(predicted_y, expected_y))
print(accuracy_score(predicted_10_y, expected_y))
print(accuracy_score(predicted_20_y, expected_y))
print(accuracy_score(predicted_30_y, expected_y))
print(accuracy_score(predicted_40_y, expected_y))
print(accuracy_score(predicted_50_y, expected_y))

x = np.arange(1,7)
y = [accuracy_score(predicted_y, expected_y),accuracy_score(predicted_10_y, expected_y),accuracy_score(predicted_20_y, expected_y)
,accuracy_score(predicted_30_y, expected_y),accuracy_score(predicted_40_y, expected_y),accuracy_score(predicted_50_y, expected_y)]

y_10_train = copy.deepcopy(y_train)
y_20_train = copy.deepcopy(y_train)
y_30_train = copy.deepcopy(y_train)
y_40_train = copy.deepcopy(y_train)
y_50_train = copy.deepcopy(y_train)

y_10_train = y_train
y_10_train = changeTheLabel(y_10_train,5000)
y_20_train = y_train
y_20_train = changeTheLabel(y_20_train,10000)
y_30_train = y_train
y_30_train = changeTheLabel(y_30_train,15000)
y_40_train = y_train
y_40_train = changeTheLabel(y_40_train,20000)
y_50_train = y_train
y_50_train = changeTheLabel(y_50_train,25000)

X_train /= 255.
X_test /= 255.

model_10 = RandomForestClassifier(n_estimators = 10)
model_20 = RandomForestClassifier(n_estimators = 10)
model_30 = RandomForestClassifier(n_estimators = 10)
model_40 = RandomForestClassifier(n_estimators = 10)
model_50 = RandomForestClassifier(n_estimators = 10)

model_10.fit(X_train,y_10_train)
model_20.fit(X_train,y_20_train)
model_30.fit(X_train,y_30_train)
model_40.fit(X_train,y_40_train)
model_50.fit(X_train,y_50_train)

expected_y  = y_test
predicted_10_y = model_10.predict(X_test)
predicted_20_y = model_20.predict(X_test)
predicted_30_y = model_30.predict(X_test)
predicted_40_y = model_40.predict(X_test)
predicted_50_y = model_50.predict(X_test)

print(accuracy_score(predicted_y, expected_y))
print(accuracy_score(predicted_10_y, expected_y))
print(accuracy_score(predicted_20_y, expected_y))
print(accuracy_score(predicted_30_y, expected_y))
print(accuracy_score(predicted_40_y, expected_y))
print(accuracy_score(predicted_50_y, expected_y))

x_asi = np.arange(1,7)
y_asi = [accuracy_score(predicted_y, expected_y),accuracy_score(predicted_10_y, expected_y),accuracy_score(predicted_20_y, expected_y)
,accuracy_score(predicted_30_y, expected_y),accuracy_score(predicted_40_y, expected_y),accuracy_score(predicted_50_y, expected_y)]

my_xticks = ['baise line','10% noise','20% noise','30% noise','40% noise','50% noise']
plt.xticks(x_asi, my_xticks)
plt.plot(x, y,'r')
plt.plot(x_asi, y_asi,'b')
plt.title('Random Forest')
plt.ylabel('accuracy')
plt.xlabel('noise')
plt.legend(['simetric','asimetric'], loc='upper left')
plt.show()

from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn import metrics
from keras.datasets import cifar10
import copy 
import random
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

(X_train, y_train), (X_test , y_test) = cifar10.load_data()
X_train_flat = X_train.reshape(X_train.shape[0], X_train.shape[1]* X_train.shape[2]* X_train.shape[3])
X_test_flat = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3])
y_10_train = copy.deepcopy(y_train)
y_20_train = copy.deepcopy(y_train)
y_30_train = copy.deepcopy(y_train)
y_40_train = copy.deepcopy(y_train)
y_50_train = copy.deepcopy(y_train)
for i in range(1000):
 y_10_train[i] = random.randint(0, 9)
for i in range(10000):
 y_20_train[i] = random.randint(0, 9)
for i in range(15000):
 y_30_train[i] = random.randint(0, 9)
for i in range(25000):
 y_40_train[i] = random.randint(0, 9)
for i in range(40000):
 y_50_train[i] = random.randint(0, 9)

a = AdaBoostClassifier(n_estimators=50,learning_rate=1)
B = AdaBoostClassifier(n_estimators=50,learning_rate=1)
C = AdaBoostClassifier(n_estimators=50,learning_rate=1)
D = AdaBoostClassifier(n_estimators=50,learning_rate=1)
E = AdaBoostClassifier(n_estimators=50,learning_rate=1)
F = AdaBoostClassifier(n_estimators=50,learning_rate=1)

baseline_model = a.fit(X_train_flat, y_train)
model_10 = B.fit(X_train_flat, y_10_train)
model_20 = C.fit(X_train_flat, y_20_train)
model_30 = D.fit(X_train_flat, y_30_train)
model_40 = E.fit(X_train_flat, y_40_train)
model_50 = F.fit(X_train_flat, y_50_train)

y_pred = baseline_model.predict(X_test_flat)
y_pred_10 = model_10.predict(X_test_flat)
y_pred_20 = model_20.predict(X_test_flat)
y_pred_30 = model_30.predict(X_test_flat)
y_pred_40 = model_40.predict(X_test_flat)
y_pred_50 = model_50.predict(X_test_flat)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_10))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_20))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_30))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_40))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_50))

x = np.arange(1,7)
y = [accuracy_score(y_pred, y_test),accuracy_score(y_pred_10, y_test),accuracy_score(y_pred_20, y_test)
,accuracy_score(y_pred_30, y_test),accuracy_score(y_pred_40, y_test),accuracy_score(y_pred_50, y_test)]

def changeTheLabel(labelNoise, num):
  for i in range(num):
    if(labelNoise[i]==0):
      labelNoise[i]=5
    elif (labelNoise[i]==1):
      labelNoise[i]=6
    elif (labelNoise[i]==2):
      labelNoise[i]=8
    elif (labelNoise[i]==3):
      labelNoise[i]=7
    elif (labelNoise[i]==4):
      labelNoise[i]=9
    elif (labelNoise[i]==5):
      labelNoise[i]=0
    elif (labelNoise[i]==6):
      labelNoise[i]=1
    elif (labelNoise[i]==7):
      labelNoise[i]=3
    elif (labelNoise[i]==8):
      labelNoise[i]=2
    elif (labelNoise[i]==9):
      labelNoise[i]=4
  return labelNoise

y_10_train = copy.deepcopy(y_train)
y_20_train = copy.deepcopy(y_train)
y_30_train = copy.deepcopy(y_train)
y_40_train = copy.deepcopy(y_train)
y_50_train = copy.deepcopy(y_train)
y_10_train = changeTheLabel(y_10_train,5000)
y_20_train = changeTheLabel(y_20_train,10000)
y_30_train = changeTheLabel(y_30_train,15000)
y_40_train = changeTheLabel(y_40_train,20000)
y_50_train = changeTheLabel(y_50_train,25000)

a = AdaBoostClassifier(n_estimators=50,learning_rate=1)
B = AdaBoostClassifier(n_estimators=50,learning_rate=1)
C = AdaBoostClassifier(n_estimators=50,learning_rate=1)
D = AdaBoostClassifier(n_estimators=50,learning_rate=1)
E = AdaBoostClassifier(n_estimators=50,learning_rate=1)
F = AdaBoostClassifier(n_estimators=50,learning_rate=1)

baseline_model = a.fit(X_train_flat, y_train)
model_10 = B.fit(X_train_flat, y_10_train)
model_20 = C.fit(X_train_flat, y_20_train)
model_30 = D.fit(X_train_flat, y_30_train)
model_40 = E.fit(X_train_flat, y_40_train)
model_50 = F.fit(X_train_flat, y_50_train)

y_pred = baseline_model.predict(X_test_flat)
y_pred_10 = model_10.predict(X_test_flat)
y_pred_20 = model_20.predict(X_test_flat)
y_pred_30 = model_30.predict(X_test_flat)
y_pred_40 = model_40.predict(X_test_flat)
y_pred_50 = model_50.predict(X_test_flat)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_10))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_20))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_30))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_40))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_50))

x_asi = np.arange(1,7)
y_asi = [accuracy_score(y_pred, y_test),accuracy_score(y_pred_10, y_test),accuracy_score(y_pred_20, y_test)
,accuracy_score(y_pred_30, y_test),accuracy_score(y_pred_40, y_test),accuracy_score(y_pred_50, y_test)]

import matplotlib.pyplot as plt
my_xticks = ['baise line','10% noise','20% noise','30% noise','40% noise','50% noise']
plt.xticks(x_asi, my_xticks)
plt.plot(x, y,'r')
plt.plot(x_asi, y_asi,'b')
plt.title('AdaBoost')
plt.ylabel('accuracy')
plt.xlabel('noise')
plt.legend(['symetric','asymetric'], loc='upper left')
plt.show()

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
x = np.arange(1,7)
y = [accuracy_score(y_pred, y_test),accuracy_score(y_pred_10, y_test),accuracy_score(y_pred_20, y_test)
,accuracy_score(y_pred_30, y_test),accuracy_score(y_pred_40, y_test),accuracy_score(y_pred_50, y_test)]

def changeTheLabel(labelNoise, num):
  for i in range(num):
    if(labelNoise[i]==0):
      labelNoise[i]=5
    elif (labelNoise[i]==1):
      labelNoise[i]=6
    elif (labelNoise[i]==2):
      labelNoise[i]=8
    elif (labelNoise[i]==3):
      labelNoise[i]=7
    elif (labelNoise[i]==4):
      labelNoise[i]=9
    elif (labelNoise[i]==5):
      labelNoise[i]=0
    elif (labelNoise[i]==6):
      labelNoise[i]=1
    elif (labelNoise[i]==7):
      labelNoise[i]=3
    elif (labelNoise[i]==8):
      labelNoise[i]=2
    elif (labelNoise[i]==9):
      labelNoise[i]=4
  return labelNoise

y_10_train = copy.deepcopy(y_train)
y_20_train = copy.deepcopy(y_train)
y_30_train = copy.deepcopy(y_train)
y_40_train = copy.deepcopy(y_train)
y_50_train = copy.deepcopy(y_train)
y_10_train = changeTheLabel(y_10_train,5000)
y_20_train = changeTheLabel(y_20_train,10000)
y_30_train = changeTheLabel(y_30_train,15000)
y_40_train = changeTheLabel(y_40_train,20000)
y_50_train = changeTheLabel(y_50_train,25000)

a = AdaBoostClassifier(n_estimators=50,learning_rate=1)
B = AdaBoostClassifier(n_estimators=50,learning_rate=1)
C = AdaBoostClassifier(n_estimators=50,learning_rate=1)
D = AdaBoostClassifier(n_estimators=50,learning_rate=1)
E = AdaBoostClassifier(n_estimators=50,learning_rate=1)
F = AdaBoostClassifier(n_estimators=50,learning_rate=1)

baseline_model = a.fit(X_train_flat, y_train)
model_10 = B.fit(X_train_flat, y_10_train)
model_20 = C.fit(X_train_flat, y_20_train)
model_30 = D.fit(X_train_flat, y_30_train)
model_40 = E.fit(X_train_flat, y_40_train)
model_50 = F.fit(X_train_flat, y_50_train)

y_pred = baseline_model.predict(X_test_flat)
y_pred_10 = model_10.predict(X_test_flat)
y_pred_20 = model_20.predict(X_test_flat)
y_pred_30 = model_30.predict(X_test_flat)
y_pred_40 = model_40.predict(X_test_flat)
y_pred_50 = model_50.predict(X_test_flat)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_10))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_20))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_30))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_40))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_50))

x_asi = np.arange(1,7)
y_asi = [accuracy_score(y_pred, y_test),accuracy_score(y_pred_10, y_test),accuracy_score(y_pred_20, y_test)
,accuracy_score(y_pred_30, y_test),accuracy_score(y_pred_40, y_test),accuracy_score(y_pred_50, y_test)]

import matplotlib.pyplot as plt
my_xticks = ['baise line','10% noise','20% noise','30% noise','40% noise','50% noise']
plt.xticks(x_asi, my_xticks)
plt.plot(x, y,'r')
plt.plot(x_asi, y_asi,'b')
plt.title('AdaBoost')
plt.ylabel('accuracy')
plt.xlabel('noise')
plt.legend(['symetric','asymetric'], loc='upper left')
plt.show()


EOF
