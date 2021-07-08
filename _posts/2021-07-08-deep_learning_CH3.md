---
title:  "밑바닥부터 시작하는 딥러닝 CH3"
excerpt: "밑바닥부터 시작하는 딥러닝"

categories:
  - Deep_Learning
tags:
  - [Blog, jekyll, Github, Git, Deep Learning]

toc: true
toc_sticky: true
 
date: 2021-07-08
last_modified_at: 2021-07-08
---

```python
def step_function(x):
  if x>0:
    return 1
  else:
    return 0
```


```python
def step_function(x):
  y = x > 0
  return y.astype(np.int)
```


```python
import numpy as np
x = np.array([-1.0,1.0,2.0])
x
```




    array([-1.,  1.,  2.])




```python
y = x > 0
y
```




    array([False,  True,  True])




```python
y = y.astype(np.int)
y
```




    array([0, 1, 1])




```python
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
  return np.array(x>0, dtype = np.int)

x= np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```


    
![basic_deep_learning_CH3_5_0](https://user-images.githubusercontent.com/55619678/124923648-9dc8c080-e035-11eb-92de-79dde43cbfd9.png)
    



```python
def sigmoid(x):
  return 1/(1+np.exp(-x))
```


```python
x = np.array([-1.0,1.0,2.0])
sigmoid(x)
```




    array([0.26894142, 0.73105858, 0.88079708])




```python
t = np.array([1.0,2.0,3.0])
1.0 + t
```




    array([2., 3., 4.])




```python
1.0 / t
```




    array([1.        , 0.5       , 0.33333333])




```python
x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()
```


    
![basic_deep_learning_CH3_10_0](https://user-images.githubusercontent.com/55619678/124923639-9acdd000-e035-11eb-971b-2df8c6b4fce5.png)
    



```python
def relu(x):
  return np.maximum(0,x)
```


```python
import numpy as np
A = np.array([1,2,3,4])
print(A)
np.ndim(A)

```

    [1 2 3 4]
    




    1




```python
A.shape
```




    (4,)




```python
A.shape[0]
```




    4




```python
B = np.array([[1,2],[3,4],[5,6]])
print(B)
```

    [[1 2]
     [3 4]
     [5 6]]
    


```python
np.ndim(B)
```




    2




```python
B.shape
```




    (3, 2)




```python
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[1,2],[3,4],[5,6]])
np.dot(A, B)
```




    array([[22, 28],
           [49, 64]])




```python
C = np.array([[1,2],[3,4]])
C.shape
```




    (2, 2)




```python
# np.dot(A,C) A,C의 행렬 곱은 불가능
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-30-bef0a2d6fb5b> in <module>()
    ----> 1 np.dot(A,C)
    

    <__array_function__ internals> in dot(*args, **kwargs)
    

    ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)



```python
X = np.array([1,2])
X.shape
```




    (2,)




```python
W = np.array([[1,3,5],[2,4,6]])
print(W)
```

    [[1 3 5]
     [2 4 6]]
    


```python
W.shape
```




    (2, 3)




```python
Y = np.dot(X,W)
```


```python
print(Y)
```

    [ 5 11 17]
    


```python
X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)
```

    (2, 3)
    (2,)
    (3,)
    


```python
A1 = np.dot(X,W1)+B1
```


```python
Z1 = sigmoid(A1)
print(A1)
print(Z1)
```

    [0.3 0.7 1.1]
    [0.57444252 0.66818777 0.75026011]
    


```python
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)
```

    (3,)
    (3, 2)
    (2,)
    


```python
A2 = np.dot(Z1, W2) + B2
```


```python
Z2 = sigmoid(A2)

print(A2)
print(Z2)
```

    [0.51615984 1.21402696]
    [0.62624937 0.7710107 ]
    


```python
def identity_function(x):
  return x

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
```


```python
def init_network():
  network={}
  network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
  network['B1']=np.array([0.1,0.2,0.3])
  network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
  network['B2']=np.array([0.1,0.2])
  network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
  network['B3']=np.array([0.1,0.2])

  return network

def forward(network, x):
  W1, W2, W3 = network['W1'],network['W2'],network['W3']
  B1, B2, B3 = network['B1'],network['B2'],network['B3']

  a1 = np.dot(x,W1)+B1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + B2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + B3
  y = identity_function(a3)
  
  return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x)
print(y)
```

    [0.31682708 0.69627909]
    


```python
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)
```

    [ 1.34985881 18.17414537 54.59815003]
    


```python
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)
```

    74.1221542101633
    


```python
y = exp_a / sum_exp_a
print(y)
```

    [0.01821127 0.24519181 0.73659691]
    


```python
def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y
```

### 이렇게 만든 softmax는 오버플로우 문제가 있을 수 있다. 지수함수가 큰값을 반환하기 때문에


```python
a = np.array([1010,1000,990])
np.exp(a)/np.sum(np.exp(a))
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: RuntimeWarning: overflow encountered in exp
      
    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: RuntimeWarning: invalid value encountered in true_divide
      
    




    array([nan, nan, nan])




```python
c = np.max(a)
a - c
```




    array([  0, -10, -20])




```python
np.exp(a-c)/np.sum(np.exp(a-c))
```




    array([9.99954600e-01, 4.53978686e-05, 2.06106005e-09])




```python
def softmax(a):
  c = np.max(a)
  exp_a = np.exp(a-c)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y
```


```python
a = np.array([0.3, 2.9,4.0])
y = softmax(a)
print(y)
```

    [0.01821127 0.24519181 0.73659691]
    


```python
np.sum(y)
```




    1.0




```python
import tensorflow as tf
from tensorflow import keras
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```


```python
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
```

    (60000, 28, 28)
    (10000, 28, 28)
    (60000,)
    (10000,)
    


```python
X_train = x_train.reshape(x_train.shape[0], 784)
X_test = x_test.reshape(x_test.shape[0], 784)
```


```python
print(X_train.shape)
print(X_test.shape)
```

    (60000, 784)
    (10000, 784)
    


```python
import sys, os
import numpy as np
sys.path.append(os.pardir)
from PIL import Image

def img_show(img):
  pil_img = Image.fromarray(np.uint8(img))
  pil_img.show()

img = X_train[0]
label = y_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)
```

    5
    (784,)
    (28, 28)
    


```python
def get_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  return x_test, y_test

def init_network():
  with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

    return network

def predict(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = softmax(a3)

  return y
```


```python
import pickle
x, y = get_data()
X = x.reshape(x.shape[0], 28*28)
network = init_network()

accuracy_cnt = 0
for i in range(len(X)):
  y_pred = predict(network, X[i])
  p = np.argmax(y_pred)
  if p == y[i]:
    accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt)/len(X)))
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: RuntimeWarning: overflow encountered in exp
      
    

    Accuracy:0.9207
    


```python
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

X = x.reshape(x.shape[0], 28*28)
X.shape
```




    (10000, 784)




```python
X[0].shape
```




    (784,)




```python
W1.shape
```




    (784, 50)




```python
W2.shape
```




    (50, 100)




```python
W3.shape
```




    (100, 10)




```python
x, t = get_data()
network = init_network()

X = x.reshape(x.shape[0], 28*28)

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(X), batch_size):
  x_batch = X[i:i+batch_size]
  y_batch = predict(network, x_batch)
  p = np.argmax(y_batch, axis = 1)
  accuracy_cnt += np.sum(p==t[i:i+batch_size])

print("Accuracy: " + str(float(accuracy_cnt)/len(X)))
```

    Accuracy: 0.9207
    

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: RuntimeWarning: overflow encountered in exp
      
    

np.argmax(,axis = 1)의 의미


```python
x = np.array([[0.1,0.8,0.1],[0.3,0.1,0.6],[0.2,0.5,0.3],[0.8,0.1,0.1]])
print(x.shape)
y_0 = np.argmax(x, axis = 0) #행렬의 열을 기준으로 해서 최대값을 찾는다.
print(y_0)
y_1 = np.argmax(x, axis = 1) #행렬의 행을 기준으로 해서 최대값을 찾는다. 
print(y_1)
```

    (4, 3)
    [3 0 1]
    [1 2 1 0]
    


```python

```
