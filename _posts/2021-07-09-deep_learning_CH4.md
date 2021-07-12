---
title:  "밑바닥부터 시작하는 딥러닝 CH4"
excerpt: "밑바닥부터 시작하는 딥러닝"

categories:
  - Deep_Learning
tags:
  - [Blog, jekyll, Github, Git, Deep Learning]

toc: true
toc_sticky: true
 
date: 2021-07-09
last_modified_at: 2021-07-09
---
## 신경망 학습

학습이란 훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 획득하는 것   
이때 지표로 손실 함수를 사용한다.   
#### 손실함수의 값이 최소가 되도록 매개변수를 찾는것이 중요하다.

* * *

#### 4.1 데이터 주도학습

이미지로 부터 특징(feature)를 추출하고 그 특징의 패턴을 기계학습으로 학습하는 방법을 사용한다.   
특징은 보통 벡터로 기술되어진다.   

이때 특징을 추출하기 위해서 hand-crafted로 사람이 직접 feature를 추출하는 방법과 end-to-end방법으로 기계가 자동으로 feature를 추출하는 방법이 있는데 주로 end-to-end방법을 사용한다. 

#### 4.2 손실함수   
- 오차제곱합   
$E=\frac{1}{2}\sum\limits_{k}((y_{k}-t_{k}))^2$     
$y_{k}$는 신경망의 출력, $t_{k}$는 정답 레이블입니다.   
따라서 신경망이 정확할수록 오차는 줄어듭니다.    
- 교차 엔트로피 오차   
$E=-\sum\limits_{k}t_{k}log{y_{k}}$    
$y_{k}$는 신경망의 출력, $t_{k}$는 정답 레이블입니다.   
자연로그 $log_{e}$는 그래프의 모양이 x = 1일때는 0이 되고 x가 0에 가까워질수록 y의 값은 점점 작아집니다. 따라서 정답에 해당하는 출력이 커질수록 0에 가까워져 오류도 0에 가까워 집니다.


```python
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))
```


```python
import numpy as np

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
cross_entropy_error(np.array(y), np.array(t))
```




    0.510825457099338




```python
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
cross_entropy_error(np.array(y), np.array(t))
```




    2.302584092994546



미니배치로 훈련을 진행하는 이유는 많은 훈련데이터를 전부 손실함수를 구하면 시간이 오래 걸립니다. 따라서 근사치를 사용하기 위해 미니매치를 사용합니다. 


```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

print(x_train.shape)
print(t_train.shape)
```

    (60000, 784)
    (60000, 10)
    


```python
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```


```python
np.random.choice(60000, 10)
```




    array([51885,  6454, 49259, 38491, 42170,  2346, 49057, 19758, 41018,
           24638])



(배치용) 교차 엔트로피 오차   
데이터가 한개일 경우와 배치로 들어올 경우 모두 처리하기위한 코드


```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size
```

원핫인코드 방식이 아닌 레이블로 주어질 경우   
np.log(y[np.arange(batch_size), t]   
이부분이 중요한데 이부분은 np.arange(batch_size)는 batch_size-1만큼의 array를 만들어내고 t와 합해져서 예를 들면 y[0,2], y[1,7]과 같은 결과를 만들어낸다.


```python
def cross_entorypy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]+ 1e-7)) / batch_size
```

#### 4.3 미분    
$\frac{\text{d}f(x)}{\text{d}x}=\lim\limits_{h \rightarrow 0}\frac{f(x+h)-f(x)}{h}$   
미분은 다음과 같은 식으로 표기되며 한순간의 변화량을 나타낸다.   

* h의 값으로 약 $10^{-4}$정도를 사용하면 좋은 결과를 얻을 수 있다고 알려져있다. 
* 여기서 하는 것처럼 아주 작은 차분으로 미분하는 것을 수치 미분이라고 합니다. 여기서 차분이란 두함수 값의 차이를 말합니다. 


```python
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)
```


```python
def function_1(x):
    return 0.01*x**2 +0.1*x
```


```python
import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
```


    
![basic_deep_learning_CH4_20_0](https://user-images.githubusercontent.com/55619678/125252277-247de600-e333-11eb-83d6-0b0070fac862.png)

    



```python
print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))
```

    0.1999999999990898
    0.2999999999986347
    

편미분


```python
def function_3(x):
    return x[0]**2+x[1]**2
def function_2(x):
    return np.sum(x**2)
```


```python
x = np.linspace(-3,3,50)
y = np.linspace(-3,3,50)
X, Y = np.meshgrid(x, y)
Z = X**2+Y**2
fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection = '3d')

ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('f(x0, x1)')

ax.contour3D(X, Y, Z, 100, cmap ='viridis')
```




    <matplotlib.contour.QuadContourSet at 0x21754219760>




    
![basic_deep_learning_CH4_24_1](https://user-images.githubusercontent.com/55619678/125252274-234cb900-e333-11eb-91c7-df4b1ce7204a.png)
    


문제 1 : $x_{0}=3, x_{1}=4$일 때, $x_{0}$에 대한 편미분을 구하라


```python
def function_tmp1(x0):
    return x0*x0+4.0**2.0

numerical_diff(function_tmp1, 3.0)
```




    6.00000000000378



#### 4.4 기울기    
모든 변수의 편미분을 벡터로 정리한것을 그레디언트라고 한다. 


```python
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val+h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp_val
    return grad
```


```python
print(numerical_gradient(function_2, np.array([3.0,4.0])))
print(numerical_gradient(function_2, np.array([0.0,2.0])))
print(numerical_gradient(function_2, np.array([3.0,0.0])))
```

    [6. 8.]
    [0. 4.]
    [6. 0.]
    
경사 하강법   
함수가 극솟값, 최솟값, 또 안장점이 되는 장소에서는 기울기가 0입니다. 

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad
    return x
```


```python
def function_2(x):
    return x[0]**2+x[1]**2

init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr = 0.1, step_num=100)
```




    array([-6.11110793e-10,  8.14814391e-10])



학습률이 큰경우 발산한다.


```python
init_x = np.array([-3.0,4.0])
gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)
```




    array([-2.58983747e+13, -1.29524862e+12])



학습률이 작은경우 거의 변화하지 않는다. 


```python
init_x = np.array([-3.0,4.0])
gradient_descent(function_2, init_x=init_x, lr =1e-10,step_num=100)
```




    array([-2.99999994,  3.99999992])




```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss
```


```python
net = simpleNet()
print(net.W)
```

    [[ 2.11212989 -0.25640996 -0.22609281]
     [ 1.47278096  0.20610628  0.31935226]]
    


```python
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
```

    [2.5927808  0.03164967 0.15176135]
    


```python
np.argmax(p)
```




    0




```python
t = np.array([0,0,1])
net.loss(x, t)
```




    2.5931290499994653




```python
def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
```

    [[ 0.51533511  0.03979281 -0.55512792]
     [ 0.77300266  0.05968922 -0.83269188]]
    


```python
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std= 0.01):
        self.params={}
        self.params['W1']=weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2']=np.zeros(output_size)
        
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = sigmoid(a2)
        
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)
        
        accuracy = np.sum(y==t)
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads={}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) 
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
```


```python
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)
```

    (784, 100)
    (100,)
    (100, 10)
    (10,)
    


```python
x = np.random.rand(100,784)
y = net.predict(x)
```


```python
x = np.random.rand(100,784)
t = np.random.rand(100, 10)

grads = net.numerical_gradient(x, t)
print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)
```

    (784, 100)
    (100,)
    (100, 10)
    (10,)
    


```python
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list=[]

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.numerical_gradient(x_batch, t_batch)
    
    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]
        
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
```
