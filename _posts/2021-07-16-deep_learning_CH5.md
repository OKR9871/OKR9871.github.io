---
title:  "밑바닥부터 시작하는 딥러닝 CH5"
excerpt: "밑바닥부터 시작하는 딥러닝"

categories:
  - Deep_Learning
tags:
  - [Blog, jekyll, Github, Git, Deep Learning]

toc: true
toc_sticky: true
 
date: 2021-07-16
last_modified_at: 2021-07-16
---
## 오차역전파법    
    
#### 5.1 계산 그래프    
계산 그래프는 계산 과정을 그래프로 나타낸 것입니다.    
계산 그래프는 계산과정을 노드와 화살표로 표현합니다. 노드는 원으로 표시하고, 원 안에 연산내용을 적습니다. 또 계산결과를 화살표 위에 적어 계산결과가 왼쪽에서 오른쪽으로 전해지게 합니다.    

이렇게 계산을 계산 그래프를 이용해서 왼쪽에서 오른쪽으로 진행하는 단계를 **순전파**라고 합니다.    
그 반대로 계산하는 것을 **역전파**라고 합니다.    

계산 그래프의 장점은 국소적 계산이 있습니다. 국소적 계산이란 전체와 달리 일부분의 계산을 진행할 수 있다는 의미입니다. 또 다른 장점으로 역전파를 통한 '미분'을 효율적으로 계산할 수 있다는 장점이 있습니다.   

#### 5.2 연쇄법칙    
역전파의 계산 절차는 들어오는 신호 E에 노드의 국소적미분 $E{\partial y\over\partial x}$을 곱한 후 다음 노드로 전달하는 것입니다.   
연쇄법칙은 합성 함수의 미분에 대한 성질이며, 합성함수의 미분은 합성함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다라는 성질입니다.    

#### 5.4 단순한 계층 구혀하기    
곱셈 계층


```python
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy
```


```python
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)
```

    220.00000000000003
    


```python
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)
```

    2.2 110.00000000000001 200
    


```python
class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout*1
        dy = dout*1
        return dx, dy
```


```python
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price)
print(int(dapple_num), dapple, int(dorange), int(dorange_num), dtax)
```

    715.0000000000001
    110 2.2 3 165 650
    

#### 5.5 활성화 함수 계층 구현하기    

ReLU 계층

ReLU 수식    

$y={\begin{cases}
x & (x>0) \\
0 & (x\le0)
\end{cases}}$    

x에 대한 y의 미분    
${\partial y \over \partial x}={\begin{cases}
1 & (x>0) \\
0 & (x\le0)
\end{cases}}$


```python
class ReLU:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x<=0)           #mask는 True/False로 구성된 numpy배열로 순전파의 입력인 x가 0이하면 인덱스는 True, 아니면 False
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
```


```python
import numpy as np
x = np.array([[1.0, -0.5],[-2.0,3.0]])
print(x)

mask = (x<=0)
print(mask)
```

    [[ 1.  -0.5]
     [-2.   3. ]]
    [[False  True]
     [ True False]]
    

Sigmoid 계층   

$y={1\over 1+exp(-x)}$ 
    
${\partial L \over \partial y}{y^{2}exp(-x)}= {\partial L \over \partial y}y(1-y)$


```python
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out
        
        return out
    
    def backward(self, dout):
        dx = dout*(1.0-self.out)*self.out
        
        return dx
```

#### 5.6 Affine/Softmax 계층 구현하기    

Affine 계층   
신경망의 순전파 때 수행하는 행렬의 곱은 기하학에서는 어파인 변환 이라고 합니다.   

배치용 Affine 계층 구현   


```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        
        return dx
```

Softmax-with-Loss 계층    

softmax 계층은 입력값을 정규화(출력의 합이 1이 되도록 변형)하여 출력합니다.


```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))
```


```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t) / batch_size
        
        return dx
```

오차 역전파법을 적용한 신경망 구하기


```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                weight_init_std = 0.01):
        #가중치 초기화
        self.params = []
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        #계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim!=1 : t = np.argmax(t, axis = 1)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    
    def gradient(self, x, t):
        self.loss(x, t)
        
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads={}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads
```


```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key+":"+str(diff))
```

    W1:1.8205532835517079e-10
    b1:8.863599082348521e-10
    W2:6.975208626518182e-08
    b2:1.3915146198362204e-07
    

오차역전파법을 사용한 학습 구현하기


```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train),(x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)
network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list=[]
train_acc_list=[]
test_acc_list=[]
iter_per_epoch = max(train_size/batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.gradient(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
```

    0.09751666666666667 0.0974
    0.77565 0.7811
    0.8741833333333333 0.8784
    0.89755 0.9005
    0.909 0.9115
    0.9155 0.9173
    0.9205666666666666 0.9217
    0.9247666666666666 0.9254
    0.9268166666666666 0.9285
    0.9312166666666667 0.9322
    0.9345333333333333 0.935
    0.93745 0.9363
    0.9394833333333333 0.9385
    0.94185 0.9391
    0.9443 0.943
    0.9459666666666666 0.945
    0.9477666666666666 0.9474
    


```python

```
