---
title:  "밑바닥부터 시작하는 딥러닝 CH6"
excerpt: "밑바닥부터 시작하는 딥러닝"

categories:
  - Deep_Learning
tags:
  - [Blog, jekyll, Github, Git, Deep Learning]

toc: true
toc_sticky: true
 
date: 2021-07-18
last_modified_at: 2021-07-18
---
## 6. 학습 관련 기술들   
#### 6.1 매개변수 갱신   
신경망 학습의 목적은 손실함수 값을 최소화 하는것입니다.   
이는 곧 매개변수의 최적값을 찾는 문제이며, 이러한 문제를 푸는 것을 **최적화(optimization)**라고 합니다. 

매개변수의 기울기를 구해, 최소화되는 방향으로 매개변수 값을 갱신하는 일을 반복해서 최적의 값에 다가가는 방법을 **SGD(확률적 경사 하강법)**이라 합니다.   
- SGD


```python
class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr
        
    def updata(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]
```

SGD의 단점   
SGD는 단순하고 구현도 쉽지만, 비효율적일 때가 있다.   
비등방성(anisotropy)함수(방향에 따라 성질이 달라지는)에서는 비효율적이다.   
따라서 무작정 기울어진 방향보다는 좀더 영리한 방법이 필요하다.    

- 모멘텀   
모멘텀은 수식으로 다음과 같이 쓸 수 있다.   
$v\leftarrow\alpha v-\ \eta{\partial L\over \partial W}$   
$W\leftarrow W + v$    
$W$는 가중치 매개변수, $\partial L\over \partial W$는 $W$에 대한 손실함수, $\eta$는 학습률, $v$는 속도에 해당합니다.   


```python
class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def updata(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self - self.lr*grads[key]
            params[key] += self.v[key]
```

- AdaGrad   
신경망에서는 학습률값이 중요합니다. 이 값이 너무 작으면 학습시간이 길어지고, 반대로 너무 크면 발산하여 학습이 이뤄지지 않습니다.    

이 학습률을 정하는 효과적 기술로 학습률감소(learning rate decay)가 있습니다.   
이는 학습을 진행하면서 학습률을 점차 줄여가는 방법입니다.   
이를 발전시킨것이 Adaptive Gradient(AdaGrad)입니다.    

$h\leftarrow h + {\partial L\over\partial W}\bigodot{\partial L\over\partial W}$    
$W\leftarrow W - \eta{1\over\sqrt{h}}{\partial L\over\partial W}$   
$W$는 갱신할 가중치 매개변수, ${\partial L\over\partial W}$는 $W$에 대한 손실함수의 기울기, $\eta$는 학습률, h는 기본 기울기값을 제곱해 계속 더해줍니다. $\bigodot$은 행렬의 원소별 곱셈을 뜻합니다. 그리고 $1\over \sqrt{h}$를 통해 학습률을 조절합니다.    


```python
class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key]/(np.sqrt(self.h[key])+1e-7)
```

- Adam   
Momentum과 AdaGrad를 융합한 아이디어가 Adam입니다.   

이 모든 optimizer중 어떤 것을 채택하면 효율적이되는지는 아직 정해진 바가 없습니다.   
각자의 장단점이 있어 문제별로 다르기 때문입니다.   


```python
# coding: utf-8
import os
import sys
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()
#optimizers['RMSprop'] = RMSprop()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []    


for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
    
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()

```

    ===========iteration:0===========
    SGD:2.399037763279977
    Momentum:2.3307062543018624
    AdaGrad:2.2949851056849244
    Adam:2.1303956508921598
    ===========iteration:100===========
    SGD:1.4376636787188573
    Momentum:0.4500997046347307
    AdaGrad:0.24468228590138372
    Adam:0.3840438928478588
    ===========iteration:200===========
    SGD:0.778150553625472
    Momentum:0.27817207159347257
    AdaGrad:0.15315683180013964
    Adam:0.24172233457159165
    ===========iteration:300===========
    SGD:0.523444416874991
    Momentum:0.23304903723319118
    AdaGrad:0.06921845055970476
    Adam:0.1480283643580775
    ===========iteration:400===========
    SGD:0.373233110100778
    Momentum:0.21177134927813263
    AdaGrad:0.0716093198209484
    Adam:0.17695189260999653
    ===========iteration:500===========
    SGD:0.38930356760787405
    Momentum:0.21132633425394184
    AdaGrad:0.12361928339998339
    Adam:0.1795569755170776
    ===========iteration:600===========
    SGD:0.2982595450398221
    Momentum:0.14117019538995101
    AdaGrad:0.04135033294850504
    Adam:0.0936292711540151
    ===========iteration:700===========
    SGD:0.2976702524059395
    Momentum:0.1142190804259127
    AdaGrad:0.03887960261174589
    Adam:0.06580664287044585
    ===========iteration:800===========
    SGD:0.2886724961442954
    Momentum:0.15808788220364633
    AdaGrad:0.08677271718770059
    Adam:0.1312107210363082
    ===========iteration:900===========
    SGD:0.24087336363990353
    Momentum:0.11794043192662199
    AdaGrad:0.03693941640481577
    Adam:0.03225034550727049
    ===========iteration:1000===========
    SGD:0.2764785416505178
    Momentum:0.09586566006933428
    AdaGrad:0.027116106624502086
    Adam:0.06366798287473861
    ===========iteration:1100===========
    SGD:0.2902150989472989
    Momentum:0.17039022439705756
    AdaGrad:0.029788732904201257
    Adam:0.05655750321911517
    ===========iteration:1200===========
    SGD:0.18367208775942112
    Momentum:0.048848169020129284
    AdaGrad:0.02315501006618212
    Adam:0.03532658656444294
    ===========iteration:1300===========
    SGD:0.1266121299542161
    Momentum:0.029555627824572224
    AdaGrad:0.019316958260368753
    Adam:0.016707005280737767
    ===========iteration:1400===========
    SGD:0.22343285640043953
    Momentum:0.14083489423543252
    AdaGrad:0.05645444885178416
    Adam:0.11776083723218242
    ===========iteration:1500===========
    SGD:0.15309184112755303
    Momentum:0.032789801796386256
    AdaGrad:0.02026794989438116
    Adam:0.04987422649699578
    ===========iteration:1600===========
    SGD:0.2817434930944731
    Momentum:0.15931173392569953
    AdaGrad:0.11514348152458029
    Adam:0.09251602123987909
    ===========iteration:1700===========
    SGD:0.14704995321919834
    Momentum:0.0850278163230222
    AdaGrad:0.02147441608931494
    Adam:0.06298960413363039
    ===========iteration:1800===========
    SGD:0.2144266635737794
    Momentum:0.05602173216214455
    AdaGrad:0.031346745198853096
    Adam:0.03809940078655434
    ===========iteration:1900===========
    SGD:0.1709093207817624
    Momentum:0.07345873474473925
    AdaGrad:0.025074129254225336
    Adam:0.03542437803483117
    


    
![basic_deep_learning_CH6_9_1](https://user-images.githubusercontent.com/55619678/126061193-e715119c-6994-4278-84a9-af01ae94f9f4.png)   
    


#### 6.2 가중치의 초기값   
오버피팅을 억제해 범용 성능을 높이는 가중치 감쇠(weight decay)기법을 소개합니다.   
가중치 감쇠는 간단히 가중치 매개변수의 값이 작아지도록 학습하는 방법입니다.   
가중치의 초깃값을 0부터 설정하는 방법은 좋은 방법이 아닙니다.   
왜냐하면 모두 같은 크기를 가지고 있어 가중치의 값이 똑같이 갱신 되기 때문입니다. 따라서 가중치를 무작위로 설정해야합니다.   


```python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

input_data = np.random.randn(1000,100)
node_num = 100
hidden_layer_size = 5
activations = {}

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
        
    #w = np.random.randn(node_num, node_num) * 1
    #w = np.random.randn(node_num, node_num) * 0.01
    #w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)   #Xavier 초기값 : 표준편차를 1/root(n)으로 만들어준다. 활성화 값들을 광범위하게 분포시킬 목적으로 가중치의 분포를 조절(활섬함수가 선형일 경우에)
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)   #He 초기값 : 활성함수가 ReLU인경우 초기화를 위해 사용하는 함수
        
    a = np.dot(x, w)
        
    z = sigmoid(a)
    # z = ReLU(a)
    # z = tanh(a)
        
    activations[i] = z
        
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1)+"-layer")
    if i != 0:
        plt.yticks([],[])
        #plt.xlim(0.1, 1)
        #plt.ylim(0, 7000)
        plt.hist(a.flatten(), 30, range=(0,1))
        
plt.show()
```


    
![basic_deep_learning_CH6_11_0](https://user-images.githubusercontent.com/55619678/126061194-87eca9a7-7a60-48b1-bb25-863cdf8e5324.png)
    


MNIST 데이터셋으로 가중치 초기값 비교


```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train),(x_test, t_test) = load_mnist(normalize = True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

weight_init_types = {'std=0.01':0.01, 'Xavier' : 'sigmoid', 'He':'relu'}
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size = 784, hidden_size_list=[100,100,100,100],
                                 output_size = 10,weight_init_std=weight_type)
    train_loss[key] = []
    
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)
        
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
        
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))
            
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0,2.5)
plt.legend()
plt.show()
```

    ===========iteration:0===========
    std=0.01:2.3025130997351715
    Xavier:2.3093680991888825
    He:2.300381148301722
    ===========iteration:100===========
    std=0.01:2.302724286178096
    Xavier:2.255622969722515
    He:1.564101868067709
    ===========iteration:200===========
    std=0.01:2.30340254686908
    Xavier:2.135445167218747
    He:0.8754664342920397
    ===========iteration:300===========
    std=0.01:2.298707145263265
    Xavier:1.7227873461097039
    He:0.5247086453889338
    ===========iteration:400===========
    std=0.01:2.302827604108482
    Xavier:1.2464490612709462
    He:0.5580491665292746
    ===========iteration:500===========
    std=0.01:2.3006057252195937
    Xavier:0.7728942565842384
    He:0.39187691777613093
    ===========iteration:600===========
    std=0.01:2.3017921166251263
    Xavier:0.6062240514933568
    He:0.3350068762892483
    ===========iteration:700===========
    std=0.01:2.3004458020644183
    Xavier:0.6286686160451573
    He:0.3714535874318532
    ===========iteration:800===========
    std=0.01:2.3060190835772776
    Xavier:0.48035750424232765
    He:0.2610431990225758
    ===========iteration:900===========
    std=0.01:2.3018515562734594
    Xavier:0.3444741545806169
    He:0.20025456290427623
    ===========iteration:1000===========
    std=0.01:2.3027877612715177
    Xavier:0.4130417189406872
    He:0.29573823613699846
    ===========iteration:1100===========
    std=0.01:2.3010494363777436
    Xavier:0.397592142431178
    He:0.29121003075383944
    ===========iteration:1200===========
    std=0.01:2.3024507160895133
    Xavier:0.5001859809928603
    He:0.37800933659116964
    ===========iteration:1300===========
    std=0.01:2.307206144723759
    Xavier:0.3700411196553751
    He:0.2465939746246114
    ===========iteration:1400===========
    std=0.01:2.301444958301751
    Xavier:0.37433939903519653
    He:0.19146783769389253
    ===========iteration:1500===========
    std=0.01:2.2966048156446144
    Xavier:0.25945076256468974
    He:0.19060579843021758
    ===========iteration:1600===========
    std=0.01:2.2941925848196605
    Xavier:0.455470323787488
    He:0.3460982688770952
    ===========iteration:1700===========
    std=0.01:2.3019183091519926
    Xavier:0.422522109653829
    He:0.381538034693619
    ===========iteration:1800===========
    std=0.01:2.3055663176147485
    Xavier:0.2723635081291297
    He:0.19634182154030466
    ===========iteration:1900===========
    std=0.01:2.3022311552679633
    Xavier:0.2520785756633468
    He:0.19659069211119978
    


    
![basic_deep_learning_CH6_13_1](https://user-images.githubusercontent.com/55619678/126061195-490cc23c-570e-4e69-8601-69b147070052.png)
    


#### 6.3 배치 정규화   
앞 절에서는 각 층의 활성화 분포를 관찰해 보며, 가중치의 초깃값을 적절히 설정하면 각 층의 활성화값 분포가 적당히 퍼지면서 학습이 수행됨을 알 수 있었습니다.    
각 층이 활성화를 적당히 퍼뜨리도록 '강제'하는 방법을 **배치 정규화** 라고 합니다.   
배치 정규화는 그 이름과 같이 학습 시 미니배치 단우로 정규화 합니다.   
구체적으로는 평균이 0, 분산이 1이 되도록 정규화 합니다.   
$\hat{x_{i}}={x_{i}-\mu{B} \over {\sqrt{\sigma_{B}^2 + \epsilon}}}$    
또한 각 배치정규화 계층 마다 정규화된 데이터에 고유한 확대와 이동변환을 수행합니다.    
$y_{i}\leftarrow \gamma \hat{x_{i}}+\beta$   
여기서 $\gamma$는 확대를, $\beta$가 이동을 담당한다.   


```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

(x_train, t_train),(x_test, t_test) = load_mnist(normalize = True)

x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size = 784, hidden_size_list=[100,100,100,100,100], output_size = 10,
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = max(train_size/batch_size, 1)
    epoch_cnt = 0
    
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
            
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
            
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list, bn_train_acc_list

weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print( "============== " + str(i+1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)
    
    plt.subplot(4,4,i+1)
    plt.title("W:"+str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')
    
plt.show()
        
```

    ============== 1/16 ==============
    epoch:0 | 0.1 - 0.11
    

    C:\Users\LG\deep-learning-from-scratch\common\multi_layer_net_extend.py:101: RuntimeWarning: overflow encountered in square
      weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
    C:\Users\LG\deep-learning-from-scratch\common\multi_layer_net_extend.py:101: RuntimeWarning: invalid value encountered in double_scalars
      weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
    C:\Users\LG\anaconda3\lib\site-packages\numpy\core\fromnumeric.py:87: RuntimeWarning: overflow encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    

    epoch:1 | 0.097 - 0.11
    epoch:2 | 0.097 - 0.126
    epoch:3 | 0.097 - 0.15
    epoch:4 | 0.097 - 0.179
    epoch:5 | 0.097 - 0.204
    epoch:6 | 0.097 - 0.222
    epoch:7 | 0.097 - 0.249
    epoch:8 | 0.097 - 0.27
    epoch:9 | 0.097 - 0.285
    epoch:10 | 0.097 - 0.297
    epoch:11 | 0.097 - 0.318
    epoch:12 | 0.097 - 0.334
    epoch:13 | 0.097 - 0.349
    epoch:14 | 0.097 - 0.356
    epoch:15 | 0.097 - 0.378
    epoch:16 | 0.097 - 0.387
    epoch:17 | 0.097 - 0.403
    epoch:18 | 0.097 - 0.418
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.097 - 0.429
    ============== 2/16 ==============
    epoch:0 | 0.092 - 0.091
    

    C:\Users\LG\deep-learning-from-scratch\common\multi_layer_net_extend.py:101: RuntimeWarning: overflow encountered in square
      weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
    C:\Users\LG\deep-learning-from-scratch\common\multi_layer_net_extend.py:101: RuntimeWarning: invalid value encountered in double_scalars
      weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
    C:\Users\LG\deep-learning-from-scratch\common\functions.py:32: RuntimeWarning: invalid value encountered in subtract
      x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    

    epoch:1 | 0.097 - 0.095
    epoch:2 | 0.097 - 0.146
    epoch:3 | 0.097 - 0.186
    epoch:4 | 0.097 - 0.211
    epoch:5 | 0.097 - 0.247
    epoch:6 | 0.097 - 0.274
    epoch:7 | 0.097 - 0.29
    epoch:8 | 0.097 - 0.316
    epoch:9 | 0.097 - 0.334
    epoch:10 | 0.097 - 0.365
    epoch:11 | 0.097 - 0.385
    epoch:12 | 0.097 - 0.405
    epoch:13 | 0.097 - 0.416
    epoch:14 | 0.097 - 0.439
    epoch:15 | 0.097 - 0.458
    epoch:16 | 0.097 - 0.483
    epoch:17 | 0.097 - 0.498
    epoch:18 | 0.097 - 0.505
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.097 - 0.526
    ============== 3/16 ==============
    epoch:0 | 0.1 - 0.077
    epoch:1 | 0.379 - 0.123
    epoch:2 | 0.523 - 0.17
    epoch:3 | 0.609 - 0.206
    epoch:4 | 0.683 - 0.258
    epoch:5 | 0.748 - 0.31
    epoch:6 | 0.805 - 0.335
    epoch:7 | 0.831 - 0.367
    epoch:8 | 0.861 - 0.4
    epoch:9 | 0.896 - 0.43
    epoch:10 | 0.93 - 0.444
    epoch:11 | 0.927 - 0.479
    epoch:12 | 0.95 - 0.501
    epoch:13 | 0.955 - 0.521
    epoch:14 | 0.963 - 0.559
    epoch:15 | 0.976 - 0.576
    epoch:16 | 0.976 - 0.591
    epoch:17 | 0.98 - 0.613
    epoch:18 | 0.984 - 0.623
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.987 - 0.632
    ============== 4/16 ==============
    epoch:0 | 0.126 - 0.116
    epoch:1 | 0.27 - 0.168
    epoch:2 | 0.398 - 0.239
    epoch:3 | 0.487 - 0.307
    epoch:4 | 0.58 - 0.392
    epoch:5 | 0.631 - 0.441
    epoch:6 | 0.663 - 0.482
    epoch:7 | 0.691 - 0.527
    epoch:8 | 0.712 - 0.58
    epoch:9 | 0.741 - 0.614
    epoch:10 | 0.755 - 0.637
    epoch:11 | 0.777 - 0.673
    epoch:12 | 0.772 - 0.694
    epoch:13 | 0.798 - 0.722
    epoch:14 | 0.809 - 0.744
    epoch:15 | 0.824 - 0.766
    epoch:16 | 0.831 - 0.779
    epoch:17 | 0.832 - 0.799
    epoch:18 | 0.847 - 0.808
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.844 - 0.813
    ============== 5/16 ==============
    epoch:0 | 0.083 - 0.077
    epoch:1 | 0.093 - 0.141
    epoch:2 | 0.113 - 0.246
    epoch:3 | 0.126 - 0.37
    epoch:4 | 0.145 - 0.485
    epoch:5 | 0.164 - 0.578
    epoch:6 | 0.176 - 0.625
    epoch:7 | 0.181 - 0.678
    epoch:8 | 0.193 - 0.722
    epoch:9 | 0.208 - 0.761
    epoch:10 | 0.214 - 0.779
    epoch:11 | 0.236 - 0.8
    epoch:12 | 0.253 - 0.823
    epoch:13 | 0.266 - 0.836
    epoch:14 | 0.284 - 0.845
    epoch:15 | 0.289 - 0.865
    epoch:16 | 0.315 - 0.875
    epoch:17 | 0.328 - 0.882
    epoch:18 | 0.333 - 0.892
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.327 - 0.907
    ============== 6/16 ==============
    epoch:0 | 0.094 - 0.135
    epoch:1 | 0.099 - 0.182
    epoch:2 | 0.116 - 0.514
    epoch:3 | 0.123 - 0.644
    epoch:4 | 0.116 - 0.716
    epoch:5 | 0.118 - 0.755
    epoch:6 | 0.127 - 0.793
    epoch:7 | 0.117 - 0.827
    epoch:8 | 0.117 - 0.833
    epoch:9 | 0.117 - 0.853
    epoch:10 | 0.117 - 0.869
    epoch:11 | 0.117 - 0.884
    epoch:12 | 0.117 - 0.892
    epoch:13 | 0.117 - 0.9
    epoch:14 | 0.117 - 0.91
    epoch:15 | 0.117 - 0.919
    epoch:16 | 0.117 - 0.922
    epoch:17 | 0.117 - 0.935
    epoch:18 | 0.117 - 0.941
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.117 - 0.948
    ============== 7/16 ==============
    epoch:0 | 0.116 - 0.116
    epoch:1 | 0.116 - 0.264
    epoch:2 | 0.116 - 0.531
    epoch:3 | 0.116 - 0.689
    epoch:4 | 0.116 - 0.754
    epoch:5 | 0.116 - 0.806
    epoch:6 | 0.117 - 0.826
    epoch:7 | 0.117 - 0.855
    epoch:8 | 0.117 - 0.891
    epoch:9 | 0.117 - 0.901
    epoch:10 | 0.117 - 0.915
    epoch:11 | 0.117 - 0.935
    epoch:12 | 0.117 - 0.943
    epoch:13 | 0.117 - 0.955
    epoch:14 | 0.117 - 0.966
    epoch:15 | 0.117 - 0.966
    epoch:16 | 0.117 - 0.974
    epoch:17 | 0.117 - 0.982
    epoch:18 | 0.117 - 0.987
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.117 - 0.987
    ============== 8/16 ==============
    epoch:0 | 0.116 - 0.111
    epoch:1 | 0.099 - 0.245
    epoch:2 | 0.105 - 0.648
    epoch:3 | 0.105 - 0.738
    epoch:4 | 0.116 - 0.774
    epoch:5 | 0.116 - 0.82
    epoch:6 | 0.117 - 0.879
    epoch:7 | 0.117 - 0.928
    epoch:8 | 0.116 - 0.945
    epoch:9 | 0.116 - 0.968
    epoch:10 | 0.116 - 0.976
    epoch:11 | 0.116 - 0.981
    epoch:12 | 0.116 - 0.987
    epoch:13 | 0.116 - 0.991
    epoch:14 | 0.116 - 0.994
    epoch:15 | 0.116 - 0.995
    epoch:16 | 0.116 - 0.997
    epoch:17 | 0.116 - 0.998
    epoch:18 | 0.116 - 0.999
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.116 - 1.0
    ============== 9/16 ==============
    epoch:0 | 0.1 - 0.107
    epoch:1 | 0.116 - 0.358
    epoch:2 | 0.116 - 0.689
    epoch:3 | 0.116 - 0.796
    epoch:4 | 0.116 - 0.86
    epoch:5 | 0.116 - 0.893
    epoch:6 | 0.116 - 0.919
    epoch:7 | 0.116 - 0.956
    epoch:8 | 0.116 - 0.963
    epoch:9 | 0.116 - 0.957
    epoch:10 | 0.116 - 0.99
    epoch:11 | 0.116 - 0.991
    epoch:12 | 0.116 - 0.996
    epoch:13 | 0.116 - 0.997
    epoch:14 | 0.116 - 0.998
    epoch:15 | 0.116 - 0.998
    epoch:16 | 0.116 - 0.998
    epoch:17 | 0.116 - 0.998
    epoch:18 | 0.116 - 0.999
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.116 - 0.999
    ============== 10/16 ==============
    epoch:0 | 0.092 - 0.151
    epoch:1 | 0.117 - 0.435
    epoch:2 | 0.116 - 0.751
    epoch:3 | 0.116 - 0.787
    epoch:4 | 0.116 - 0.929
    epoch:5 | 0.105 - 0.944
    epoch:6 | 0.105 - 0.977
    epoch:7 | 0.105 - 0.968
    epoch:8 | 0.105 - 0.988
    epoch:9 | 0.117 - 0.986
    epoch:10 | 0.116 - 0.992
    epoch:11 | 0.116 - 0.976
    epoch:12 | 0.117 - 0.992
    epoch:13 | 0.116 - 0.996
    epoch:14 | 0.116 - 0.998
    epoch:15 | 0.116 - 0.998
    epoch:16 | 0.116 - 0.999
    epoch:17 | 0.116 - 1.0
    epoch:18 | 0.116 - 1.0
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.116 - 1.0
    ============== 11/16 ==============
    epoch:0 | 0.116 - 0.174
    epoch:1 | 0.105 - 0.564
    epoch:2 | 0.116 - 0.659
    epoch:3 | 0.116 - 0.658
    epoch:4 | 0.116 - 0.677
    epoch:5 | 0.117 - 0.767
    epoch:6 | 0.117 - 0.774
    epoch:7 | 0.117 - 0.788
    epoch:8 | 0.117 - 0.881
    epoch:9 | 0.117 - 0.861
    epoch:10 | 0.117 - 0.884
    epoch:11 | 0.117 - 0.898
    epoch:12 | 0.116 - 0.894
    epoch:13 | 0.116 - 0.898
    epoch:14 | 0.116 - 0.914
    epoch:15 | 0.117 - 0.989
    epoch:16 | 0.117 - 0.991
    epoch:17 | 0.117 - 0.982
    epoch:18 | 0.117 - 0.991
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.117 - 0.993
    ============== 12/16 ==============
    epoch:0 | 0.116 - 0.168
    epoch:1 | 0.116 - 0.426
    epoch:2 | 0.117 - 0.564
    epoch:3 | 0.117 - 0.646
    epoch:4 | 0.116 - 0.715
    epoch:5 | 0.117 - 0.629
    epoch:6 | 0.117 - 0.644
    epoch:7 | 0.117 - 0.697
    epoch:8 | 0.117 - 0.697
    epoch:9 | 0.117 - 0.7
    epoch:10 | 0.117 - 0.757
    epoch:11 | 0.117 - 0.697
    epoch:12 | 0.117 - 0.718
    epoch:13 | 0.117 - 0.669
    epoch:14 | 0.117 - 0.777
    epoch:15 | 0.117 - 0.714
    epoch:16 | 0.117 - 0.716
    epoch:17 | 0.117 - 0.812
    epoch:18 | 0.117 - 0.803
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.117 - 0.813
    ============== 13/16 ==============
    epoch:0 | 0.116 - 0.144
    epoch:1 | 0.105 - 0.371
    epoch:2 | 0.117 - 0.59
    epoch:3 | 0.117 - 0.43
    epoch:4 | 0.116 - 0.637
    epoch:5 | 0.117 - 0.628
    epoch:6 | 0.117 - 0.661
    epoch:7 | 0.117 - 0.677
    epoch:8 | 0.117 - 0.63
    epoch:9 | 0.117 - 0.645
    epoch:10 | 0.117 - 0.679
    epoch:11 | 0.117 - 0.69
    epoch:12 | 0.117 - 0.704
    epoch:13 | 0.117 - 0.7
    epoch:14 | 0.117 - 0.707
    epoch:15 | 0.117 - 0.639
    epoch:16 | 0.117 - 0.705
    epoch:17 | 0.117 - 0.715
    epoch:18 | 0.117 - 0.667
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.117 - 0.716
    ============== 14/16 ==============
    epoch:0 | 0.092 - 0.168
    epoch:1 | 0.116 - 0.285
    epoch:2 | 0.117 - 0.36
    epoch:3 | 0.117 - 0.364
    epoch:4 | 0.117 - 0.488
    epoch:5 | 0.117 - 0.502
    epoch:6 | 0.117 - 0.509
    epoch:7 | 0.117 - 0.521
    epoch:8 | 0.117 - 0.517
    epoch:9 | 0.117 - 0.519
    epoch:10 | 0.117 - 0.535
    epoch:11 | 0.117 - 0.597
    epoch:12 | 0.117 - 0.636
    epoch:13 | 0.117 - 0.64
    epoch:14 | 0.117 - 0.636
    epoch:15 | 0.116 - 0.693
    epoch:16 | 0.116 - 0.688
    epoch:17 | 0.116 - 0.703
    epoch:18 | 0.116 - 0.715
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.116 - 0.712
    ============== 15/16 ==============
    epoch:0 | 0.105 - 0.097
    epoch:1 | 0.105 - 0.301
    epoch:2 | 0.116 - 0.378
    epoch:3 | 0.116 - 0.399
    epoch:4 | 0.116 - 0.4
    epoch:5 | 0.117 - 0.403
    epoch:6 | 0.116 - 0.393
    epoch:7 | 0.116 - 0.4
    epoch:8 | 0.116 - 0.402
    epoch:9 | 0.116 - 0.394
    epoch:10 | 0.116 - 0.407
    epoch:11 | 0.116 - 0.411
    epoch:12 | 0.116 - 0.414
    epoch:13 | 0.116 - 0.407
    epoch:14 | 0.116 - 0.407
    epoch:15 | 0.116 - 0.405
    epoch:16 | 0.117 - 0.412
    epoch:17 | 0.116 - 0.421
    epoch:18 | 0.117 - 0.509
    

    No handles with labels found to put in legend.
    

    epoch:19 | 0.117 - 0.527
    ============== 16/16 ==============
    epoch:0 | 0.097 - 0.097
    epoch:1 | 0.117 - 0.259
    epoch:2 | 0.117 - 0.347
    epoch:3 | 0.117 - 0.388
    epoch:4 | 0.117 - 0.423
    epoch:5 | 0.116 - 0.423
    epoch:6 | 0.117 - 0.422
    epoch:7 | 0.117 - 0.373
    epoch:8 | 0.117 - 0.409
    epoch:9 | 0.117 - 0.415
    epoch:10 | 0.117 - 0.423
    epoch:11 | 0.117 - 0.426
    epoch:12 | 0.117 - 0.429
    epoch:13 | 0.117 - 0.43
    epoch:14 | 0.117 - 0.431
    epoch:15 | 0.116 - 0.431
    epoch:16 | 0.117 - 0.433
    epoch:17 | 0.117 - 0.432
    epoch:18 | 0.117 - 0.432
    epoch:19 | 0.116 - 0.431
    


    
![basic_deep_learning_CH6_15_35](https://user-images.githubusercontent.com/55619678/126061197-fc47c9e2-a6de-43de-bd0a-05fc808f4630.png)
    


#### 6.4 바른 학습을 위해

오버피팅은 다음 두 경우에 일어납니다.    
- 매개변수가 많고, 표현력이 높은 모델    
- 훈련데이터가 적음    
일부러 오버피팅을 일으켜보겠습니다.    


```python
# coding: utf-8
import os
import sys

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train = x_train[:300]
t_train = t_train[:300]

weight_decay_lambda = 0 
#weight_decay_lambda = 0.1

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
```

    epoch:0, train acc:0.14, test acc:0.107
    epoch:1, train acc:0.16666666666666666, test acc:0.131
    epoch:2, train acc:0.18333333333333332, test acc:0.1521
    epoch:3, train acc:0.19666666666666666, test acc:0.165
    epoch:4, train acc:0.2, test acc:0.1766
    epoch:5, train acc:0.21666666666666667, test acc:0.1868
    epoch:6, train acc:0.23333333333333334, test acc:0.2038
    epoch:7, train acc:0.25666666666666665, test acc:0.2169
    epoch:8, train acc:0.2633333333333333, test acc:0.2302
    epoch:9, train acc:0.2866666666666667, test acc:0.2426
    epoch:10, train acc:0.30333333333333334, test acc:0.2507
    epoch:11, train acc:0.31333333333333335, test acc:0.2572
    epoch:12, train acc:0.32, test acc:0.2672
    epoch:13, train acc:0.32666666666666666, test acc:0.2714
    epoch:14, train acc:0.34, test acc:0.2777
    epoch:15, train acc:0.35, test acc:0.2819
    epoch:16, train acc:0.35333333333333333, test acc:0.2941
    epoch:17, train acc:0.36333333333333334, test acc:0.2966
    epoch:18, train acc:0.37333333333333335, test acc:0.3044
    epoch:19, train acc:0.38666666666666666, test acc:0.3068
    epoch:20, train acc:0.37333333333333335, test acc:0.3073
    epoch:21, train acc:0.4, test acc:0.3185
    epoch:22, train acc:0.44, test acc:0.3395
    epoch:23, train acc:0.44, test acc:0.3465
    epoch:24, train acc:0.45666666666666667, test acc:0.3558
    epoch:25, train acc:0.46, test acc:0.3539
    epoch:26, train acc:0.47333333333333333, test acc:0.3657
    epoch:27, train acc:0.49666666666666665, test acc:0.3927
    epoch:28, train acc:0.5, test acc:0.4041
    epoch:29, train acc:0.5066666666666667, test acc:0.4095
    epoch:30, train acc:0.5166666666666667, test acc:0.4146
    epoch:31, train acc:0.53, test acc:0.4287
    epoch:32, train acc:0.54, test acc:0.4343
    epoch:33, train acc:0.5433333333333333, test acc:0.4359
    epoch:34, train acc:0.5533333333333333, test acc:0.4452
    epoch:35, train acc:0.5866666666666667, test acc:0.4551
    epoch:36, train acc:0.5966666666666667, test acc:0.4693
    epoch:37, train acc:0.6133333333333333, test acc:0.478
    epoch:38, train acc:0.6533333333333333, test acc:0.5142
    epoch:39, train acc:0.68, test acc:0.5199
    epoch:40, train acc:0.6866666666666666, test acc:0.5333
    epoch:41, train acc:0.7166666666666667, test acc:0.5464
    epoch:42, train acc:0.7, test acc:0.5403
    epoch:43, train acc:0.7366666666666667, test acc:0.5586
    epoch:44, train acc:0.7633333333333333, test acc:0.5734
    epoch:45, train acc:0.78, test acc:0.5854
    epoch:46, train acc:0.79, test acc:0.5955
    epoch:47, train acc:0.8233333333333334, test acc:0.6029
    epoch:48, train acc:0.84, test acc:0.6317
    epoch:49, train acc:0.85, test acc:0.641
    epoch:50, train acc:0.86, test acc:0.6496
    epoch:51, train acc:0.8566666666666667, test acc:0.647
    epoch:52, train acc:0.86, test acc:0.6439
    epoch:53, train acc:0.8533333333333334, test acc:0.6538
    epoch:54, train acc:0.8666666666666667, test acc:0.6507
    epoch:55, train acc:0.8766666666666667, test acc:0.6612
    epoch:56, train acc:0.8766666666666667, test acc:0.6749
    epoch:57, train acc:0.9066666666666666, test acc:0.6816
    epoch:58, train acc:0.91, test acc:0.6841
    epoch:59, train acc:0.9066666666666666, test acc:0.6736
    epoch:60, train acc:0.91, test acc:0.6891
    epoch:61, train acc:0.9133333333333333, test acc:0.6975
    epoch:62, train acc:0.9166666666666666, test acc:0.7012
    epoch:63, train acc:0.9166666666666666, test acc:0.6898
    epoch:64, train acc:0.92, test acc:0.6986
    epoch:65, train acc:0.92, test acc:0.6971
    epoch:66, train acc:0.9166666666666666, test acc:0.7054
    epoch:67, train acc:0.92, test acc:0.7004
    epoch:68, train acc:0.9166666666666666, test acc:0.7135
    epoch:69, train acc:0.9233333333333333, test acc:0.7079
    epoch:70, train acc:0.9333333333333333, test acc:0.7102
    epoch:71, train acc:0.93, test acc:0.7085
    epoch:72, train acc:0.94, test acc:0.7185
    epoch:73, train acc:0.9366666666666666, test acc:0.7211
    epoch:74, train acc:0.9433333333333334, test acc:0.7208
    epoch:75, train acc:0.9466666666666667, test acc:0.7255
    epoch:76, train acc:0.95, test acc:0.7206
    epoch:77, train acc:0.9533333333333334, test acc:0.7272
    epoch:78, train acc:0.9433333333333334, test acc:0.7267
    epoch:79, train acc:0.96, test acc:0.7274
    epoch:80, train acc:0.95, test acc:0.7299
    epoch:81, train acc:0.9433333333333334, test acc:0.7221
    epoch:82, train acc:0.95, test acc:0.7227
    epoch:83, train acc:0.9566666666666667, test acc:0.7292
    epoch:84, train acc:0.9533333333333334, test acc:0.731
    epoch:85, train acc:0.96, test acc:0.7294
    epoch:86, train acc:0.96, test acc:0.7276
    epoch:87, train acc:0.96, test acc:0.7345
    epoch:88, train acc:0.9533333333333334, test acc:0.7292
    epoch:89, train acc:0.9566666666666667, test acc:0.7297
    epoch:90, train acc:0.96, test acc:0.7338
    epoch:91, train acc:0.9633333333333334, test acc:0.738
    epoch:92, train acc:0.97, test acc:0.7393
    epoch:93, train acc:0.9633333333333334, test acc:0.739
    epoch:94, train acc:0.97, test acc:0.7383
    epoch:95, train acc:0.97, test acc:0.736
    epoch:96, train acc:0.9666666666666667, test acc:0.7353
    epoch:97, train acc:0.9733333333333334, test acc:0.7414
    epoch:98, train acc:0.9766666666666667, test acc:0.7427
    epoch:99, train acc:0.9766666666666667, test acc:0.7395
    epoch:100, train acc:0.9766666666666667, test acc:0.734
    epoch:101, train acc:0.9766666666666667, test acc:0.7413
    epoch:102, train acc:0.98, test acc:0.7429
    epoch:103, train acc:0.9833333333333333, test acc:0.7463
    epoch:104, train acc:0.98, test acc:0.7453
    epoch:105, train acc:0.9833333333333333, test acc:0.7461
    epoch:106, train acc:0.9866666666666667, test acc:0.7404
    epoch:107, train acc:0.9833333333333333, test acc:0.7443
    epoch:108, train acc:0.99, test acc:0.742
    epoch:109, train acc:0.9866666666666667, test acc:0.7441
    epoch:110, train acc:0.99, test acc:0.7452
    epoch:111, train acc:0.99, test acc:0.7412
    epoch:112, train acc:0.9933333333333333, test acc:0.7423
    epoch:113, train acc:0.99, test acc:0.7416
    epoch:114, train acc:0.9866666666666667, test acc:0.7462
    epoch:115, train acc:0.9833333333333333, test acc:0.7466
    epoch:116, train acc:0.9933333333333333, test acc:0.7504
    epoch:117, train acc:0.9866666666666667, test acc:0.749
    epoch:118, train acc:0.99, test acc:0.7482
    epoch:119, train acc:0.9933333333333333, test acc:0.7517
    epoch:120, train acc:0.99, test acc:0.7511
    epoch:121, train acc:0.9933333333333333, test acc:0.7488
    epoch:122, train acc:0.9933333333333333, test acc:0.7472
    epoch:123, train acc:0.9933333333333333, test acc:0.748
    epoch:124, train acc:0.9966666666666667, test acc:0.7499
    epoch:125, train acc:0.9933333333333333, test acc:0.7492
    epoch:126, train acc:0.9966666666666667, test acc:0.7511
    epoch:127, train acc:0.9966666666666667, test acc:0.7514
    epoch:128, train acc:0.9933333333333333, test acc:0.7486
    epoch:129, train acc:0.9966666666666667, test acc:0.7484
    epoch:130, train acc:0.9966666666666667, test acc:0.7501
    epoch:131, train acc:0.9933333333333333, test acc:0.7516
    epoch:132, train acc:0.9966666666666667, test acc:0.7486
    epoch:133, train acc:0.9966666666666667, test acc:0.7445
    epoch:134, train acc:0.9966666666666667, test acc:0.7488
    epoch:135, train acc:1.0, test acc:0.7497
    epoch:136, train acc:0.9966666666666667, test acc:0.7527
    epoch:137, train acc:0.9966666666666667, test acc:0.7513
    epoch:138, train acc:0.9966666666666667, test acc:0.7534
    epoch:139, train acc:0.9966666666666667, test acc:0.7542
    epoch:140, train acc:0.9966666666666667, test acc:0.7553
    epoch:141, train acc:0.9966666666666667, test acc:0.7527
    epoch:142, train acc:1.0, test acc:0.754
    epoch:143, train acc:1.0, test acc:0.7525
    epoch:144, train acc:1.0, test acc:0.7547
    epoch:145, train acc:1.0, test acc:0.7522
    epoch:146, train acc:1.0, test acc:0.7531
    epoch:147, train acc:1.0, test acc:0.7573
    epoch:148, train acc:1.0, test acc:0.7551
    epoch:149, train acc:1.0, test acc:0.7556
    epoch:150, train acc:1.0, test acc:0.7565
    epoch:151, train acc:1.0, test acc:0.7538
    epoch:152, train acc:1.0, test acc:0.7554
    epoch:153, train acc:1.0, test acc:0.7524
    epoch:154, train acc:1.0, test acc:0.7549
    epoch:155, train acc:1.0, test acc:0.7554
    epoch:156, train acc:1.0, test acc:0.7546
    epoch:157, train acc:1.0, test acc:0.7558
    epoch:158, train acc:1.0, test acc:0.7551
    epoch:159, train acc:1.0, test acc:0.7556
    epoch:160, train acc:1.0, test acc:0.7549
    epoch:161, train acc:1.0, test acc:0.7542
    epoch:162, train acc:1.0, test acc:0.7574
    epoch:163, train acc:1.0, test acc:0.7562
    epoch:164, train acc:1.0, test acc:0.7548
    epoch:165, train acc:1.0, test acc:0.7562
    epoch:166, train acc:1.0, test acc:0.7563
    epoch:167, train acc:1.0, test acc:0.7563
    epoch:168, train acc:1.0, test acc:0.7578
    epoch:169, train acc:1.0, test acc:0.7573
    epoch:170, train acc:1.0, test acc:0.7547
    epoch:171, train acc:1.0, test acc:0.7556
    epoch:172, train acc:1.0, test acc:0.7559
    epoch:173, train acc:1.0, test acc:0.7563
    epoch:174, train acc:1.0, test acc:0.7568
    epoch:175, train acc:1.0, test acc:0.7554
    epoch:176, train acc:1.0, test acc:0.7573
    epoch:177, train acc:1.0, test acc:0.7561
    epoch:178, train acc:1.0, test acc:0.7574
    epoch:179, train acc:1.0, test acc:0.7576
    epoch:180, train acc:1.0, test acc:0.7587
    epoch:181, train acc:1.0, test acc:0.7582
    epoch:182, train acc:1.0, test acc:0.7589
    epoch:183, train acc:1.0, test acc:0.7586
    epoch:184, train acc:1.0, test acc:0.7582
    epoch:185, train acc:1.0, test acc:0.7569
    epoch:186, train acc:1.0, test acc:0.7569
    epoch:187, train acc:1.0, test acc:0.7563
    epoch:188, train acc:1.0, test acc:0.7568
    epoch:189, train acc:1.0, test acc:0.7581
    epoch:190, train acc:1.0, test acc:0.7586
    epoch:191, train acc:1.0, test acc:0.7584
    epoch:192, train acc:1.0, test acc:0.7594
    epoch:193, train acc:1.0, test acc:0.7584
    epoch:194, train acc:1.0, test acc:0.7595
    epoch:195, train acc:1.0, test acc:0.759
    epoch:196, train acc:1.0, test acc:0.7586
    epoch:197, train acc:1.0, test acc:0.757
    epoch:198, train acc:1.0, test acc:0.7587
    epoch:199, train acc:1.0, test acc:0.7565
    epoch:200, train acc:1.0, test acc:0.759
    


    
![basic_deep_learning_CH6_18_1](https://user-images.githubusercontent.com/55619678/126061198-7554b073-a4b3-40b0-9057-005f3b8b4de7.png)
    



```python
# coding: utf-8
import os
import sys

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train = x_train[:300]
t_train = t_train[:300]

#weight_decay_lambda = 0 
weight_decay_lambda = 0.1

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
```

    epoch:0, train acc:0.09, test acc:0.1113
    epoch:1, train acc:0.10666666666666667, test acc:0.1218
    epoch:2, train acc:0.12, test acc:0.1331
    epoch:3, train acc:0.15, test acc:0.1432
    epoch:4, train acc:0.15, test acc:0.1531
    epoch:5, train acc:0.18, test acc:0.1722
    epoch:6, train acc:0.18666666666666668, test acc:0.1796
    epoch:7, train acc:0.22333333333333333, test acc:0.1897
    epoch:8, train acc:0.26, test acc:0.2087
    epoch:9, train acc:0.29, test acc:0.2312
    epoch:10, train acc:0.31666666666666665, test acc:0.2406
    epoch:11, train acc:0.35, test acc:0.2601
    epoch:12, train acc:0.37666666666666665, test acc:0.2796
    epoch:13, train acc:0.38, test acc:0.2819
    epoch:14, train acc:0.38666666666666666, test acc:0.2954
    epoch:15, train acc:0.43666666666666665, test acc:0.3297
    epoch:16, train acc:0.44333333333333336, test acc:0.3308
    epoch:17, train acc:0.48333333333333334, test acc:0.3453
    epoch:18, train acc:0.49666666666666665, test acc:0.3595
    epoch:19, train acc:0.4866666666666667, test acc:0.3593
    epoch:20, train acc:0.5066666666666667, test acc:0.3663
    epoch:21, train acc:0.5266666666666666, test acc:0.3754
    epoch:22, train acc:0.5333333333333333, test acc:0.3771
    epoch:23, train acc:0.5433333333333333, test acc:0.3844
    epoch:24, train acc:0.5533333333333333, test acc:0.394
    epoch:25, train acc:0.5366666666666666, test acc:0.3879
    epoch:26, train acc:0.5333333333333333, test acc:0.3891
    epoch:27, train acc:0.56, test acc:0.4033
    epoch:28, train acc:0.5566666666666666, test acc:0.3962
    epoch:29, train acc:0.57, test acc:0.4072
    epoch:30, train acc:0.6, test acc:0.4188
    epoch:31, train acc:0.5966666666666667, test acc:0.4182
    epoch:32, train acc:0.5766666666666667, test acc:0.4102
    epoch:33, train acc:0.5966666666666667, test acc:0.4227
    epoch:34, train acc:0.57, test acc:0.4178
    epoch:35, train acc:0.5866666666666667, test acc:0.4254
    epoch:36, train acc:0.5933333333333334, test acc:0.4334
    epoch:37, train acc:0.6533333333333333, test acc:0.4548
    epoch:38, train acc:0.67, test acc:0.4678
    epoch:39, train acc:0.6366666666666667, test acc:0.4503
    epoch:40, train acc:0.6533333333333333, test acc:0.4663
    epoch:41, train acc:0.6733333333333333, test acc:0.478
    epoch:42, train acc:0.6866666666666666, test acc:0.4732
    epoch:43, train acc:0.6766666666666666, test acc:0.4779
    epoch:44, train acc:0.6766666666666666, test acc:0.4748
    epoch:45, train acc:0.69, test acc:0.4849
    epoch:46, train acc:0.7, test acc:0.4915
    epoch:47, train acc:0.7233333333333334, test acc:0.5103
    epoch:48, train acc:0.7533333333333333, test acc:0.5184
    epoch:49, train acc:0.74, test acc:0.522
    epoch:50, train acc:0.75, test acc:0.5393
    epoch:51, train acc:0.7566666666666667, test acc:0.5349
    epoch:52, train acc:0.76, test acc:0.5444
    epoch:53, train acc:0.75, test acc:0.5429
    epoch:54, train acc:0.76, test acc:0.5535
    epoch:55, train acc:0.76, test acc:0.571
    epoch:56, train acc:0.78, test acc:0.595
    epoch:57, train acc:0.7666666666666667, test acc:0.5685
    epoch:58, train acc:0.78, test acc:0.5769
    epoch:59, train acc:0.8, test acc:0.6051
    epoch:60, train acc:0.81, test acc:0.6205
    epoch:61, train acc:0.8033333333333333, test acc:0.6116
    epoch:62, train acc:0.7933333333333333, test acc:0.6046
    epoch:63, train acc:0.79, test acc:0.5977
    epoch:64, train acc:0.83, test acc:0.6326
    epoch:65, train acc:0.8, test acc:0.6291
    epoch:66, train acc:0.81, test acc:0.6254
    epoch:67, train acc:0.8133333333333334, test acc:0.6363
    epoch:68, train acc:0.8133333333333334, test acc:0.626
    epoch:69, train acc:0.84, test acc:0.6507
    epoch:70, train acc:0.83, test acc:0.6529
    epoch:71, train acc:0.8233333333333334, test acc:0.6552
    epoch:72, train acc:0.8366666666666667, test acc:0.6498
    epoch:73, train acc:0.84, test acc:0.6539
    epoch:74, train acc:0.8333333333333334, test acc:0.6483
    epoch:75, train acc:0.83, test acc:0.6531
    epoch:76, train acc:0.85, test acc:0.6559
    epoch:77, train acc:0.8233333333333334, test acc:0.6456
    epoch:78, train acc:0.84, test acc:0.6531
    epoch:79, train acc:0.8333333333333334, test acc:0.6428
    epoch:80, train acc:0.8466666666666667, test acc:0.6506
    epoch:81, train acc:0.8466666666666667, test acc:0.6596
    epoch:82, train acc:0.8566666666666667, test acc:0.6743
    epoch:83, train acc:0.8666666666666667, test acc:0.6737
    epoch:84, train acc:0.85, test acc:0.6752
    epoch:85, train acc:0.85, test acc:0.6781
    epoch:86, train acc:0.8566666666666667, test acc:0.6741
    epoch:87, train acc:0.8633333333333333, test acc:0.681
    epoch:88, train acc:0.86, test acc:0.6875
    epoch:89, train acc:0.85, test acc:0.6764
    epoch:90, train acc:0.8666666666666667, test acc:0.6725
    epoch:91, train acc:0.8433333333333334, test acc:0.6624
    epoch:92, train acc:0.85, test acc:0.6662
    epoch:93, train acc:0.85, test acc:0.671
    epoch:94, train acc:0.88, test acc:0.6888
    epoch:95, train acc:0.8833333333333333, test acc:0.6921
    epoch:96, train acc:0.8833333333333333, test acc:0.6934
    epoch:97, train acc:0.8566666666666667, test acc:0.6769
    epoch:98, train acc:0.87, test acc:0.6841
    epoch:99, train acc:0.8733333333333333, test acc:0.6882
    epoch:100, train acc:0.8733333333333333, test acc:0.6884
    epoch:101, train acc:0.88, test acc:0.6941
    epoch:102, train acc:0.8766666666666667, test acc:0.6934
    epoch:103, train acc:0.8866666666666667, test acc:0.6995
    epoch:104, train acc:0.88, test acc:0.6901
    epoch:105, train acc:0.8933333333333333, test acc:0.7025
    epoch:106, train acc:0.87, test acc:0.6738
    epoch:107, train acc:0.8733333333333333, test acc:0.6972
    epoch:108, train acc:0.8833333333333333, test acc:0.7055
    epoch:109, train acc:0.8833333333333333, test acc:0.6964
    epoch:110, train acc:0.8833333333333333, test acc:0.7004
    epoch:111, train acc:0.8833333333333333, test acc:0.689
    epoch:112, train acc:0.8733333333333333, test acc:0.6853
    epoch:113, train acc:0.8833333333333333, test acc:0.7026
    epoch:114, train acc:0.8866666666666667, test acc:0.701
    epoch:115, train acc:0.8833333333333333, test acc:0.6963
    epoch:116, train acc:0.8866666666666667, test acc:0.6949
    epoch:117, train acc:0.89, test acc:0.7101
    epoch:118, train acc:0.88, test acc:0.6997
    epoch:119, train acc:0.89, test acc:0.7101
    epoch:120, train acc:0.8866666666666667, test acc:0.7051
    epoch:121, train acc:0.8866666666666667, test acc:0.698
    epoch:122, train acc:0.8866666666666667, test acc:0.7015
    epoch:123, train acc:0.8833333333333333, test acc:0.7099
    epoch:124, train acc:0.8833333333333333, test acc:0.7043
    epoch:125, train acc:0.8866666666666667, test acc:0.7051
    epoch:126, train acc:0.89, test acc:0.7131
    epoch:127, train acc:0.8933333333333333, test acc:0.7032
    epoch:128, train acc:0.8866666666666667, test acc:0.7008
    epoch:129, train acc:0.8666666666666667, test acc:0.6903
    epoch:130, train acc:0.8866666666666667, test acc:0.706
    epoch:131, train acc:0.8933333333333333, test acc:0.7068
    epoch:132, train acc:0.89, test acc:0.7103
    epoch:133, train acc:0.8866666666666667, test acc:0.7127
    epoch:134, train acc:0.88, test acc:0.7067
    epoch:135, train acc:0.8733333333333333, test acc:0.7071
    epoch:136, train acc:0.8766666666666667, test acc:0.713
    epoch:137, train acc:0.8766666666666667, test acc:0.7048
    epoch:138, train acc:0.8766666666666667, test acc:0.705
    epoch:139, train acc:0.8766666666666667, test acc:0.7081
    epoch:140, train acc:0.8933333333333333, test acc:0.7119
    epoch:141, train acc:0.8966666666666666, test acc:0.7154
    epoch:142, train acc:0.8933333333333333, test acc:0.7164
    epoch:143, train acc:0.8966666666666666, test acc:0.7108
    epoch:144, train acc:0.89, test acc:0.702
    epoch:145, train acc:0.8933333333333333, test acc:0.7148
    epoch:146, train acc:0.89, test acc:0.7087
    epoch:147, train acc:0.89, test acc:0.7132
    epoch:148, train acc:0.8833333333333333, test acc:0.7118
    epoch:149, train acc:0.8766666666666667, test acc:0.7068
    epoch:150, train acc:0.8933333333333333, test acc:0.7165
    epoch:151, train acc:0.8833333333333333, test acc:0.7037
    epoch:152, train acc:0.8833333333333333, test acc:0.7126
    epoch:153, train acc:0.8833333333333333, test acc:0.7148
    epoch:154, train acc:0.8933333333333333, test acc:0.7112
    epoch:155, train acc:0.89, test acc:0.7192
    epoch:156, train acc:0.89, test acc:0.712
    epoch:157, train acc:0.8833333333333333, test acc:0.7166
    epoch:158, train acc:0.88, test acc:0.7092
    epoch:159, train acc:0.89, test acc:0.7158
    epoch:160, train acc:0.88, test acc:0.7178
    epoch:161, train acc:0.89, test acc:0.7128
    epoch:162, train acc:0.8866666666666667, test acc:0.7148
    epoch:163, train acc:0.8833333333333333, test acc:0.7117
    epoch:164, train acc:0.88, test acc:0.7007
    epoch:165, train acc:0.89, test acc:0.7173
    epoch:166, train acc:0.89, test acc:0.723
    epoch:167, train acc:0.89, test acc:0.7103
    epoch:168, train acc:0.89, test acc:0.7108
    epoch:169, train acc:0.89, test acc:0.7158
    epoch:170, train acc:0.8966666666666666, test acc:0.7128
    epoch:171, train acc:0.8933333333333333, test acc:0.7085
    epoch:172, train acc:0.8833333333333333, test acc:0.7111
    epoch:173, train acc:0.8933333333333333, test acc:0.7207
    epoch:174, train acc:0.8966666666666666, test acc:0.7223
    epoch:175, train acc:0.8966666666666666, test acc:0.7246
    epoch:176, train acc:0.8966666666666666, test acc:0.7257
    epoch:177, train acc:0.9033333333333333, test acc:0.7161
    epoch:178, train acc:0.8833333333333333, test acc:0.7189
    epoch:179, train acc:0.9033333333333333, test acc:0.7212
    epoch:180, train acc:0.8833333333333333, test acc:0.7148
    epoch:181, train acc:0.89, test acc:0.7202
    epoch:182, train acc:0.8966666666666666, test acc:0.721
    epoch:183, train acc:0.8966666666666666, test acc:0.7157
    epoch:184, train acc:0.8866666666666667, test acc:0.712
    epoch:185, train acc:0.9, test acc:0.7141
    epoch:186, train acc:0.8933333333333333, test acc:0.7172
    epoch:187, train acc:0.9, test acc:0.7211
    epoch:188, train acc:0.8933333333333333, test acc:0.7167
    epoch:189, train acc:0.89, test acc:0.7153
    epoch:190, train acc:0.8933333333333333, test acc:0.7127
    epoch:191, train acc:0.89, test acc:0.7138
    epoch:192, train acc:0.8933333333333333, test acc:0.7116
    epoch:193, train acc:0.89, test acc:0.7171
    epoch:194, train acc:0.8833333333333333, test acc:0.7056
    epoch:195, train acc:0.9, test acc:0.7137
    epoch:196, train acc:0.89, test acc:0.7165
    epoch:197, train acc:0.8966666666666666, test acc:0.7135
    epoch:198, train acc:0.89, test acc:0.7092
    epoch:199, train acc:0.8966666666666666, test acc:0.715
    epoch:200, train acc:0.8933333333333333, test acc:0.7082
    


    
![basic_deep_learning_CH6_19_1](https://user-images.githubusercontent.com/55619678/126061200-4294d253-290f-48c3-a350-d6dae7f4cc0a.png)
    

가중치 감쇠는 간단하게 구현할 수 있고 어느 정도 지나친 학습을 억제할 수 있습니다.    
신경망 모델이 복잡해지면 가중치 감쇠만으로는 대응하기 어려워 진다. 이때 드롭아웃을 적용합니다.   
**드롭아웃**은 뉴런을 임의로 삭제하면서 학습하는 방법입니다. 

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train = x_train[:300]
t_train = t_train[:300]

use_dropout = True
dropout_ration = 0.2

network = MultiLayerNetExtend(input_size = 784, hidden_size_list=[100,100,100,100,100,100],
                             output_size = 10, use_dropout=use_dropout, dropout_ration = dropout_ration)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                 epochs = 301, mini_batch_size = 100,
                 optimizer='sgd', optimizer_param = {'lr':0.01}, verbose=True)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
```

    train loss:2.2929957576356657
    === epoch:1, train acc:0.08333333333333333, test acc:0.079 ===
    train loss:2.2837685984357354
    train loss:2.2930126881750157
    train loss:2.304420442715956
    === epoch:2, train acc:0.08333333333333333, test acc:0.0815 ===
    train loss:2.2972070423516864
    train loss:2.292692350656348
    train loss:2.289788687239025
    === epoch:3, train acc:0.08666666666666667, test acc:0.0831 ===
    train loss:2.2943163811095784
    train loss:2.30592390094486
    train loss:2.287645211640996
    === epoch:4, train acc:0.08666666666666667, test acc:0.0844 ===
    train loss:2.2974155903310614
    train loss:2.2913899536114286
    train loss:2.2831860330280853
    === epoch:5, train acc:0.08666666666666667, test acc:0.0849 ===
    train loss:2.2849571320555055
    train loss:2.2955828645541425
    train loss:2.2877990126633274
    === epoch:6, train acc:0.09, test acc:0.0851 ===
    train loss:2.277384888353862
    train loss:2.2939798146094375
    train loss:2.290608395919614
    === epoch:7, train acc:0.09333333333333334, test acc:0.0872 ===
    train loss:2.301508684050851
    train loss:2.287540624435578
    train loss:2.293234170531827
    === epoch:8, train acc:0.09333333333333334, test acc:0.0873 ===
    train loss:2.2809457958924733
    train loss:2.2857480707422573
    train loss:2.2843574167309315
    === epoch:9, train acc:0.09333333333333334, test acc:0.0896 ===
    train loss:2.2924552566394425
    train loss:2.281592509075946
    train loss:2.281896718235245
    === epoch:10, train acc:0.09666666666666666, test acc:0.0926 ===
    train loss:2.286038127202362
    train loss:2.28220877617054
    train loss:2.28514756108398
    === epoch:11, train acc:0.09666666666666666, test acc:0.0922 ===
    train loss:2.280971832160121
    train loss:2.280417513022796
    train loss:2.2944939273007634
    === epoch:12, train acc:0.10333333333333333, test acc:0.0956 ===
    train loss:2.28039972241009
    train loss:2.2876226139592957
    train loss:2.2667715773916366
    === epoch:13, train acc:0.10333333333333333, test acc:0.098 ===
    train loss:2.286032153185473
    train loss:2.287334578768909
    train loss:2.280195111738972
    === epoch:14, train acc:0.10333333333333333, test acc:0.099 ===
    train loss:2.275692200656738
    train loss:2.2694467910276033
    train loss:2.2698392097710847
    === epoch:15, train acc:0.09666666666666666, test acc:0.1027 ===
    train loss:2.2828142757460856
    train loss:2.289096629148217
    train loss:2.2753004720834324
    === epoch:16, train acc:0.11, test acc:0.1068 ===
    train loss:2.279023474480668
    train loss:2.2864562595907225
    train loss:2.2740941341774663
    === epoch:17, train acc:0.13, test acc:0.1092 ===
    train loss:2.2793004216213233
    train loss:2.275688074965826
    train loss:2.281043150124167
    === epoch:18, train acc:0.13666666666666666, test acc:0.11 ===
    train loss:2.276550832384799
    train loss:2.2792794023479113
    train loss:2.2717196556742305
    === epoch:19, train acc:0.13666666666666666, test acc:0.1131 ===
    train loss:2.283417984207431
    train loss:2.2742967002935224
    train loss:2.267581774381722
    === epoch:20, train acc:0.14666666666666667, test acc:0.1172 ===
    train loss:2.27533022437017
    train loss:2.2713509518383783
    train loss:2.2838140383610606
    === epoch:21, train acc:0.15333333333333332, test acc:0.1222 ===
    train loss:2.2742387353031024
    train loss:2.2662437321698614
    train loss:2.2650937808879403
    === epoch:22, train acc:0.14666666666666667, test acc:0.1261 ===
    train loss:2.2717508738715653
    train loss:2.272283072566036
    train loss:2.2733591916592544
    === epoch:23, train acc:0.15, test acc:0.1275 ===
    train loss:2.2625335854660795
    train loss:2.285188359156058
    train loss:2.2861999206994827
    === epoch:24, train acc:0.15333333333333332, test acc:0.134 ===
    train loss:2.27124801683009
    train loss:2.266625955000538
    train loss:2.2693733990015734
    === epoch:25, train acc:0.16333333333333333, test acc:0.1421 ===
    train loss:2.274845509903196
    train loss:2.2698140248250485
    train loss:2.278730389392477
    === epoch:26, train acc:0.18, test acc:0.1503 ===
    train loss:2.281997254865894
    train loss:2.271710303632892
    train loss:2.268307845116296
    === epoch:27, train acc:0.17333333333333334, test acc:0.153 ===
    train loss:2.2717879984641436
    train loss:2.264883154822119
    train loss:2.2755083768787845
    === epoch:28, train acc:0.17333333333333334, test acc:0.1598 ===
    train loss:2.274267225440908
    train loss:2.271648281685839
    train loss:2.2601082060588165
    === epoch:29, train acc:0.18333333333333332, test acc:0.1663 ===
    train loss:2.2721636974713544
    train loss:2.2752029691067572
    train loss:2.2593215234565376
    === epoch:30, train acc:0.18666666666666668, test acc:0.1706 ===
    train loss:2.2659990823111924
    train loss:2.270510779814654
    train loss:2.2733756151311626
    === epoch:31, train acc:0.19, test acc:0.1716 ===
    train loss:2.252876526612879
    train loss:2.2701518086894072
    train loss:2.2598359316506738
    === epoch:32, train acc:0.2, test acc:0.1743 ===
    train loss:2.2720709248456337
    train loss:2.263035925508859
    train loss:2.257557151998559
    === epoch:33, train acc:0.2, test acc:0.1784 ===
    train loss:2.269322171002923
    train loss:2.2706868356284016
    train loss:2.265850546227316
    === epoch:34, train acc:0.21333333333333335, test acc:0.1819 ===
    train loss:2.2536929317244265
    train loss:2.265151538506043
    train loss:2.257228195607976
    === epoch:35, train acc:0.21333333333333335, test acc:0.1862 ===
    train loss:2.267345770072515
    train loss:2.260617174343351
    train loss:2.2626857043795727
    === epoch:36, train acc:0.21, test acc:0.1921 ===
    train loss:2.2590404948863387
    train loss:2.260005513856984
    train loss:2.2703822551695776
    === epoch:37, train acc:0.22, test acc:0.1984 ===
    train loss:2.25698319609665
    train loss:2.2598611589415545
    train loss:2.2507520282953135
    === epoch:38, train acc:0.23, test acc:0.2023 ===
    train loss:2.2544940461851226
    train loss:2.251211258112219
    train loss:2.2637202875168767
    === epoch:39, train acc:0.24, test acc:0.2092 ===
    train loss:2.261791831042171
    train loss:2.2538007428308844
    train loss:2.264865339696933
    === epoch:40, train acc:0.25666666666666665, test acc:0.2123 ===
    train loss:2.264388447218963
    train loss:2.271714393263389
    train loss:2.2549377739115353
    === epoch:41, train acc:0.26, test acc:0.2161 ===
    train loss:2.2535686585619676
    train loss:2.2507551255567106
    train loss:2.2525398008334943
    === epoch:42, train acc:0.2733333333333333, test acc:0.2225 ===
    train loss:2.2466768722430377
    train loss:2.2524499498615045
    train loss:2.253362006096909
    === epoch:43, train acc:0.2966666666666667, test acc:0.221 ===
    train loss:2.259007412140792
    train loss:2.239214501231375
    train loss:2.254247704076447
    === epoch:44, train acc:0.29, test acc:0.2281 ===
    train loss:2.2552347497269936
    train loss:2.2495148703174808
    train loss:2.2578288904121946
    === epoch:45, train acc:0.30666666666666664, test acc:0.2321 ===
    train loss:2.2415301754330086
    train loss:2.2540251616231775
    train loss:2.2401385286369573
    === epoch:46, train acc:0.30666666666666664, test acc:0.2363 ===
    train loss:2.237378007969846
    train loss:2.2577443176842418
    train loss:2.25641322009945
    === epoch:47, train acc:0.30333333333333334, test acc:0.2393 ===
    train loss:2.244902642370966
    train loss:2.2417674618125947
    train loss:2.257071540391864
    === epoch:48, train acc:0.31333333333333335, test acc:0.2412 ===
    train loss:2.2603902082832246
    train loss:2.228679312749188
    train loss:2.246904834317734
    === epoch:49, train acc:0.31666666666666665, test acc:0.2444 ===
    train loss:2.2311059270892444
    train loss:2.2416088506336407
    train loss:2.2536605938814116
    === epoch:50, train acc:0.30666666666666664, test acc:0.2437 ===
    train loss:2.2587854872121658
    train loss:2.2519883532412965
    train loss:2.2553155242459306
    === epoch:51, train acc:0.30666666666666664, test acc:0.2448 ===
    train loss:2.2527420293777842
    train loss:2.248161493767635
    train loss:2.220618037125972
    === epoch:52, train acc:0.30666666666666664, test acc:0.2496 ===
    train loss:2.2423192043649434
    train loss:2.237057688302358
    train loss:2.2514758199433866
    === epoch:53, train acc:0.31333333333333335, test acc:0.25 ===
    train loss:2.232265441006199
    train loss:2.2512074910395476
    train loss:2.240110749256724
    === epoch:54, train acc:0.31, test acc:0.2513 ===
    train loss:2.230149098447879
    train loss:2.24733173831508
    train loss:2.233342990544255
    === epoch:55, train acc:0.30666666666666664, test acc:0.254 ===
    train loss:2.2481219700432424
    train loss:2.2415332080157464
    train loss:2.2376025868285403
    === epoch:56, train acc:0.31, test acc:0.255 ===
    train loss:2.235067341834691
    train loss:2.240925612945467
    train loss:2.2394596581909885
    === epoch:57, train acc:0.32, test acc:0.258 ===
    train loss:2.25045539533116
    train loss:2.259849453168476
    train loss:2.230548332701498
    === epoch:58, train acc:0.3233333333333333, test acc:0.2623 ===
    train loss:2.239276292676427
    train loss:2.2332266509231467
    train loss:2.2582382575308144
    === epoch:59, train acc:0.33, test acc:0.2666 ===
    train loss:2.243553376571965
    train loss:2.257242512589079
    train loss:2.236210069181446
    === epoch:60, train acc:0.33666666666666667, test acc:0.2693 ===
    train loss:2.24168578443338
    train loss:2.2288068677951935
    train loss:2.2306394632334534
    === epoch:61, train acc:0.34, test acc:0.2724 ===
    train loss:2.2358857745699185
    train loss:2.231319776091986
    train loss:2.221589424148029
    === epoch:62, train acc:0.33, test acc:0.27 ===
    train loss:2.231354429410774
    train loss:2.2386370457724127
    train loss:2.228294277442363
    === epoch:63, train acc:0.33, test acc:0.2673 ===
    train loss:2.2429327644028927
    train loss:2.228018963009078
    train loss:2.2285958269621737
    === epoch:64, train acc:0.33666666666666667, test acc:0.2709 ===
    train loss:2.2444275668630023
    train loss:2.2171481917439175
    train loss:2.2352795752935215
    === epoch:65, train acc:0.34, test acc:0.273 ===
    train loss:2.2076713037891618
    train loss:2.220468218841763
    train loss:2.225830748149271
    === epoch:66, train acc:0.3433333333333333, test acc:0.2751 ===
    train loss:2.213627001969976
    train loss:2.221421373547001
    train loss:2.2302722516453697
    === epoch:67, train acc:0.34, test acc:0.2761 ===
    train loss:2.2282100237751536
    train loss:2.235698719687542
    train loss:2.2448621681974017
    === epoch:68, train acc:0.3466666666666667, test acc:0.2782 ===
    train loss:2.2200268441350945
    train loss:2.206483521282781
    train loss:2.2342887198768278
    === epoch:69, train acc:0.35, test acc:0.2752 ===
    train loss:2.2093724029387847
    train loss:2.238115413870365
    train loss:2.227538889735948
    === epoch:70, train acc:0.3566666666666667, test acc:0.2818 ===
    train loss:2.254028088299552
    train loss:2.2372342514797277
    train loss:2.2329636801230537
    === epoch:71, train acc:0.36333333333333334, test acc:0.2834 ===
    train loss:2.20366000404085
    train loss:2.2116684575556147
    train loss:2.2321568900049855
    === epoch:72, train acc:0.37, test acc:0.2889 ===
    train loss:2.2190309917097464
    train loss:2.2222221387139394
    train loss:2.211660989212612
    === epoch:73, train acc:0.37, test acc:0.2888 ===
    train loss:2.20412743624223
    train loss:2.1996249437417306
    train loss:2.211870461094953
    === epoch:74, train acc:0.37333333333333335, test acc:0.2879 ===
    train loss:2.223743791069751
    train loss:2.2243572341799323
    train loss:2.204618401979234
    === epoch:75, train acc:0.36666666666666664, test acc:0.2851 ===
    train loss:2.224534903795112
    train loss:2.2040566803404693
    train loss:2.218732341132964
    === epoch:76, train acc:0.37, test acc:0.2905 ===
    train loss:2.221632888229754
    train loss:2.2070956770125743
    train loss:2.2338436540891284
    === epoch:77, train acc:0.37, test acc:0.2917 ===
    train loss:2.200005163627647
    train loss:2.221392451485109
    train loss:2.2087528555563183
    === epoch:78, train acc:0.36666666666666664, test acc:0.2925 ===
    train loss:2.2029687097386508
    train loss:2.2367942143191404
    train loss:2.2139622946991486
    === epoch:79, train acc:0.37333333333333335, test acc:0.2956 ===
    train loss:2.1879296800381005
    train loss:2.2277672611005515
    train loss:2.202822783936937
    === epoch:80, train acc:0.38333333333333336, test acc:0.3028 ===
    train loss:2.214710133883997
    train loss:2.1841328782961416
    train loss:2.2285958569844557
    === epoch:81, train acc:0.39, test acc:0.3116 ===
    train loss:2.218228186077227
    train loss:2.1796976058597393
    train loss:2.2268900924431225
    === epoch:82, train acc:0.3933333333333333, test acc:0.3153 ===
    train loss:2.2238109448117527
    train loss:2.196397841118269
    train loss:2.2045867273767494
    === epoch:83, train acc:0.4033333333333333, test acc:0.32 ===
    train loss:2.2017952038586417
    train loss:2.186495706979112
    train loss:2.1983910422394293
    === epoch:84, train acc:0.4066666666666667, test acc:0.3201 ===
    train loss:2.1944536996696224
    train loss:2.211991904111244
    train loss:2.204754176352616
    === epoch:85, train acc:0.4066666666666667, test acc:0.3257 ===
    train loss:2.211539967054745
    train loss:2.1739261515270774
    train loss:2.192373094389509
    === epoch:86, train acc:0.41, test acc:0.325 ===
    train loss:2.212652129439957
    train loss:2.1972919269350353
    train loss:2.1884115064602305
    === epoch:87, train acc:0.42, test acc:0.3282 ===
    train loss:2.199286025250797
    train loss:2.2074733695940516
    train loss:2.2122856379638334
    === epoch:88, train acc:0.4166666666666667, test acc:0.3288 ===
    train loss:2.179477316940183
    train loss:2.172939612230164
    train loss:2.186145566637715
    === epoch:89, train acc:0.41333333333333333, test acc:0.3263 ===
    train loss:2.196478206073391
    train loss:2.206544567140901
    train loss:2.171032726867087
    === epoch:90, train acc:0.4166666666666667, test acc:0.331 ===
    train loss:2.167719724711111
    train loss:2.162773825178933
    train loss:2.1880032185714304
    === epoch:91, train acc:0.42, test acc:0.3292 ===
    train loss:2.190810609901234
    train loss:2.1784248051203856
    train loss:2.173215166776346
    === epoch:92, train acc:0.42333333333333334, test acc:0.3308 ===
    train loss:2.1861299647487593
    train loss:2.174576993048985
    train loss:2.1920567158217206
    === epoch:93, train acc:0.42, test acc:0.3335 ===
    train loss:2.1937780427860063
    train loss:2.170562047949698
    train loss:2.2031062575395044
    === epoch:94, train acc:0.42333333333333334, test acc:0.3347 ===
    train loss:2.169510969067764
    train loss:2.174805422346829
    train loss:2.152585116553217
    === epoch:95, train acc:0.4266666666666667, test acc:0.3336 ===
    train loss:2.2233406895386483
    train loss:2.176522632198752
    train loss:2.1633496954459166
    === epoch:96, train acc:0.42333333333333334, test acc:0.3346 ===
    train loss:2.1892158186715336
    train loss:2.169541263004046
    train loss:2.2022894657335383
    === epoch:97, train acc:0.42333333333333334, test acc:0.3386 ===
    train loss:2.198851685163181
    train loss:2.1961372919467523
    train loss:2.1998366406477667
    === epoch:98, train acc:0.4266666666666667, test acc:0.3391 ===
    train loss:2.163555752280676
    train loss:2.181595705285912
    train loss:2.1838220604786467
    === epoch:99, train acc:0.42333333333333334, test acc:0.3387 ===
    train loss:2.173110653934143
    train loss:2.1621869892594763
    train loss:2.1626647033239035
    === epoch:100, train acc:0.42333333333333334, test acc:0.3403 ===
    train loss:2.193601407712737
    train loss:2.176907950328446
    train loss:2.142378996823213
    === epoch:101, train acc:0.42333333333333334, test acc:0.3404 ===
    train loss:2.187578177604164
    train loss:2.16869082689007
    train loss:2.1888871772386014
    === epoch:102, train acc:0.4266666666666667, test acc:0.342 ===
    train loss:2.1558530072739432
    train loss:2.159501724678086
    train loss:2.1677933934987315
    === epoch:103, train acc:0.42333333333333334, test acc:0.342 ===
    train loss:2.1620831653695576
    train loss:2.1456128769319376
    train loss:2.187269408927831
    === epoch:104, train acc:0.42, test acc:0.3391 ===
    train loss:2.1653144459044493
    train loss:2.1377479383690847
    train loss:2.1445757789031
    === epoch:105, train acc:0.4266666666666667, test acc:0.3404 ===
    train loss:2.1826983057219485
    train loss:2.127339792323527
    train loss:2.1295413540303936
    === epoch:106, train acc:0.43, test acc:0.3414 ===
    train loss:2.1455178511558577
    train loss:2.1662826536678423
    train loss:2.130318505231242
    === epoch:107, train acc:0.42333333333333334, test acc:0.3405 ===
    train loss:2.1391601898391053
    train loss:2.1358415304431557
    train loss:2.18607155904055
    === epoch:108, train acc:0.4066666666666667, test acc:0.3339 ===
    train loss:2.161735874605261
    train loss:2.116029678139466
    train loss:2.1653177955818506
    === epoch:109, train acc:0.41333333333333333, test acc:0.3356 ===
    train loss:2.1375814257396337
    train loss:2.170045971087456
    train loss:2.1127539420909183
    === epoch:110, train acc:0.41333333333333333, test acc:0.3353 ===
    train loss:2.1406468143756694
    train loss:2.1337467353058175
    train loss:2.170387708228169
    === epoch:111, train acc:0.4166666666666667, test acc:0.3332 ===
    train loss:2.1093811856892426
    train loss:2.1628173467688487
    train loss:2.1222603837043397
    === epoch:112, train acc:0.42, test acc:0.3385 ===
    train loss:2.14251235905206
    train loss:2.0793095215308486
    train loss:2.091608274323569
    === epoch:113, train acc:0.42, test acc:0.3359 ===
    train loss:2.1010561693588325
    train loss:2.137480167320845
    train loss:2.123112513794363
    === epoch:114, train acc:0.41333333333333333, test acc:0.3348 ===
    train loss:2.1653838636006517
    train loss:2.108763021429545
    train loss:2.130580550982975
    === epoch:115, train acc:0.42, test acc:0.3361 ===
    train loss:2.116623010600293
    train loss:2.0551742805367863
    train loss:2.0842912971887255
    === epoch:116, train acc:0.41, test acc:0.3337 ===
    train loss:2.102982741364117
    train loss:2.1293835151837146
    train loss:2.085450548950088
    === epoch:117, train acc:0.4166666666666667, test acc:0.3328 ===
    train loss:2.0988532060430596
    train loss:2.136919720776102
    train loss:2.141961440525049
    === epoch:118, train acc:0.4066666666666667, test acc:0.3326 ===
    train loss:2.0835967226981036
    train loss:2.1237193316974534
    train loss:2.06308914296241
    === epoch:119, train acc:0.41, test acc:0.3321 ===
    train loss:2.057720528015062
    train loss:2.1260302128686877
    train loss:2.109491922101884
    === epoch:120, train acc:0.4066666666666667, test acc:0.3314 ===
    train loss:2.088894941690834
    train loss:2.1423567547535125
    train loss:2.1517959876954684
    === epoch:121, train acc:0.4066666666666667, test acc:0.3333 ===
    train loss:2.1025265804728517
    train loss:2.0953719159557753
    train loss:2.0732372687419045
    === epoch:122, train acc:0.4066666666666667, test acc:0.3338 ===
    train loss:2.122802188986911
    train loss:2.1153311544566984
    train loss:2.114177869013966
    === epoch:123, train acc:0.4166666666666667, test acc:0.3381 ===
    train loss:2.100929907991983
    train loss:2.1236787571964046
    train loss:2.0973329362729802
    === epoch:124, train acc:0.42, test acc:0.3368 ===
    train loss:2.0848885634867282
    train loss:2.1373738313492976
    train loss:2.0514362054370654
    === epoch:125, train acc:0.42, test acc:0.3358 ===
    train loss:2.0696424888824696
    train loss:2.0799408852020034
    train loss:2.166689627510419
    === epoch:126, train acc:0.4266666666666667, test acc:0.3404 ===
    train loss:2.041579344594268
    train loss:2.098403808247085
    train loss:2.0749985196603
    === epoch:127, train acc:0.43, test acc:0.3451 ===
    train loss:2.0822658123404545
    train loss:2.0874872111888942
    train loss:2.070117273531785
    === epoch:128, train acc:0.42333333333333334, test acc:0.3416 ===
    train loss:2.059356309111261
    train loss:2.077723486316351
    train loss:2.067789878179804
    === epoch:129, train acc:0.4266666666666667, test acc:0.3433 ===
    train loss:2.0805561991700445
    train loss:2.0675096444590007
    train loss:2.068410072604564
    === epoch:130, train acc:0.42, test acc:0.3436 ===
    train loss:2.101025016445323
    train loss:2.078259180171681
    train loss:2.1021740828661204
    === epoch:131, train acc:0.43, test acc:0.3461 ===
    train loss:2.1207896186468065
    train loss:2.1133742614395783
    train loss:2.0806022975545413
    === epoch:132, train acc:0.43, test acc:0.3473 ===
    train loss:2.125523179362704
    train loss:2.085519209183788
    train loss:2.0591999001313015
    === epoch:133, train acc:0.45, test acc:0.3559 ===
    train loss:2.0640369530812914
    train loss:2.112910437160053
    train loss:2.0581315188537674
    === epoch:134, train acc:0.4533333333333333, test acc:0.3567 ===
    train loss:2.0490334192093247
    train loss:2.0444476030053518
    train loss:2.108633970074986
    === epoch:135, train acc:0.45666666666666667, test acc:0.36 ===
    train loss:2.0724873906939485
    train loss:2.0911698459369417
    train loss:2.0109911530633355
    === epoch:136, train acc:0.45666666666666667, test acc:0.3614 ===
    train loss:2.0660828159728513
    train loss:1.9857942146633427
    train loss:2.0496854052186597
    === epoch:137, train acc:0.44333333333333336, test acc:0.3601 ===
    train loss:2.0139345796734984
    train loss:2.013498556204626
    train loss:2.0871302785221015
    === epoch:138, train acc:0.4533333333333333, test acc:0.3654 ===
    train loss:2.049277120680762
    train loss:1.9822823111447097
    train loss:2.007524339892834
    === epoch:139, train acc:0.45666666666666667, test acc:0.3658 ===
    train loss:2.0252378443629144
    train loss:2.0098565733790665
    train loss:2.040173732337065
    === epoch:140, train acc:0.44333333333333336, test acc:0.361 ===
    train loss:1.9729952259529258
    train loss:1.9760085593415764
    train loss:2.0128133744424117
    === epoch:141, train acc:0.43666666666666665, test acc:0.3561 ===
    train loss:2.0332543410941204
    train loss:2.0022306949196205
    train loss:1.9509534369349613
    === epoch:142, train acc:0.43333333333333335, test acc:0.3557 ===
    train loss:2.0813258114312485
    train loss:2.0291631534848777
    train loss:2.015482691618166
    === epoch:143, train acc:0.44333333333333336, test acc:0.3634 ===
    train loss:2.000410829208293
    train loss:1.9918493069838414
    train loss:2.041300130431799
    === epoch:144, train acc:0.44333333333333336, test acc:0.3624 ===
    train loss:2.0105931565918507
    train loss:2.049933684495818
    train loss:2.024186121442044
    === epoch:145, train acc:0.4533333333333333, test acc:0.3645 ===
    train loss:2.0362974548291306
    train loss:2.056507240815238
    train loss:1.952146202586902
    === epoch:146, train acc:0.45, test acc:0.3676 ===
    train loss:2.0201128785331455
    train loss:2.0374496095836108
    train loss:2.040881021246539
    === epoch:147, train acc:0.46, test acc:0.37 ===
    train loss:2.0393499485859206
    train loss:2.014956164362309
    train loss:2.0336408180004013
    === epoch:148, train acc:0.4633333333333333, test acc:0.3732 ===
    train loss:2.0013237794809076
    train loss:2.0352016600725844
    train loss:1.9852840409930002
    === epoch:149, train acc:0.4666666666666667, test acc:0.3721 ===
    train loss:2.0485896910790498
    train loss:1.9755268093681984
    train loss:1.9364229850694759
    === epoch:150, train acc:0.45666666666666667, test acc:0.3744 ===
    train loss:2.0387054168094156
    train loss:1.9264890045238008
    train loss:1.9866668847604834
    === epoch:151, train acc:0.46, test acc:0.3769 ===
    train loss:2.0754081501533634
    train loss:1.9949412480635167
    train loss:1.9233887321457066
    === epoch:152, train acc:0.4666666666666667, test acc:0.3818 ===
    train loss:1.994137449194418
    train loss:2.0108591829837597
    train loss:2.0379329620082824
    === epoch:153, train acc:0.4766666666666667, test acc:0.388 ===
    train loss:1.922462131424403
    train loss:2.0059186344093116
    train loss:2.0370217256034233
    === epoch:154, train acc:0.47333333333333333, test acc:0.3883 ===
    train loss:1.939200794532535
    train loss:1.9555964174549458
    train loss:1.999441088139203
    === epoch:155, train acc:0.47333333333333333, test acc:0.3855 ===
    train loss:1.984390416354196
    train loss:2.0349740919366437
    train loss:2.0399736027504263
    === epoch:156, train acc:0.48, test acc:0.3901 ===
    train loss:1.9760928631668246
    train loss:1.9019398285773752
    train loss:1.951321876933129
    === epoch:157, train acc:0.4866666666666667, test acc:0.3924 ===
    train loss:1.8841051059514973
    train loss:1.9160395967327692
    train loss:1.967204670380876
    === epoch:158, train acc:0.4866666666666667, test acc:0.395 ===
    train loss:1.8674670751267686
    train loss:2.007244010189038
    train loss:1.9644183704726645
    === epoch:159, train acc:0.49, test acc:0.3988 ===
    train loss:1.9503983245315697
    train loss:1.970101959575764
    train loss:1.954381888140506
    === epoch:160, train acc:0.4866666666666667, test acc:0.3979 ===
    train loss:1.9665904396502787
    train loss:1.9160129115421354
    train loss:1.8952691144506888
    === epoch:161, train acc:0.48333333333333334, test acc:0.3964 ===
    train loss:1.8420676512342766
    train loss:1.8325741412218708
    train loss:1.8486870365921346
    === epoch:162, train acc:0.4766666666666667, test acc:0.3897 ===
    train loss:1.9014842009200352
    train loss:1.9835292843494843
    train loss:1.9270187566256873
    === epoch:163, train acc:0.48333333333333334, test acc:0.3897 ===
    train loss:1.9151805760265979
    train loss:1.9606657731543238
    train loss:1.9492210039712279
    === epoch:164, train acc:0.49, test acc:0.3907 ===
    train loss:1.8868944661493796
    train loss:1.9075844317193158
    train loss:1.9338411616390891
    === epoch:165, train acc:0.4866666666666667, test acc:0.3934 ===
    train loss:1.8871140492157996
    train loss:1.9808979117558858
    train loss:1.8861569721278544
    === epoch:166, train acc:0.4866666666666667, test acc:0.3896 ===
    train loss:1.971025957837818
    train loss:1.8557567023166666
    train loss:1.9393605812553514
    === epoch:167, train acc:0.4866666666666667, test acc:0.3925 ===
    train loss:1.8530654738468477
    train loss:1.8418480017275942
    train loss:1.9679899901895754
    === epoch:168, train acc:0.48333333333333334, test acc:0.3888 ===
    train loss:1.8825021232552601
    train loss:1.9411777244913795
    train loss:1.9284018221284116
    === epoch:169, train acc:0.48, test acc:0.3893 ===
    train loss:1.9040982703635743
    train loss:1.9508005088512979
    train loss:1.8327528670072701
    === epoch:170, train acc:0.48333333333333334, test acc:0.3877 ===
    train loss:1.9394763827646082
    train loss:1.8300599062001677
    train loss:1.9340628087540765
    === epoch:171, train acc:0.48, test acc:0.3905 ===
    train loss:1.9079158113689192
    train loss:1.9897202346236107
    train loss:1.8458518121928487
    === epoch:172, train acc:0.49333333333333335, test acc:0.3918 ===
    train loss:1.8751610312100215
    train loss:1.852138941638069
    train loss:1.8370020966174838
    === epoch:173, train acc:0.48333333333333334, test acc:0.3885 ===
    train loss:1.8594887451460085
    train loss:1.8933632138564764
    train loss:1.918185116765616
    === epoch:174, train acc:0.49333333333333335, test acc:0.3963 ===
    train loss:1.8293136239989107
    train loss:1.902044320405341
    train loss:1.8158790534777127
    === epoch:175, train acc:0.5033333333333333, test acc:0.3961 ===
    train loss:1.8919433277146651
    train loss:1.7728472017634085
    train loss:1.9124583968042628
    === epoch:176, train acc:0.51, test acc:0.3993 ===
    train loss:1.908719281317965
    train loss:1.96101301455109
    train loss:1.8542429862443763
    === epoch:177, train acc:0.5033333333333333, test acc:0.4012 ===
    train loss:1.7727528135855086
    train loss:1.8346954034109086
    train loss:1.843572080470418
    === epoch:178, train acc:0.5066666666666667, test acc:0.3986 ===
    train loss:1.8216339639153165
    train loss:1.817059761188972
    train loss:1.753166417098862
    === epoch:179, train acc:0.51, test acc:0.3993 ===
    train loss:1.8085119543528345
    train loss:1.8572669471870134
    train loss:1.8438749300700163
    === epoch:180, train acc:0.5066666666666667, test acc:0.3982 ===
    train loss:1.7530310611002613
    train loss:1.8152800852782611
    train loss:1.795737619491512
    === epoch:181, train acc:0.5033333333333333, test acc:0.3971 ===
    train loss:1.9032182441755143
    train loss:1.8343357763078172
    train loss:1.8368875028517846
    === epoch:182, train acc:0.5, test acc:0.3985 ===
    train loss:1.782430111279262
    train loss:1.9223707108402084
    train loss:1.8383613156016585
    === epoch:183, train acc:0.5, test acc:0.3978 ===
    train loss:1.8011578916975444
    train loss:1.8884966529767613
    train loss:1.7418763160869744
    === epoch:184, train acc:0.5, test acc:0.4001 ===
    train loss:1.8642876296400601
    train loss:1.608341697234036
    train loss:1.8727304708465469
    === epoch:185, train acc:0.5066666666666667, test acc:0.4033 ===
    train loss:1.831393293091454
    train loss:1.7791575471263008
    train loss:1.7730344326891643
    === epoch:186, train acc:0.5166666666666667, test acc:0.4069 ===
    train loss:1.8043698519219589
    train loss:1.7471308184864685
    train loss:1.850255894809509
    === epoch:187, train acc:0.5233333333333333, test acc:0.4088 ===
    train loss:1.7332856418522358
    train loss:1.7167390169873185
    train loss:1.828932666620966
    === epoch:188, train acc:0.5233333333333333, test acc:0.4105 ===
    train loss:1.7491200535778015
    train loss:1.817333699284984
    train loss:1.7625992030753386
    === epoch:189, train acc:0.5266666666666666, test acc:0.4125 ===
    train loss:1.7804865561525478
    train loss:1.6662940867760236
    train loss:1.7229086981758215
    === epoch:190, train acc:0.5233333333333333, test acc:0.4106 ===
    train loss:1.7891448547973525
    train loss:1.753051471152777
    train loss:1.7418391859731934
    === epoch:191, train acc:0.52, test acc:0.4114 ===
    train loss:1.751730511449405
    train loss:1.6804497759906911
    train loss:1.7026312348532706
    === epoch:192, train acc:0.5233333333333333, test acc:0.4119 ===
    train loss:1.6986452174995472
    train loss:1.7817885893943748
    train loss:1.757316799202671
    === epoch:193, train acc:0.52, test acc:0.4126 ===
    train loss:1.80454846643304
    train loss:1.771773236018617
    train loss:1.7580838360132782
    === epoch:194, train acc:0.5233333333333333, test acc:0.4139 ===
    train loss:1.8138052802666187
    train loss:1.6650725885984101
    train loss:1.912100995100094
    === epoch:195, train acc:0.5233333333333333, test acc:0.4189 ===
    train loss:1.7041665981497855
    train loss:1.6882809289481182
    train loss:1.7229947376663455
    === epoch:196, train acc:0.5233333333333333, test acc:0.4192 ===
    train loss:1.75223329000004
    train loss:1.7396634506573374
    train loss:1.7681843303893354
    === epoch:197, train acc:0.5266666666666666, test acc:0.4218 ===
    train loss:1.699146157404079
    train loss:1.6973586033191872
    train loss:1.8316381322037643
    === epoch:198, train acc:0.5233333333333333, test acc:0.4243 ===
    train loss:1.783418852289197
    train loss:1.7559961556940828
    train loss:1.7256699502816495
    === epoch:199, train acc:0.5233333333333333, test acc:0.4244 ===
    train loss:1.7109426544616608
    train loss:1.7284615288054272
    train loss:1.7895478418487967
    === epoch:200, train acc:0.5266666666666666, test acc:0.4284 ===
    train loss:1.7034584714509666
    train loss:1.768721137872804
    train loss:1.6630469621234005
    === epoch:201, train acc:0.5266666666666666, test acc:0.432 ===
    train loss:1.6310553126746132
    train loss:1.706937016282864
    train loss:1.71138561264831
    === epoch:202, train acc:0.52, test acc:0.431 ===
    train loss:1.6713398468077114
    train loss:1.7666928876264367
    train loss:1.729768497583584
    === epoch:203, train acc:0.52, test acc:0.429 ===
    train loss:1.6182172001154438
    train loss:1.6819784316804973
    train loss:1.7016904441836822
    === epoch:204, train acc:0.52, test acc:0.4298 ===
    train loss:1.7299522984731632
    train loss:1.709613774916573
    train loss:1.7424486440910028
    === epoch:205, train acc:0.5266666666666666, test acc:0.4336 ===
    train loss:1.634994593355056
    train loss:1.6823657213912335
    train loss:1.7497284874235868
    === epoch:206, train acc:0.5233333333333333, test acc:0.4328 ===
    train loss:1.7455476636050147
    train loss:1.7319437549638144
    train loss:1.6240722902100826
    === epoch:207, train acc:0.5233333333333333, test acc:0.4328 ===
    train loss:1.7120200699850958
    train loss:1.7506056771368284
    train loss:1.7042215279788258
    === epoch:208, train acc:0.5233333333333333, test acc:0.4322 ===
    train loss:1.659701837738464
    train loss:1.7529847878582367
    train loss:1.6596184694389542
    === epoch:209, train acc:0.52, test acc:0.43 ===
    train loss:1.5970295564341785
    train loss:1.8103284623690643
    train loss:1.6932242069997094
    === epoch:210, train acc:0.52, test acc:0.4337 ===
    train loss:1.6618472684533452
    train loss:1.6143936719757763
    train loss:1.628914317309148
    === epoch:211, train acc:0.52, test acc:0.4315 ===
    train loss:1.653199491953448
    train loss:1.657010695737054
    train loss:1.5885360583358716
    === epoch:212, train acc:0.5166666666666667, test acc:0.4309 ===
    train loss:1.6440841072402848
    train loss:1.7416986871519142
    train loss:1.6411124429112303
    === epoch:213, train acc:0.52, test acc:0.4326 ===
    train loss:1.60526012112698
    train loss:1.6334642209677779
    train loss:1.6152374181036697
    === epoch:214, train acc:0.5133333333333333, test acc:0.4346 ===
    train loss:1.7801844666118847
    train loss:1.5931785343633889
    train loss:1.711546154470973
    === epoch:215, train acc:0.5266666666666666, test acc:0.438 ===
    train loss:1.6087052924952627
    train loss:1.6384680913748786
    train loss:1.6383008597489919
    === epoch:216, train acc:0.53, test acc:0.4406 ===
    train loss:1.6839727152715829
    train loss:1.6623346907578724
    train loss:1.7145457459152578
    === epoch:217, train acc:0.5366666666666666, test acc:0.441 ===
    train loss:1.5965550968293585
    train loss:1.4868319280873101
    train loss:1.6898677154263708
    === epoch:218, train acc:0.5366666666666666, test acc:0.4397 ===
    train loss:1.674700597496244
    train loss:1.5329854181763651
    train loss:1.5607024790456165
    === epoch:219, train acc:0.5366666666666666, test acc:0.4407 ===
    train loss:1.560903752439243
    train loss:1.581124940846695
    train loss:1.6810000417198991
    === epoch:220, train acc:0.5433333333333333, test acc:0.446 ===
    train loss:1.6716091946948892
    train loss:1.6444055143061096
    train loss:1.48310456393844
    === epoch:221, train acc:0.5433333333333333, test acc:0.4531 ===
    train loss:1.6860101860113017
    train loss:1.6252651240393647
    train loss:1.5140755844830347
    === epoch:222, train acc:0.5466666666666666, test acc:0.4552 ===
    train loss:1.6543708101241352
    train loss:1.588620662704295
    train loss:1.6696616672023137
    === epoch:223, train acc:0.5433333333333333, test acc:0.455 ===
    train loss:1.7192710707992702
    train loss:1.4922717828519856
    train loss:1.6028223740077459
    === epoch:224, train acc:0.54, test acc:0.4492 ===
    train loss:1.642921254864626
    train loss:1.6154246887001358
    train loss:1.6079988355824832
    === epoch:225, train acc:0.54, test acc:0.4452 ===
    train loss:1.567807071119478
    train loss:1.5459511628182354
    train loss:1.5870827210168512
    === epoch:226, train acc:0.5433333333333333, test acc:0.4503 ===
    train loss:1.6695587074178966
    train loss:1.5513969337685416
    train loss:1.6506609908256908
    === epoch:227, train acc:0.5433333333333333, test acc:0.4474 ===
    train loss:1.582963611298511
    train loss:1.5217498766811863
    train loss:1.5635139264669384
    === epoch:228, train acc:0.5433333333333333, test acc:0.449 ===
    train loss:1.588932486367469
    train loss:1.5598747646597595
    train loss:1.5451232311763858
    === epoch:229, train acc:0.5433333333333333, test acc:0.4601 ===
    train loss:1.5670987341774196
    train loss:1.4986456880933938
    train loss:1.4796672199286376
    === epoch:230, train acc:0.55, test acc:0.4596 ===
    train loss:1.5404383070862533
    train loss:1.503907671577164
    train loss:1.5812989017919457
    === epoch:231, train acc:0.5433333333333333, test acc:0.4559 ===
    train loss:1.6420437308821545
    train loss:1.5765007340189
    train loss:1.4961775618988682
    === epoch:232, train acc:0.5566666666666666, test acc:0.4651 ===
    train loss:1.5004899704852201
    train loss:1.597599379494632
    train loss:1.501159516407104
    === epoch:233, train acc:0.5433333333333333, test acc:0.4641 ===
    train loss:1.5771797751195686
    train loss:1.4615371081478403
    train loss:1.5299786767469117
    === epoch:234, train acc:0.5433333333333333, test acc:0.4639 ===
    train loss:1.6178052570711214
    train loss:1.6851777434096078
    train loss:1.5843641364304064
    === epoch:235, train acc:0.5533333333333333, test acc:0.4745 ===
    train loss:1.517382173772393
    train loss:1.5090939049014147
    train loss:1.519380884065349
    === epoch:236, train acc:0.5533333333333333, test acc:0.473 ===
    train loss:1.4956310161694908
    train loss:1.6215235397697594
    train loss:1.5229683241583591
    === epoch:237, train acc:0.5633333333333334, test acc:0.4779 ===
    train loss:1.510767795738377
    train loss:1.5328729357215138
    train loss:1.5066382649398136
    === epoch:238, train acc:0.5866666666666667, test acc:0.4944 ===
    train loss:1.5227222193707004
    train loss:1.5200632263893668
    train loss:1.4768243019322194
    === epoch:239, train acc:0.59, test acc:0.4958 ===
    train loss:1.3589126419653956
    train loss:1.4899805078512542
    train loss:1.4797010593649182
    === epoch:240, train acc:0.59, test acc:0.4997 ===
    train loss:1.6176183520439273
    train loss:1.607167773624996
    train loss:1.6273186200910077
    === epoch:241, train acc:0.5733333333333334, test acc:0.4926 ===
    train loss:1.4123020347736048
    train loss:1.5747868218980607
    train loss:1.3840799356696414
    === epoch:242, train acc:0.5666666666666667, test acc:0.4889 ===
    train loss:1.5722876900955793
    train loss:1.4764732872109985
    train loss:1.4594074438428124
    === epoch:243, train acc:0.5733333333333334, test acc:0.4936 ===
    train loss:1.5641393335771305
    train loss:1.5616879647014934
    train loss:1.442707874115732
    === epoch:244, train acc:0.5766666666666667, test acc:0.4983 ===
    train loss:1.494846842954998
    train loss:1.4338985614394328
    train loss:1.4587741818373405
    === epoch:245, train acc:0.5933333333333334, test acc:0.5065 ===
    train loss:1.4778041791572922
    train loss:1.5160958255838997
    train loss:1.471498649613902
    === epoch:246, train acc:0.5733333333333334, test acc:0.492 ===
    train loss:1.4521784525438095
    train loss:1.4409425864364238
    train loss:1.502138539680063
    === epoch:247, train acc:0.5533333333333333, test acc:0.4872 ===
    train loss:1.5203008086248468
    train loss:1.5600892887756779
    train loss:1.4890257223262802
    === epoch:248, train acc:0.5466666666666666, test acc:0.4827 ===
    train loss:1.3171807373563758
    train loss:1.437092152414802
    train loss:1.5008613954910603
    === epoch:249, train acc:0.5566666666666666, test acc:0.4862 ===
    train loss:1.6041862345870361
    train loss:1.472500394988364
    train loss:1.508357527788722
    === epoch:250, train acc:0.5566666666666666, test acc:0.4889 ===
    train loss:1.3605066954685472
    train loss:1.4438987809685029
    train loss:1.4202120919071985
    === epoch:251, train acc:0.5833333333333334, test acc:0.4997 ===
    train loss:1.4614101947828666
    train loss:1.5704824626036764
    train loss:1.466594518255114
    === epoch:252, train acc:0.5933333333333334, test acc:0.5084 ===
    train loss:1.4829348380542058
    train loss:1.3809409222516689
    train loss:1.4221229905074737
    === epoch:253, train acc:0.5833333333333334, test acc:0.502 ===
    train loss:1.4573191961122185
    train loss:1.472200334700569
    train loss:1.5226185316624117
    === epoch:254, train acc:0.5866666666666667, test acc:0.4998 ===
    train loss:1.469209425068447
    train loss:1.3155396605473584
    train loss:1.5108525887594566
    === epoch:255, train acc:0.6, test acc:0.5145 ===
    train loss:1.454929695089181
    train loss:1.4568471436424932
    train loss:1.539753795773884
    === epoch:256, train acc:0.6, test acc:0.5157 ===
    train loss:1.4931978926883946
    train loss:1.3530662080516822
    train loss:1.3081754528804534
    === epoch:257, train acc:0.6033333333333334, test acc:0.5127 ===
    train loss:1.448125248685991
    train loss:1.4651627151578714
    train loss:1.44427774083217
    === epoch:258, train acc:0.6, test acc:0.5163 ===
    train loss:1.4028551716474484
    train loss:1.5760142044301872
    train loss:1.4680489889566013
    === epoch:259, train acc:0.6033333333333334, test acc:0.5168 ===
    train loss:1.2486792094456785
    train loss:1.4808528208404939
    train loss:1.4131902427136314
    === epoch:260, train acc:0.5966666666666667, test acc:0.515 ===
    train loss:1.451068985320243
    train loss:1.5264088602399408
    train loss:1.3875641990163299
    === epoch:261, train acc:0.57, test acc:0.5016 ===
    train loss:1.453631652047058
    train loss:1.4533300718436832
    train loss:1.2393902363823721
    === epoch:262, train acc:0.5533333333333333, test acc:0.4929 ===
    train loss:1.549295492758078
    train loss:1.3871917640985632
    train loss:1.4997923847252919
    === epoch:263, train acc:0.5766666666666667, test acc:0.5057 ===
    train loss:1.496375848176225
    train loss:1.4348949093690733
    train loss:1.3448018114449996
    === epoch:264, train acc:0.5433333333333333, test acc:0.4928 ===
    train loss:1.3188345182255663
    train loss:1.3335956294122693
    train loss:1.4263082438899486
    === epoch:265, train acc:0.56, test acc:0.4986 ===
    train loss:1.4422696084182847
    train loss:1.2688648026898226
    train loss:1.375391625390147
    === epoch:266, train acc:0.5866666666666667, test acc:0.5095 ===
    train loss:1.3102191585205674
    train loss:1.4350380120775899
    train loss:1.4644739118382235
    === epoch:267, train acc:0.56, test acc:0.4998 ===
    train loss:1.2277711616445612
    train loss:1.4839221863247207
    train loss:1.3724701328280837
    === epoch:268, train acc:0.58, test acc:0.5091 ===
    train loss:1.3961442914794653
    train loss:1.3472059271418046
    train loss:1.3911339194878352
    === epoch:269, train acc:0.5966666666666667, test acc:0.5224 ===
    train loss:1.4694084956426465
    train loss:1.372291523892594
    train loss:1.3916565810729364
    === epoch:270, train acc:0.59, test acc:0.5104 ===
    train loss:1.4255359306040198
    train loss:1.325843587889629
    train loss:1.422243590000616
    === epoch:271, train acc:0.6033333333333334, test acc:0.5154 ===
    train loss:1.440914678679604
    train loss:1.2922808905295902
    train loss:1.3295124013462907
    === epoch:272, train acc:0.6066666666666667, test acc:0.5193 ===
    train loss:1.3847931092935475
    train loss:1.3731185705492936
    train loss:1.3609743728334809
    === epoch:273, train acc:0.5766666666666667, test acc:0.5013 ===
    train loss:1.2241903621516246
    train loss:1.4200182426118235
    train loss:1.4712529794406324
    === epoch:274, train acc:0.58, test acc:0.508 ===
    train loss:1.2378663216521424
    train loss:1.378908006070413
    train loss:1.274974278801556
    === epoch:275, train acc:0.5733333333333334, test acc:0.5006 ===
    train loss:1.3486846880343173
    train loss:1.3848586063377704
    train loss:1.3574688195915237
    === epoch:276, train acc:0.5666666666666667, test acc:0.5032 ===
    train loss:1.3626874160470521
    train loss:1.4266076267946315
    train loss:1.2969430787301548
    === epoch:277, train acc:0.59, test acc:0.511 ===
    train loss:1.356240773872018
    train loss:1.209496680610666
    train loss:1.3486708471174171
    === epoch:278, train acc:0.58, test acc:0.5081 ===
    train loss:1.3707302690787184
    train loss:1.3496663382397052
    train loss:1.4247609114031732
    === epoch:279, train acc:0.5966666666666667, test acc:0.513 ===
    train loss:1.2623495568127583
    train loss:1.3115889321101037
    train loss:1.3800527975196968
    === epoch:280, train acc:0.62, test acc:0.5232 ===
    train loss:1.2453287653036091
    train loss:1.3627193859091034
    train loss:1.2828596628530946
    === epoch:281, train acc:0.6233333333333333, test acc:0.5244 ===
    train loss:1.4709919098596311
    train loss:1.2470134905072874
    train loss:1.2494327125856466
    === epoch:282, train acc:0.6133333333333333, test acc:0.5201 ===
    train loss:1.352302644862836
    train loss:1.227172146726415
    train loss:1.3676217350296047
    === epoch:283, train acc:0.6366666666666667, test acc:0.53 ===
    train loss:1.3025333703203652
    train loss:1.3193075258883065
    train loss:1.29912983975204
    === epoch:284, train acc:0.6166666666666667, test acc:0.5231 ===
    train loss:1.3272606623604246
    train loss:1.2684054374263147
    train loss:1.4006524140242962
    === epoch:285, train acc:0.59, test acc:0.5108 ===
    train loss:1.3581568906480987
    train loss:1.3547943464625123
    train loss:1.2988005295109653
    === epoch:286, train acc:0.5833333333333334, test acc:0.5127 ===
    train loss:1.2932753525020726
    train loss:1.3256340813786627
    train loss:1.3417547975860336
    === epoch:287, train acc:0.58, test acc:0.5086 ===
    train loss:1.1963029013744841
    train loss:1.3561043672650706
    train loss:1.1853701485091845
    === epoch:288, train acc:0.5833333333333334, test acc:0.5079 ===
    train loss:1.1947501219910461
    train loss:1.178625771310784
    train loss:1.2843615212850037
    === epoch:289, train acc:0.6133333333333333, test acc:0.5246 ===
    train loss:1.3166066978990474
    train loss:1.2642159217956896
    train loss:1.298558035390393
    === epoch:290, train acc:0.62, test acc:0.5273 ===
    train loss:1.2506788764901309
    train loss:1.3579625594885838
    train loss:1.206094153313327
    === epoch:291, train acc:0.6166666666666667, test acc:0.5257 ===
    train loss:1.1363552096550025
    train loss:1.2677551316554974
    train loss:1.1788809972051117
    === epoch:292, train acc:0.6233333333333333, test acc:0.5276 ===
    train loss:1.0908732244763448
    train loss:1.1530632227286912
    train loss:1.1585116217954794
    === epoch:293, train acc:0.6333333333333333, test acc:0.5265 ===
    train loss:1.3253703031837025
    train loss:1.2123514407833376
    train loss:1.2857026652111672
    === epoch:294, train acc:0.64, test acc:0.5304 ===
    train loss:1.296365921958642
    train loss:1.1614110748487634
    train loss:1.3300154911495392
    === epoch:295, train acc:0.6333333333333333, test acc:0.5255 ===
    train loss:1.216838807284506
    train loss:1.2947152225741378
    train loss:1.2627914730272685
    === epoch:296, train acc:0.6433333333333333, test acc:0.5381 ===
    train loss:1.2819797755458888
    train loss:1.2038469673066297
    train loss:1.2092754932254235
    === epoch:297, train acc:0.64, test acc:0.5344 ===
    train loss:1.3509812644632442
    train loss:1.2435262156941302
    train loss:1.207038664375838
    === epoch:298, train acc:0.6433333333333333, test acc:0.5295 ===
    train loss:1.2365611432881458
    train loss:1.2580169964069718
    train loss:1.1681064661216707
    === epoch:299, train acc:0.63, test acc:0.53 ===
    train loss:1.3511259882525934
    train loss:1.0860321956444445
    train loss:1.156554184142209
    === epoch:300, train acc:0.63, test acc:0.5292 ===
    train loss:1.1761415495980274
    train loss:1.1757602329918766
    train loss:1.2487897658237508
    === epoch:301, train acc:0.6233333333333333, test acc:0.5271 ===
    train loss:1.206794955981638
    train loss:1.172476486266099
    =============== Final Test Accuracy ===============
    test acc:0.5292
    


    
![basic_deep_learning_CH6_21_1](https://user-images.githubusercontent.com/55619678/126061201-72fdea3d-fd98-47a5-be83-0efe329ce222.png)
    


#### 6.5 적절한 하이퍼파라미터 값 찾기    
검증 데이터는 하이퍼파라미터를 조정할 때 사용하는 데이터셋이다.    
훈련데이터 셋에서 20%정도를 검증 데이터로 먼저 분리한다.    



```python
from common.util import shuffle_dataset
(x_train, t_train),(x_test, t_test) = load_mnist()

x_train, t_train = shuffle_dataset(x_train, t_train)
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]
```

하이퍼파라미터 최적화   
- 0단계   
하이퍼파라미터 값의 범위를 설정합니다.   
- 1단계   
설정된 범위에서 하이퍼파라미터의 값을 무작위로 추출합니다.   
- 2단계    
1단계에서 샘플링한 하이퍼파라미터 값을 사용하여 학습하고, 검증 데이터로 정확도를 평가합니다.(단, 에포크는 작게 설정합니다.)   
- 3단계   
1단계와 2단계를 특정횟수 반복하여, 그 정확도를 보고 하이퍼 파라미터의 범위를 좁게 합니다.   



```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir) 
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train = x_train[:500]
t_train = t_train[:500]

validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()

```

    val acc:0.07 | lr:5.758906675745511e-05, weight decay:7.447633464357572e-05
    val acc:0.1 | lr:4.072347976567432e-05, weight decay:5.280762118996739e-08
    val acc:0.15 | lr:0.00030110596021014455, weight decay:4.310290637001971e-06
    val acc:0.12 | lr:2.3057053256308817e-05, weight decay:2.4504286389780386e-08
    val acc:0.54 | lr:0.0033383905700179427, weight decay:2.148592615435151e-08
    val acc:0.07 | lr:9.95668879343498e-06, weight decay:4.55779115360553e-07
    val acc:0.15 | lr:0.000267475960405829, weight decay:4.1184619363802586e-05
    val acc:0.12 | lr:2.306142061253457e-05, weight decay:2.6679206756952307e-08
    val acc:0.77 | lr:0.008049516241707272, weight decay:3.14668290465784e-08
    val acc:0.07 | lr:3.4740640150021196e-06, weight decay:1.4533743311012176e-05
    val acc:0.08 | lr:1.0194411455560113e-06, weight decay:8.402055519375313e-05
    val acc:0.14 | lr:5.507607496202541e-06, weight decay:4.010780553556185e-07
    val acc:0.62 | lr:0.0041050407558751315, weight decay:9.715566910505246e-08
    val acc:0.31 | lr:0.0024267493393560513, weight decay:4.7348229352022755e-08
    val acc:0.43 | lr:0.003607738864073263, weight decay:1.425180823881133e-07
    val acc:0.09 | lr:1.828362739023399e-05, weight decay:1.1042878013460518e-07
    val acc:0.07 | lr:9.15679646928125e-06, weight decay:6.445045999239191e-08
    val acc:0.05 | lr:1.2407188761244974e-05, weight decay:8.642947074398384e-07
    val acc:0.12 | lr:2.674551091273158e-06, weight decay:3.151303518925205e-05
    val acc:0.06 | lr:2.4098712821063808e-05, weight decay:4.081011574153761e-06
    val acc:0.13 | lr:1.7079578071040656e-06, weight decay:5.681799475199219e-07
    val acc:0.2 | lr:2.1348488640590538e-05, weight decay:3.046296929960554e-07
    val acc:0.08 | lr:8.636819722116476e-05, weight decay:2.0547220210677445e-08
    val acc:0.1 | lr:0.00012270923061411993, weight decay:2.2884791308839253e-08
    val acc:0.13 | lr:6.297598197141983e-06, weight decay:3.2816022888954602e-06
    val acc:0.13 | lr:0.00041724185602916444, weight decay:1.3591575562225996e-08
    val acc:0.39 | lr:0.0029981480990989725, weight decay:6.79375305407049e-05
    val acc:0.07 | lr:2.2919257106239146e-05, weight decay:6.795692037115702e-07
    val acc:0.07 | lr:4.38770381379706e-06, weight decay:2.744546752697543e-07
    val acc:0.38 | lr:0.003170941364458359, weight decay:1.787173959545218e-05
    val acc:0.18 | lr:0.00057918982565942, weight decay:5.949334697106133e-06
    val acc:0.08 | lr:1.5379902018166434e-06, weight decay:4.927842660412149e-06
    val acc:0.14 | lr:0.0009692714758038179, weight decay:2.2115208115835122e-06
    val acc:0.15 | lr:0.0007834960910876347, weight decay:4.288693041330753e-08
    val acc:0.1 | lr:6.664757051819684e-06, weight decay:8.697329196078444e-07
    val acc:0.75 | lr:0.00947050418051443, weight decay:3.3455802376768806e-07
    val acc:0.15 | lr:9.970081556510419e-05, weight decay:3.51587492396811e-07
    val acc:0.66 | lr:0.004431936966013703, weight decay:1.149766297332287e-06
    val acc:0.1 | lr:0.0003189995844251872, weight decay:1.0075725551722244e-06
    val acc:0.11 | lr:5.6192957348986435e-05, weight decay:3.446807260784576e-06
    val acc:0.1 | lr:2.1442549654926126e-05, weight decay:1.65147550911748e-07
    val acc:0.13 | lr:1.0148920086342475e-06, weight decay:5.813437784426107e-07
    val acc:0.04 | lr:1.6845165926146416e-06, weight decay:5.295783956566704e-05
    val acc:0.07 | lr:0.0002761569536410312, weight decay:1.252732403408627e-05
    val acc:0.1 | lr:7.084206849578939e-06, weight decay:3.036739262492248e-08
    val acc:0.2 | lr:0.001277761548583415, weight decay:1.1429376142578316e-07
    val acc:0.09 | lr:3.391200275271707e-05, weight decay:2.489879209839867e-08
    val acc:0.78 | lr:0.007838087488496253, weight decay:1.1693315279371039e-08
    val acc:0.1 | lr:0.00019547212804388517, weight decay:4.46284217903074e-06
    val acc:0.13 | lr:0.0007214508374075356, weight decay:3.2170127779444285e-06
    val acc:0.73 | lr:0.007949196664989167, weight decay:3.05754312783711e-08
    val acc:0.11 | lr:0.00041988421330712596, weight decay:5.284590759791823e-05
    val acc:0.12 | lr:0.000436047018795135, weight decay:6.49076411690711e-06
    val acc:0.11 | lr:6.857899646425793e-05, weight decay:4.2784953261039704e-07
    val acc:0.11 | lr:1.5028690496769064e-05, weight decay:1.1899753560923599e-08
    val acc:0.12 | lr:6.515638360654896e-05, weight decay:2.576526542141665e-05
    val acc:0.07 | lr:8.207740905324532e-06, weight decay:6.159148189928739e-08
    val acc:0.07 | lr:3.482020541041986e-06, weight decay:1.0694837442290575e-08
    val acc:0.09 | lr:8.842515496881717e-05, weight decay:6.021344816088884e-05
    val acc:0.13 | lr:6.995550766326027e-06, weight decay:2.4128029177526507e-08
    val acc:0.2 | lr:0.0008197217038923134, weight decay:7.938634029553711e-05
    val acc:0.79 | lr:0.008827156799707518, weight decay:1.3125163207936922e-06
    val acc:0.06 | lr:1.8413747571608863e-05, weight decay:4.851207563264252e-08
    val acc:0.47 | lr:0.003118798171451386, weight decay:1.5253994265678303e-08
    val acc:0.74 | lr:0.005861297829471083, weight decay:1.421239708880994e-07
    val acc:0.76 | lr:0.006533997525796829, weight decay:1.7464294435588557e-07
    val acc:0.18 | lr:0.0001308161761337307, weight decay:7.633443517663294e-08
    val acc:0.1 | lr:0.0005295995845464467, weight decay:4.3963788834365205e-05
    val acc:0.12 | lr:0.00013267770506874893, weight decay:9.507101642802728e-07
    val acc:0.13 | lr:6.6505872420578934e-06, weight decay:3.2926670478569875e-07
    val acc:0.08 | lr:0.00015307923296719933, weight decay:3.5779480065645566e-05
    val acc:0.1 | lr:4.8882084590472394e-05, weight decay:8.226990947388022e-05
    val acc:0.4 | lr:0.0017733920676094546, weight decay:1.9169144268339686e-06
    val acc:0.08 | lr:1.6173232555096796e-06, weight decay:6.905182367162858e-05
    val acc:0.11 | lr:2.718334000940521e-05, weight decay:1.1772045446528541e-06
    val acc:0.05 | lr:8.320372480345382e-05, weight decay:1.7940287021421644e-06
    val acc:0.11 | lr:3.437475627588776e-05, weight decay:5.64663123137178e-06
    val acc:0.61 | lr:0.007339756165045262, weight decay:8.536027314924091e-08
    val acc:0.12 | lr:1.436240149526577e-06, weight decay:1.7705635104553974e-08
    val acc:0.24 | lr:0.0016216773775975712, weight decay:1.9869101710555566e-07
    val acc:0.52 | lr:0.0035319898426630835, weight decay:2.4397069277242184e-07
    val acc:0.05 | lr:1.4935629903749611e-06, weight decay:1.322975720565985e-07
    val acc:0.09 | lr:0.0001003830078678246, weight decay:9.216218385644984e-06
    val acc:0.09 | lr:2.513469129534134e-05, weight decay:1.2602649618627133e-05
    val acc:0.06 | lr:1.922800544097568e-06, weight decay:3.095471799899687e-08
    val acc:0.21 | lr:0.0012640876223623035, weight decay:3.3356606689870557e-06
    val acc:0.11 | lr:2.4740126795207066e-06, weight decay:5.640597834588352e-07
    val acc:0.16 | lr:0.0006626465090578259, weight decay:5.097901136271769e-07
    val acc:0.1 | lr:1.3742145565224613e-05, weight decay:1.9824398414069234e-08
    val acc:0.06 | lr:2.571540563724613e-06, weight decay:5.2207787092211425e-05
    val acc:0.06 | lr:3.960380784078805e-05, weight decay:4.520312756549795e-07
    val acc:0.09 | lr:2.4132610476664803e-06, weight decay:6.2205509286036474e-06
    val acc:0.12 | lr:1.8047038924296327e-05, weight decay:1.4852241980954865e-05
    val acc:0.76 | lr:0.006915614620458979, weight decay:3.58322181100234e-06
    val acc:0.72 | lr:0.008177620010087518, weight decay:2.854994608759924e-07
    val acc:0.11 | lr:2.24681324917202e-06, weight decay:1.957200615934284e-06
    val acc:0.27 | lr:0.0016159964158172588, weight decay:3.204312472559032e-08
    val acc:0.09 | lr:0.0001372116793583112, weight decay:8.895410397988153e-06
    val acc:0.03 | lr:4.571938911239016e-05, weight decay:4.328329205192541e-05
    val acc:0.04 | lr:0.00016943884151944538, weight decay:2.8869392058483434e-05
    =========== Hyper-Parameter Optimization Result ===========
    Best-1(val acc:0.79) | lr:0.008827156799707518, weight decay:1.3125163207936922e-06
    Best-2(val acc:0.78) | lr:0.007838087488496253, weight decay:1.1693315279371039e-08
    Best-3(val acc:0.77) | lr:0.008049516241707272, weight decay:3.14668290465784e-08
    Best-4(val acc:0.76) | lr:0.006533997525796829, weight decay:1.7464294435588557e-07
    Best-5(val acc:0.76) | lr:0.006915614620458979, weight decay:3.58322181100234e-06
    Best-6(val acc:0.75) | lr:0.00947050418051443, weight decay:3.3455802376768806e-07
    Best-7(val acc:0.74) | lr:0.005861297829471083, weight decay:1.421239708880994e-07
    Best-8(val acc:0.73) | lr:0.007949196664989167, weight decay:3.05754312783711e-08
    Best-9(val acc:0.72) | lr:0.008177620010087518, weight decay:2.854994608759924e-07
    Best-10(val acc:0.66) | lr:0.004431936966013703, weight decay:1.149766297332287e-06
    Best-11(val acc:0.62) | lr:0.0041050407558751315, weight decay:9.715566910505246e-08
    Best-12(val acc:0.61) | lr:0.007339756165045262, weight decay:8.536027314924091e-08
    Best-13(val acc:0.54) | lr:0.0033383905700179427, weight decay:2.148592615435151e-08
    Best-14(val acc:0.52) | lr:0.0035319898426630835, weight decay:2.4397069277242184e-07
    Best-15(val acc:0.47) | lr:0.003118798171451386, weight decay:1.5253994265678303e-08
    Best-16(val acc:0.43) | lr:0.003607738864073263, weight decay:1.425180823881133e-07
    Best-17(val acc:0.4) | lr:0.0017733920676094546, weight decay:1.9169144268339686e-06
    Best-18(val acc:0.39) | lr:0.0029981480990989725, weight decay:6.79375305407049e-05
    Best-19(val acc:0.38) | lr:0.003170941364458359, weight decay:1.787173959545218e-05
    Best-20(val acc:0.31) | lr:0.0024267493393560513, weight decay:4.7348229352022755e-08
    


    
![basic_deep_learning_CH6_25_1](https://user-images.githubusercontent.com/55619678/126061202-5f9651d6-ddec-4b25-886a-fec42d622a14.png)
    



```python

```
