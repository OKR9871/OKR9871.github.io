---
title:  "[Paper Review] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
excerpt: "ResNet Paper Review"

categories:
  - Paper_Review
tags:
  - [Blog, jekyll, Github, Git]

toc: true
toc_sticky: true
 
date: 2021-08-01
last_modified_at: 2021-08-01
---
# Batch Normalization    
## Abstract    
깊은 층의 Neural Network를 학습시키는 것은 각 layer에 들어가는 input의 분포가 훈련중 변화기 때문에 어렵습니다.    
또한 낮은 learning rate, parameter의 초기화, non-linearity의 감소와 같이 여러 문제점이 존재하는데 이를 'internal covariate shift'라고 합니다.    
논문에서 제시하는 배치정규화는 더 높은 learning rate를 사용하게 해주고, 초기화에 덜 민감하고, 규제의 역할도 수행한다고 합니다.   
## 1. Introduction    
딥러닝은 여러 분야에서 발전해 왔습니다. 그중에서 SGD를 포함한 다양한 방법은 이러한 deep network의 효과적은 학습을 가능하게 해주었습니다.   
SGD는 parameter $\theta$를 loss값이 최소가 되도록 최적화 해주는 방법입니다. 식으로 적으면 아래와 같습니다.   
$\theta = argmin{1\over N}\sum\limits_{i=1}^{N}l(x_{i},\theta)$    
전체 N개의 training set이 있을때 SGD는 훈련중에는 mini-batch를 사용합니다. mini-batch에 대한 gradient계산은 다음과 같습니다.   
${1\over m}{\partial l(x_{i}, \theta)\over\partial\theta}$    
이렇게 mini-batch를 사용하는데는 여러 장점이 있습니다. 첫번째 장점은 loss함수의 gradient는 추정치이지만 batch의 사이즈에 따라서 더 높은 정확서을 보여줄 수 있습니다. 둘째로 병렬처리가 가능한 현재 기술로 m개의 batch로 나누어서 처리하는 것이 계산상의 효율을 높이기 때문입니다.   
이렇게 SGD를 사용하는 방법이 간단하면서도 효과적이지만 단점이 존재합니다.   
단점으로는 많은 hyper parameter의 조정을 요구합니다. 또한 훈련중 input값의 많은 변화로인해 layer의 paremeter들이 많은 영향을 받습니다. 이러한 문제를 이전에 'covariate shift'라고 부리기로 했습니다.    
'covariate shift'문제는 domain adaptation으로 해결하고자 합니다. 하지만 이 문제는 input layer뿐만아니라 훈련중 모든 과정에 거쳐서 나타나게 됩니다. 예를 들면 $l = F_{2}(F_{1}(u,\theta_{1}), \theta_{2})$를 network가 계산하고 $F_{1}, F_{2}$가 임의 변환함수라 할때, parameter $\theta_{1},\theta_{2}$를 $l$을 최소화 하도록 학습시켜주어야 합니다.   
이대 위 식을 두부분으로 분리해서 생각하면 $x=F_{1}(u,\theta_{1})$라 하고 이를 $F_{2}$의 입력으로 넣을 수 있습니다. 이때 입력으로 들어오는 값이 고정된 분포를 가지고 있다면 sub-network의 외부에서는 긍정적인 결과를 보여줍니다. 하지만 이렇게 통과한 값이 activation function을 거쳐가면서 값은 점차 작아지고, gradient또한 작아지는 모습을 보이면서 사라지게 되는 gradient vaninshing문제가 발생하기도 합니다. 이러한 문제는 네트워크의 층이 깊어질 수록 심해집니다.    
이런문제를 해결하기 위해 현재는 ReLU함수의 사용, parameter초기화 방법, 작은 learning rate의 사용으로 해결하고 있습니다. 하지만 각 layer의 input의 분포가 학습중에 안정된다면 위의 문제를 해결하는데 도움이 될것입니다.   
이 논문에서 제시하는 Batch Normalization(배치 정규화)는 'internal covariate shift'문제를 해결하는데 도움을 줍니다. 이를 위해서 input의 고정된 평균, 분산을 이용해 분포를 맞춰줍니다. 또한 gradient flow가 초기값에 예민한 문제를 해결하였고, 높은 learning rate의 사용을 가능하게 하였습니다. 또한 saturated 구간에 빠지는 것을 막아줍니다.   
## 2. Towards Reducing Internal Covariate Shift    
우리는 'internal covariate shift'를 학습중 parameter의 변화에 따라 활성값의 분포가 변화하는 문제로 정의하였습니다.   
각 layer의 input의 분포를 맞춰준다면 학습을 더 효율적이고 빠르게 할 수 있습니다. 이러한 방법은 오래전에 whitening이라는 방법을 사용하고 있었습니다.   
이 whitening이라는 방법은 평균을 0, 분산을 일정하게 맞춰주고 decorrelation하게 함으로써 정규화를 진행해주는 방법입니다. 하지만 이러한 변형이 최적화 과정중에 들어가게 된다면, gradient descent 과정에서 parameter를 업데이트하는데 gradient의 효과가 없어지게 됩니다. 예를 들면, bias b를 추가해서 정규화와 학습을 진행하는 경우 b에 대한 gradient값은 계산중에 사라지게 되어 학습이 올바르게 되지 않습니다. 또한 whitening의 경우 계산시에 covariance matrix의 계산을 진행하는데 이는 계산량이 많아져 비효율적인 문제를 발생시킵니다. 또한 역전파를 위한 covariance matrix의 역함수 게산도 많은 계산을 요구하기 때문에 비효율적입니다. 이러한 문제들 때문에 필자는 새로운 normalize를 위한 방법을 생각했습니다.   
## 3. Normalization via Mini-Batch Statistics     
이 논문에서는 input과 output의 feature에 대해 whitening을 적용하는것 대신에 평균을 0으로, 분산을 1로 정규화 과정을 진행하였습니다.   
$\hat{x}={x^{(k)}-E[x^{(k)}]\over\sqrt{Var[x^{(k)}]}}$   
이렇게 normalizing함으로써 feature에 변화가 일어날 수 있습니다. 
![2021-08-06-1](https://user-images.githubusercontent.com/55619678/128522608-669ba101-1f2b-41f9-a40d-ca38e3d05884.PNG)      
예를들면 위 그림과 같이 sigmoid 함수에 input으로 들어가면 비선형함수이지만 비선형성을 제한하는 문제가 발생할 수도 있습니다. 따라서 $\gamma^{k},\beta^{k}$와 같은 파라미터를 도입해서 해결하였습니다.   
$y^{(k)}=\gamma^{(k)}\hat{x}^{(k)}+\beta^{(k)}$    
이 parameter들은 학습중에 다른 parameter와 같이 학습됩니다.   
학습되어지기는 하지만 $\gamma^{k} = \sqrt{Var[x^{k}]},\beta^{k}=E[x^{k}]$로 설정하면 원래의 activation function을 복원할 수 있습니다.   
전체 training set을 사용하는것은 비효율적일 수 있습니다. 따라서 mini-batch를 사용하는데 평균과 분산을 계산할때 또한 각 mini-batch의 분산과 평균을 이용해서 정규화를 진행합니다.   
다음은 batch normalization을 진행하는 algorithm입니다.   
![2021-08-06-2](https://user-images.githubusercontent.com/55619678/128522610-1bf7fac6-6102-4615-a496-5bc0e931b6d0.png)
역전파 계산을 하귕해서 loss에 대한 gradient를 계산할때는 chain rule을 이용해서 계산해줍니다.   
![2021-08-06-3](https://user-images.githubusercontent.com/55619678/128524111-36e93d2f-343a-42a4-9e23-3b3052791242.png)
위의 식과 같이 기존의 역전파 계산과 같이 loss값을 필요한 parameter들로 편미분하여 gradient를 계산해줍니다.   
이제 위에서 설명한것을 모두 합하여 training과정을 알고리즘으로 나타낸 것을 살펴보겠습니다.   
![2021-08-06-4](https://user-images.githubusercontent.com/55619678/128524105-078a4715-2b30-4e3a-800c-d61479effeda.png)   
학습은 기존의 방식과 같이 mini-batch를 이용해서 순전파와 역전파단계를 거쳐서 학습을 진행하고 validation 단계에서는 모든 파라미터는 고정시킨채로 deterministic한 결과를 뽑아내야하기에 moving average를 이용해 training 과정에서의 미니 배치를 뽑으면서 sample mean, sample variance를 구해서 이를 고정으로 사용합니다.     
Batch normalization은 주로 non-linear activation fucntion바로 앞에 위치하며 이렇게 함으로써 출력값이 안정된 분포를 가지게 해줍니다.   
## 5. Conclusion    
이 논문에서 제시한 방법은 deep network의 학습을 월등히 빠르고 효율적으로 가능하게 합니다. 'internal covariate shift'문제에 기반해 학습이 잘 되지않는 문제를 해결하였습니다. 또한 오직 두개의 parameter만을 추가하여 계산의 복잡성을 높이지 않아 network의 성능을 그대로 유지하였으며 dropout을 사용하지 않고도 규제의 효과를 낼 수 있었습니다.   

이 논문을 읽으면서 Batch Normalization의 아이디어가 나온 원인을 알 수 있었으며 간단한 방법을 통해 학습의 효과를 높이 끌어올릴 수 있다는 점을 들어 현재까지도 많은 network에서 사용중인 이유를 이해할 수 있었습니다. 