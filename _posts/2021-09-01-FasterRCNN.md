---
title:  "[Paper Review] Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
excerpt: "Faster R-CNN Paper Review"

categories:
  - Paper_Review
tags:
  - [Blog, jekyll, Github, Git]

toc: true
toc_sticky: true
 
date: 2021-09-01
last_modified_at: 2021-09-01
---
# Faster R-CNN 논문 리뷰   
## 1. Introduction   
이전까지 Region Proposal하는 방법의 경우 주로 Selective Search 알고리즘에 의해서 진행되었습니다. 하지만 Selective Search의 경우 상당히 느리다는 단점이 있습니다. 이후에 EdgeBoxes논문에서 다른 region proposal을 소개했지만 여전히 시간적인 소모가 많았습니다.   
이 논문에서는 proposal을 제안하는데 있어서 RPN(Region Proposal Networks)를 통해서 convolution layer는 공유하는 새로운 알고리즘을 제시하였습니다. 이 RPN의 경우 convolution layer를 거쳐서 나온 feature map을 이용해서 동시에 regress region bounds와 objectness score를 계산합니다. RPN은 또한 fully connected network로 이루어져 있고 전체 네트워크를 한번에 학습시킬 수 있습니다.    
또한 효율적으로 region proposal을 예측하기 위해서 anchor라는 개념을 사용하였으며 이는 pyramid of regression reference의 형태입니다. 이 anchor를 사용함으로써 다양한 크기 또는 비율에 따른 bounding box를 예측할 수 있게 되었습니다.   
또한 이러한 RPN과 Faster R-CNN의 프레임워크는 다른 3d object detection, instance segmentation과 같은 방법에 사용할 수 있습니다.    
이렇게 논문에서 제시한 방법은 ILSVRC2015, COCO2015에서 1위를 차지하였습니다.   
## 3. Faster R-CNN   
![2021-09-01-8](https://user-images.githubusercontent.com/55619678/131686352-42823b89-2df2-425d-9b9c-8d96815c5b94.png)    
Faster R-CNN은 두개의 모듈로 구분해서 설명할 수 있습니다. 첫번째 모듈은  deep fully convolution network인 RPN, 두번째 모듈은 Fast R-CNN의 detector로 생각합니다.   
전체 네트워크는 위의 그림과 같습니다.   
### 3.1 Region Proposal Network   
RPN의 input은 임의의 크기의 feature map을 받습니다. 이후 output으로 bounding box의 4개의 좌표와 각 bounding box에 object가 있는지를 알아보는 objectness score를 반환합니다.   
RPN에서 region proposal을 만들어 내기 위해서 sliding window방식을 feature map에 사용합니다. 각 sliding window는 feature map을 낮은 차원으로 맵핑시켜줍니다. 이후 각 feature들이 두개의 독립적인 fully connected network인 box-regression layer, box-classification layer에 들어갑니다. 이렇게 fully-connected network를 사용함으로써 공간적인 정보를 유지할 수 있게 됩니다.   
- Anchors   

![2021-09-01-9](https://user-images.githubusercontent.com/55619678/131686357-c56a47b4-16bd-4b82-8aeb-381a9b53fb55.png)    
각 sliding-window를 적용하는 위치에서 동시에 다양한 region proposal을 예측합니다. 이때 k라는 하이퍼파라미터가 등장하는데 k는 각 location에서 나오는 region proposal의 수를 의미합니다. regression layer에서는 4(각 region proposal의 좌표)xk개의 output을 내놓고, classification layer에서는 2(물체의 존재 여부에 대한 score)xk개의 output을 내놓습니다.   
이 논문에서는 기본적으로 k=9를 사용하고 3개의 크기와 3개의 비율에 대해서 각각 region proposal을 만들어 냅니다.   
또한 이 논문에서의 접근법은 translation invariant하다는 특징이 있습니다. Translation invariant는 간단하게 input의 위치가 변하더라도 classification결과는 일치하는 특성입니다.   
이렇게 translation invariant한 특성을 가지는 이유는 anchor를 사용하기 때문입니다. 모든 이미지에 대해서 공통된 anchor를 사용해서 계산하므로 translation invaraint한 특성을 가질 뿐 아니라 model의 size도 줄여주는 효과를 가지고 있습니다. 예를 들면, Faster r-cnn의 output layer에서 (512x(4+2)x9)개의 파라미터수를 가지기 때문입니다. 이는 다른 방법과 비교했을때 적습니다.   
다른 다양한 scale을 다루기위한 방법들로는 image/feature pyramid, sliding window of multiple scale의 방법이 있는데 이들은 모든 경우의 수에 대해서 계산해야하므로 시간이 오래걸립니다. 하지만 이 논문에서 사용하는 pyramid of anchor의 방식은 훨씬 시간 소비를줄일 수 있습니다.   
- Loss Function   

![2021-09-01-10](https://user-images.githubusercontent.com/55619678/131686360-7659ef0d-bfb1-4052-ba5d-39b85abee600.png)   
- i : anchor의 index   
- $p_{i}$ : anchor i가 물체인지 아닌지를 예측한 결과
- $p_{i}^{*}$ : ground truth label   
- $t_{i}$ : 예측한 box의 좌표   
- $t_{i}^{*}$ : ground truth box coordinate   
- $\lambda$ : default는 10, 두 loss를 맞추기위한 가중치   

Faster R-CNN을 학습하기 위해서는 위와 같은 multi-task loss를 사용합니다.   
첫번째 loss의 경우 classification loss로 두개의 클래스에 대한 cross-entropy loss를 사용하며, 두번째 loss인 regression loss의 경우 smooth L1 loss를 사용합니다. 여기서 $p^{*}L_{reg}$를 사용하는 이유는 실제 존재한 positive anchor(background가 아닌 물체)에 대해서만 적용하기 위함입니다.   
![2021-09-01-12](https://user-images.githubusercontent.com/55619678/131686680-75c55b35-911c-42c8-9ce4-e16182b28906.png)    
- $x,x_{a},x^{*}$ : 예측한 box, anchor box, ground truth   

Bounding box regression을 사용하기 위해서 위의 4가지 좌표에 대한 계산을 이용합니다. 위의 식을 살펴보면 anchor box의 좌표를 기준으로 사용하기에 anchor box로 부터 ground truth로 regression을 한다고 생각할 수 있습니다.   
- Training RPNs   

RPN은 end-to-end의 방식으로 한번에 학습되어지며 가중치는 가우시안 분포를 사용해서 초기화 해줍니다. 또한 learning rate은 0.001로 시작해서 0.0001로 감소해서 학습을 진행하고, momentum은 0.9, weight decay는 0.0005를 사용했습니다.   
### 3.2 Sharing Features for RPN and Fast R-CNN   
Convolution layer를 공유하는 두개의 네트워크를 학습시키기 위해서 따로 독립적으로 학습시키기 보다 한번에 학습시키는 방법을 알아보겠습니다.   
1. Alternating training : 먼저 RPN을 학습시키고 proposal을 Fast R-CNN을 학습할때 이용하고 Fast R-CNN을 이용해 RPN을 초기화 해주는 이 과정을 반복합니다.   
2. Approximate joint training : 위에 보여준 RPN과 Fast R-CNN의 통합된 네트워크와 같이 하나로 묶어서 학습시킵니다. 이렇게 학습한 결과 실험을 통해서 첫번째 방법과 성능은 비슷하지만 훈련시간이 약 25~50%정도 감소하는 것을 알 수 있었습니다.   
3. Non-approximate joint training : 이 방법에 대해서는 논문에서 자세히 다루고 있지 않습니다.   

이 논문에서는 4-step alternating training을 사용하고 있으며 첫번째 step에서는 RPN을 ImageNet으로 pretrain된 모델을 가지고 초기화 하며 fine-tuning 해줍니다. 두번째 step에서는 Fast R-CNN을 RPN에서 생성한 region proposal을 이용해서 학습합니다. 세번째 단계에서는 모두 이용해서 나머지는 fix시킨채로 RPN을 fine tuning합니다.  
네번째 단계에서는 CNN과 RPN을 fix한채로 Fast R-CNN을 fine-tuning시켜줍니다.   

## 5. Conclusion    
![2021-09-01-11](https://user-images.githubusercontent.com/55619678/131686362-1ab52202-96cb-43e1-b47a-30b47ee73a1a.png)
이 논문에서는 RPN이라는 Network를 이용해서 시간을 단축시키고 더 좋은 성능을 이끌어 내었습니다. 
- - -    
이 논문은 처음으로 구현하려는 목표를 가지고 읽은 논문입니다. 따라서 더 자세하게 각각의 네트워크에 대해서 파악하며 읽으려 노력하였는데 region proposal을 만들어 내는 RPN구조의 아이디어를 통해서 시간을 많이 단축시키고 성능을 올렸다는 점을 인상깊게 읽었습니다. 하지만 여전히 고정된 크기의 anchor box를 이용해서 region proposal을 만들어내는 과정이 small object나 다양한 형태의 모든 object를 검출하기엔 힘들 수도 있겠다는 생각을 하였습니다. 