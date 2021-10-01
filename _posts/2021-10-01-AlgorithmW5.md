---
title:  "Algorithm Week5"
excerpt: "인하대학교 김영호 교수님의 알고리즘 수업 week5 Review"

categories:
  - Algorithm
tags:
  - [Blog, jekyll, Github, Git]

toc: true
toc_sticky: true
 
date: 2021-10-01
last_modified_at: 2021-10-01
---   
# Algorithm Week5   
> 포스팅하는 내용은 모두 인하대학교 김영호 교수님의 강의 내용을 바탕으로 정리하였습니다.   

## 4. Sorting   
- Insertion Sort(삽입정렬)   

Insertion sort의 worst complexity를 분석해 보면 아래와 같습니다.   
$W(n)=\sum\limits_{i=1}^{n-1}i=n(n-1)/2\in\Theta(n^2)$    
최악의 수행시간의 경우 i개의 element가 있을 때 i만큼의 비교연산을 수행해야합니다.   
이제 Average Case에 대해서 분석해 보겠습니다.    
![2021-10-01-1](https://user-images.githubusercontent.com/55619678/135649451-cfdce674-127d-429c-8f11-adae9170d96b.png)    
Average Case의 경우 위의 그림과 같이 index에 따라서 비교 연산이 달라지게 됩니다. 따라서 index i에 대한 각각의 비교연산의 수를 파악해보면 총 i+1번의 case가 존재하고 각각은 1,2,3,...,i까지 비교연산을 하게 됩니다. 따라서 이를 식으로 표현하면   
${1\over(i+1)}\sum\limits_{j=1}^i j+{i\over{(i+1)}}={i/2}+1-{1\over{i+1}}$   
$A(n)=\sum\limits_{i=1}^{n-1}{i\over2}-1-{1\over(i+1)}\approx {n^2\over4}$   
이렇게 Insertion sort는 worst case에 대해서 $n(n-1)\over2$, average case에 대해서 $n(n-1)\over4$의 복잡도를 가지는데 이는 순차적인 접근(연산의 제약조건)만 되는 경우에 한해서 최적의 알고리즘이 될 수 있습니다.   
- Quick-Sort   

Quick sort의 경우 일반적으로 가장 빠르다고 알려진 알고리즘으로 divide and conquer전략과 randomization을 사용합니다. 이 방법은 3개의 step을 가지는데 이는 작은문제로 나눈다(divide), 각 sub-problem을 해결한다(conquer), 해결한 해를 합친다(combine)을 합니다.   
Quick sort를 간단하게 설명하면 pivot이라는 값을 기준으로 3개의 group으로 나누어서 최소한 한번에 pivot하나만큼은 정렬하면서 진행하는 알고리즘입니다. 여기서 3개의 group은 L(pivot보다 작은 수의 집합), E(pivot과 같은 수의 집합), G(pivot보다 큰 수의 집합)입니다.   
Divide Step인 Partition함수를 살펴보겠습니다.  
![2021-10-01-2](https://user-images.githubusercontent.com/55619678/135649424-556eff1e-df8e-4547-8768-7bf6043c6830.png)    
위의 알고리즘의 Worst Case에 대해서 분석해보면 minimum이나 maximum으로 pivot이 설정될 경우 가장 많은 비교연산을 수행하게 됩니다. 왜냐하면 아래 그림과 같이 n개의 element를 가지는 배열을 tree의 형태로 분석해볼수 있는데 이때 트리의 depth가 깊어질 수록 더 많은 비교연산을 하게 되고 이렇게 될 경우가 한쪽으로 치우친 트리의 형태를 가질때 이기 때문입니다.   
![2021-10-01-3](https://user-images.githubusercontent.com/55619678/135649431-bc22bec1-67ba-4056-9048-da85579c08fa.png)   
따라서 이때의 worst case를 분석해보면 $n+(n-1)+...+2+1$로써 $O(n^2)$에 수행하게 됩니다.   
이제는 Average case에 대해서 분석해보겠습니다.    
![2021-10-01-4](https://user-images.githubusercontent.com/55619678/135649433-c9ac4b04-783a-44dc-a956-34b00c107415.png)    
다음으로는 In-Place Quick-Sort에 대해서 살펴보겠습니다.   
In-Place알고리즘이란 입력으로 들어오는 size n보다 추가적으로 $O(1)$만큼의 space만을 더 사용하는 경우입니다.   
![2021-10-01-5](https://user-images.githubusercontent.com/55619678/135649434-198b1e43-2e99-49ac-b9d5-8c2e2638279d.png)    
위의 그림과 같은 알고리즘에 의해 실행되며 차이점은 L, G단 두개만으로 그룹을 분리한다는 점입니다. 이후 각 group에 대해서 L에 대해서는 처음부터, G에 대해서는 마지막 원소부터 시작해 pivot보다 L은 큰값이 나오면, G는 작은값이 나오면 두 값을 교환해줍니다. 이 과정을 반복하면서 두 index가 만나면 iteration을 중단해줍니다.   
이렇게 하게 되면 경계를 기준으로 해서 왼쪽은 모두 작고, 오른쪽은 모두 그룹으로 나뉘게 되면서 정렬을 수행합니다.    
- Merge-Sort    

이 Merge-Sort의 방법은 두개의 정렬된 sequence를 하나의 정렬된 Sequence인 C로 merge하면서 정렬을 수행하는 방법입니다.   
각각의 정렬된 A, B sequence의 값을 앞에서 부터 비교하면서 C에 넣어주면 정렬이 됩니다.   
![2021-10-01-6](https://user-images.githubusercontent.com/55619678/135649438-e6f6df26-f25e-4a6c-aeb8-005ab42f7354.png)    
![2021-10-01-7](https://user-images.githubusercontent.com/55619678/135649444-3f9876f9-b318-4c60-9e3a-2f463d906bf2.png)    
Worst Case에 대해서 분석해보면 최악의 경우 하나씩 비교하면서 모든 원소를 비교할때 입니다. 이때 마지막 비교의 경우 두개를 비교하면 하나는 자동으로 자리가 정해지므로 $W(n) = n-1\in\Theta(n)$이라 할 수 있습니다.   
이를 다르게 표현하면 $W(n)=W({n\over2})+W({n\over2})+W_{merge}(n)\in\Theta(nlogn)$으로 표현할 수 있습니다. 이는 input n을 재귀적으로 2개의 부분으로 나누어가면서 worst case에 대해서 분석하고 각각을 merge하는데 걸리는 time complexity를 계산한다고 생각할 수 있습니다.   
![2021-10-01-8](https://user-images.githubusercontent.com/55619678/135649449-12008d01-b50c-4d24-8015-e56baf494965.png)    
위의 그림으로 이해하면 더 이해하기 간단합니다. 결국 맨 아래의 divide된 원소는 하나만 존재하기에 정렬된 sequence로 볼 수 있고 이를 이용해서 merge하면서 정렬을 진행하는 방법입니다.   
Quick sort, Merge sort를 비교해보면 Quick sort는 partition하면서, merge sort는 combine하면서 정렬을 수행합니다. 이를 트리의 순회로 생각해본다면, quick sort는 preorder의 순서, merge sort는 postorder의 순서로 수행합니다.   
