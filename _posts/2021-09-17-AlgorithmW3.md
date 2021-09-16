---
title:  "Algorithm Week3"
excerpt: "인하대학교 김영호 교수님의 알고리즘 수업 week3 Review"

categories:
  - Algorithm
tags:
  - [Blog, jekyll, Github, Git]

toc: true
toc_sticky: true
 
date: 2021-09-17
last_modified_at: 2021-09-17
---   
# Algorithm Week3   
> 포스팅하는 내용은 모두 인하대학교 김영호 교수님의 강의 내용을 바탕으로 정리하였습니다.   

## 1. Analyzing Algorithms and Problems : Principles and Examples   

- Properties of $O(g), \Theta(g), \Omega(g)$   
Transitive(이행성) : 3단논법과 비슷하며, 만약 $f\in O(g), g\in O(h)$이면 $f\in O(h)$를 따른다. 이는 $O, \Theta, \Omega, \omega, w$모두 적용된다.   
Reflexive(반사성) : $f\in\Theta(f)$이다. 이는 $O, \Omega$도 만족한다.   
Symmetric(대칭성) : $f\in\Theta(g)$이면 $g\in\Theta(f)$이다. 오직 $\Theta$에 대해서만 만족한다.   
위의 3개의 property를 동시에 만족하는경우 equivalence relation이라고 한다.   
추가적으로 $f\in O(g)$와 $g\in\Omega(f)$는 동시에 서로를 만족한다. 또 $O(f+g)=O(max(f,g))$를 만족한다.   

$O(1)$의 경우 constant라고 하며 input size에 상관없이 일정한 수의 연산 or 공간을 사용한다.   
$f\in\Theta(n)$의 경우, f는 linear하다고 하고, $f\in\Theta(n^2)$는 quadratic, $f\in\Theta(n^3)$은 cubic이라고 한다.   
$logn\in o(n^\alpha)$를 만족하며 이는 $logn$의 증가율은 다항함수보다 더 느리고, 작다는 것을 의미합니다.   
$n^k \in o(c^n)$은 지수함수는 다항식보다 연산이 많이 필요하다는 것을 의미합니다.   

### Searching an Ordered Array    
이전 수업시간에서 배웠던 정렬되지 않은 배열에서의 탐색은 Worst case에 W(n), Average case에 $q[(n+1)/2]+(1-q)n$입니다.   
이러한 경우 문제의 복잡도를 분석했을때도 F(n) = n으로 최적의 알고리즘임을 알 수 있었습니다.   
하지만 정렬된 배열이 입력으로 주어진 경우도 같을지 생각해 보겠습니다.   
입력이 오름차순으로 정렬된 배열일때 주어진 K보다 값이 커질 경우 탐색을 중단하고 -1을 반환하는 방법에 대해서 생각해볼 수 있습니다.   
![2021-09-17-1](https://user-images.githubusercontent.com/55619678/133654577-d9f7b79b-d27c-4ece-bac3-f62258fb3c17.png)    
이때의 Worst Case에 대해서 분석해보면 $W(n)=n+1\approx n$입니다.   
Average Case를 분석해보면 아래의 그림과 같습니다.   
![2021-09-17-2](https://user-images.githubusercontent.com/55619678/133654584-c7dabc4b-9f4e-442e-a092-6deb7cf47387.png)    
성공할 경우의 Average case를 구하고, 실패할 경우의 Average를 구한다음 두 개의 확률의 합으로 생각해줍니다.   
다른 아이디어로는 Binary Search 알고리즘에 대해서 살펴보겠습니다.   
Binary Search의 경우 입력으로 array E, first, last값이 주어집니다.   
![2021-09-17-3](https://user-images.githubusercontent.com/55619678/133654586-4f6b2eb4-21f6-46a7-8efd-9d8ee845f625.png)    
Worst Case에 대해서 생각해보면 Basic Operation을 K와 Entry간 비교연산으로 선택하고 처음에는 n개 한번의 비교 연산으로 $1\over 2$로 줄기에 최악의 경우 원소의 개수가 1개일때 까지 줄어들면 됩니다.   
$n\times {1\over 2}\times...\times {1\over 2}=1$이 되는 경우로 생각할 수 있습니다.   
이때 비교연산의 수는 $n\times ({1\over 2})^k$로 생각할 수 있으며 k를 n을 이용해서 표현하면 $k=logn$이 됩니다.    
따라서 $W(n)=\lfloor\log n\rfloor + 1=\lceil log(n+1)\rceil\in\Theta(log n)$입니다.     
이제는 Average Case에 대해서 계산해보겠습니다.   
이를 위해서는 3가지 가정이 필요합니다.   
1. 발생하는 모든 case는 같은 확률로 발생한다. 
2. input size n은 $2^d-1$이다.    
3. n개의 entry는 중복되지 않는다.   

![2021-09-17-4](https://user-images.githubusercontent.com/55619678/133654588-8745e292-f046-41ed-8159-156238ab25e6.png)   
