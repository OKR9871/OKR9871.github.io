---
title:  "Algorithm Week2"
excerpt: "인하대학교 김영호 교수님의 알고리즘 수업 week2 Review"

categories:
  - Algorithm
tags:
  - [Blog, jekyll, Github, Git]

toc: true
toc_sticky: true
 
date: 2021-09-08
last_modified_at: 2021-09-08
---   
# Algorithm Week2   
> 포스팅하는 내용은 모두 인하대학교 김영호 교수님의 강의 내용을 바탕으로 정리하였습니다.   

## 1. Analyzing Algorithms and Problems : Principles and Examples   

### Analyzing Algorithms and Problems    
어떤 알고리즘과 문제를 분석하는 이유는 알고리즘마다 효율성이 다르기 때문입니다.   
이때문에 **어떤 알고리즘을 선택해야하는지 기준이 되기 위해서** 또는 **기존의 알고리즘을 개선하기 위해서** 우리는 알고리즘을 분석합니다.   
알고리즘을 분석하기 위한 지표들이 correctness, amount of work(time complexity), space complexity, optimality, simplicity가 있습니다.   
- Correctness   
알고리즘은 어떤 문제를 해결하는 단계적 절차, 방법입니다.   
따라서 Input데이터가 Output으로 변환하는데 사용한 step(절차)의 나열로 구성이 되어있습니다.   
이때 Correctness를 증명하는 방법으로는 loop invarianct(루프 불변성)이 있습니다. 이는 대부분의 알고리즘이 loop를 통해서 이루어지는데 이를 이용하는 방법으로 수학적 귀납법과 유사합니다.   
- Amount of Work Done(Time Complexity)   
알고리즘의 효율성을 이야기할때는 일의 양을 측정합니다.    
이론적으로 분석하기 위해서는 하드웨어, 프로그래밍 언어, 프로그래머의 능력, 구현적인 세부적 내용에 독립적으로 분석해야합니다.   
일의 양을 측정하는데 중요하게 생각되는 것은 input size입니다.   
Basic Operation은 Primitive Operation에서 가장 기본의 되는 연산을 말하며 이를 이용해서 대략적으로 basic operation의 수를 n에 관한 함수로 나타내는 방법입니다.   
Primitive Operation은 기계어측면에서 상수시간에 수행되는 연산들로 덧셈, 뺄셈, 곱셈, 나눗셈등을 의미합니다.   
일의 양을 측정할때는 Input의 조건을 잘 확인해야합니다.   

Worst-Case Complexity는 문제를 해결할대 아무리 늦어도 이 시간안에는 해결할 수 있다는 것을 의미합니다.   
$D_n$을 사이즈가 n인 input들의 집합이라고 보고, $t(I)$는 input I에 관해서 알고리즘이 수행하는 basic operation의 수를 지칭합니다. 그때 Worst-case complexity를 생각하면 $W_n=max({t(I)|I\in D_n})$입니다.   
다시한번 생각해보면 size가 n인 임의의 입력에 대해서 알고리즘이 수행하는 basic operation의 최대의 수라고 생각할 수 있습니다.   

Average Complexity는 worst와 best의 중간이 아니라 다양한 데이터에 대해서 평균적으로 이 시간내에 수행된다는 것을 의미합니다.   
Average Complexity를 확률의 개념을 포함합니다.   
$Pr(I)$가 특정 input이 발생할 수 있는 확률임을 이용해서 나타냅니다.   
$A(n)=\sum_limits_{I\in D_n}Pr(I)t(I)$   
따라서 위의 식을 해석해보면 어떤 사건이 발생할 확률에 그때의 basic operation의 수를 곱하고 모든 경우에 대해서 더한 값을 의미합니다.   
여기서 $t(I)$의 경우는 알고리즘을 분석하면 이를 구할 수 있지만, $Pr(I)$는 알고리즘의 분석으로는 알 수 없고 통계적인 수치, 가정을 이용해서 정의 합니다.   
![2021-09-08-2](https://user-images.githubusercontent.com/55619678/132504542-a4c980e1-a902-46bd-b96c-7f869db59db5.png)   
![2021-09-08-1](https://user-images.githubusercontent.com/55619678/132504536-40a317c1-2a39-41dd-a24b-35633c55fa11.png)    
위의 알고리즘의 Average Complexity를 구하는 과정입니다.   
- Space complexity
공간 복잡도의 경우 시간복잡도와 구하는 방법이 비슷하고, input에 따라서 사용되는 메모리의 양이 결정된다면 분석할 수 있습니다.   
대부분의 알고리즘은 Time 과 Space간 tradeoff관계를 가집니다.   
- Simplicity   
간결성은 불필요한 연산들을 제거하여 표현하는 것입니다.   
간결성의 장점으로는 **알고리즘의 correctness에 대한 증명을 쉽게 할 수 있다**와 **프로그램의 작성, 구현, 디버깅, 수정할때 쉬워진다**라는 두가지 장점이 있습니다.   
- Optimality   
Optimality를 분석하기 위해서는 문제의 복잡도를 분석해야합니다.   
각 문제는 각자의 고유한 복잡도를 가지고 있습니다.   
이 복잡도는 문제를 해결하기 위한 최소 basic operation의 수가 존재한다로 표현할 수 있습니다.    
이를 구하기 위해서는 먼저 문제에 대한 basic operation을 설정(알고리즘이 사용하는 연산의 type을 정의)하고 그 수를 세어 계산합니다.   
따라서 그 문제를 해결하기 위해서는 실제 얼마나 많은 연산의 수가 필요하냐 라고 생각 할 수 있습니다.   
최적의 알고리즘이다라고 표현하는 것은 문제의 복잡도와 알고리즘의 복잡도가 같은 경우 그렇게 표현합니다.   
이 두 복잡도가 다른 경우는 두가지의 경우로 생각할 수 있는데 첫번째는 더 효율적인 알고리즘이 존재하는가, 두번째는 더 좋은 lower bound가 존재하는가 로 생가할 수 있습니다.   

### Classifying Functions by Their Asymptotic Growth Rates    
Asymptotic analysis는 점근적분석으로 이론적인 분석을 위한 방법입니다. 이는 Basic Opeartion을 선택하고 이를 input size n에 관해서 나타내는 방법입니다.   
Asymptotic growth rate는 input size n이 충분히 커졌을 때 그 함수의 증가율을 말합니다.   
이때 자료구조 시간에 배웠듯이 최고차항만을 가지고 비교합니다.   
복잡도를 표기하기 위한 방법으로는 크게 3가지가 있습니다.   
- $O(g)$ : g 함수의 증가율 보다 더 빠를 수 없다. 또는 증가율이 같거나 더 작은 함수들의 집합이라고 생각할 수 있습니다.   
- $\Omega(g)$ : 적어도 g 함수의 증가율 만큼 빠른 증가율을 갖는 함수들의 집합이라고 생각할 수 있습니다.   
- $\Theta(g)$ : 위의 두 함수의 교집합을 의미하며 g 함수와 증가율이 같은 함수들의 집합을 말합니다.   

이제 위의 표현들을 두개의 함수 f, g를 가지고 정리하겠습니다.    
이를 정리하기 위해서는 양의 상수 c, 음이 아닌 정수 $n_0$가 필요합니다.   
- $O(g)$ : $f(n)\le cg(n)$을 만족하며 n의 범위는 $n\ge n_0$입니다. 
- $\Omega(g)$ : $f(n)\ge cg(n)$을 만족하며 n의 범위는 $n\ge n_0$입니다.
- $\Theta(g)$ : 위의 두 식을 동시에 만족하는 c, $n_0$를 찾습니다.   

다음은 $f(n),g(n)$을 비교하는 방법으로 n이 무한대로 가는 점근적 분석을 이용해서도 비교할 수 있습니다.   
$\lim\limits_{n\rightarrow\infty}{f(n)\over g(n)}$   
위의 식을 계산한 결과를 통해 비교할 수 있습니다.   
- 0, 상수(C) : f함수의 증가율이 g보다 작으면 0, 같으면 상수에 수렴한다.   
- 상수(C), $\infty$ : f함수의 증가율이 g보다 크거나 같을때 나온다.   
- 상수(C) : f함수의 증가율이 g와 같을 경우에 나온다.   
- 0 : 항상 f함수의 증가율이 작은 경우인데 이러한 경우를 "little oh of g"라고한다.   
- $\infty$ : 항상 f함수의 증가율이 큰 경우인데 이러한 경우를 "little omega of g"라고 한다.   
