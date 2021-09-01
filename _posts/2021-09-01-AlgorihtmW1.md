---
title:  "Algorithm Week1"
excerpt: "인하대학교 김영호 교수님의 알고리즘 수업 week1 Review"

categories:
  - Algorithm
tags:
  - [Blog, jekyll, Github, Git]

toc: true
toc_sticky: true
 
date: 2021-09-01
last_modified_at: 2021-09-01
---   
# Algorithm Week1   
> 포스팅하는 내용은 모두 인하대학교 김영호 교수님의 강의 내용을 바탕으로 정리하였습니다.    
## 1. Analyzing Algorithms and Problems : Principles and Examples   

### Introduction   
컴퓨터 알고리즘이란 무엇인가에 대해서 먼저 알아보겠습니다.    
컴퓨터 알고리즘이란 문제를 해결하기 위한 구체적인 step-by-step의 방법을 말합니다. 다르게 표현하면 **컴퓨터를 사용하여 문제를 해결하는 단계적 절차**를 의미합니다. 여기서 조건이 붙는데 이는 유한시간(finite time)내에 해결해야한다는 점입니다.    
그렇다면 컴퓨터를 이용해서 문제를 해결하는 절차에 대해서 알아보겠습니다.   
1. 문제를 정의합니다. 여기서 문제를 정의하는 방법은 Input과 Output을 이용해서 문제를 정의합니다.   
2. 문제를 해결할 전략을 세웁니다.   
3. 알고리즘을 서술합니다. 알고리즘을 서술할 때는 Input, Output, Step(단계)에 따라서 서술합니다.   
4. 알고리즘을 분석합니다. 알고리즘을 분석할때는 correctness(정확성), time & space(efficiency), optimality(최적성)을 분석합니다. 나중에 설명하겠지만 최적성이란 알고리즘의 복잡도가 문제의 복잡도가 같은경우를 optimal algorithm이라고 합니다.   
5. 알고리즘을 구현합니다. 
6. 알고리즘을 검증합니다.    

알고리즘을 분석하는 방법에는 2가지가 있습니다. 첫번째 방법은 실험적분석, 두번째 방법은 이론적분석입니다.    
실험적분석을 하는 경우에는 3가지의 단점이 존재합니다.  
1. 알고리즘을 분석하기 위해서는 구현이 필수적으로 요구되는데 구현이 쉽지 않은 경우가 존재합니다.  
2. 실험에 포함되지 않는 input의 수행시간에 대해서는 알 수 없습니다.   
3. 하나의 문제를 해결하는 algorithm A, algorithm B의 효율성을 비교하기 위해서는 소프트웨어와 하드웨어의 스펙을 동일하게 해야합니다.   

이론적분석을 하는 경우에는 4가지의 장점이 존재합니다.   
1. 구현대상에 대해서 상위레벨로 서술할 수 있습니다.  
2. 수행시간을 input size인 n으로 표현할 수 있습니다.   
3. 모든 가능한 input에 대해서 고려 가능합니다.   
4. 소프트웨어와 하드웨어의 스펙에 독립적으로 분석이 가능합니다.   

그렇다면 이론적인 분석을 하기위해서 알고리즘의 양을 어떻게 측정할지 고려해야합니다.   
Basic Operation은 알고리즘이 수행될 때 가장 기본이 되는 operation을 의미합니다.   
이후 Worst-Case 분석을 진행합니다. $W(n)$은 input size인 n에 대해서 수행하는 basic operation의 최대 개수를 말합니다.    
예를 들어, 가장 간단한 search알고리즘에서 비교분석을 통해 n개의 배열의 원소를 찾는다면, $W(n)=n$이 될 수 있습니다.   
또한 다른 분석방법으로 Average-Behavior Analysis, Optimality, Correctness가 있습니다.   
알고리즘은 Pseudo-code를 통해서 간략하게 서술합니다. 이때 일반적인 Pseudo-code는 언어의 제약이 없으며, 한국어, 영어에 상관없이 서술이 가능하며 문장으로 이어서 작성해도 됩니다. 또한 너무 자세히 적는것이 아닌 해당 알고리즘의 전략이나 기술에 대해서 적어야합니다.   
### Mathematical Background   
알고리즘을 이해하기 위한 간단한 수학적 배경지식으로는 수열의 합을 구하는 방법과 논리학의 표현법이 있습니다.   
Arithmetic series    
$\sum\limits_{i=1}^{n}i={n(n+1)\over 2}$   
Polynomial Series   
$\sum\limits_{i=1}^{n}i^{2}={2n^{3}+3n^{2}+n\over 6}\approx {n^{3}\over 3}$    
$\sum\limits_{i=0}^{n}i^{k}\approx {n^{k+1}\over {k+1}}$    
Power of 2   
$\sum\limits_{i=1}^{n}2^{i}=2^{k+1}-1$   
Arithmetic-Geometric Series   
$\sum\limits_{i=1}^{n}i2^{i}=(k-1)2^{k+1}+2$    
위의 공식을 모두 증명하지는 않았지만 Power of 2, Arithmetic-Geometric에 대해서는 증명을 연습해보았습니다.   
Logic에 대한 간단한 예로는   
$A\Rightarrow B$ 는 $\neg A\vee B $ 와 같습니다.  
$\neg(A\vee B)$는 $\neg A\land\neg B$와 드모르간의 법칙에 의해 동치입니다.   
$\neg(A\land B)$는 $\neg A\vee\neg B$와 드모르간의 법칙에 의해 동치입니다.   
증명을 위한 수학적은 배경지식으로는 Counterexample(반례), Contraposition(대우), Contradiction(모순법, 귀류법), Mathematical Induction(수학적 귀류법)등이 있습니다.   
