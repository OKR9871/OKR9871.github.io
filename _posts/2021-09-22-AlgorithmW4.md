---
title:  "Algorithm Week4"
excerpt: "인하대학교 김영호 교수님의 알고리즘 수업 week4 Review"

categories:
  - Algorithm
tags:
  - [Blog, jekyll, Github, Git]

toc: true
toc_sticky: true
 
date: 2021-09-22
last_modified_at: 2021-09-22
---   
# Algorithm Week4   
> 포스팅하는 내용은 모두 인하대학교 김영호 교수님의 강의 내용을 바탕으로 정리하였습니다.   

## 1. Analyzing Algorithms and Problems : Principles and Examples   
Binary Search의 경위 $O(logn)$의 복잡도를 가집니다. 이를 더 향상시킬 수 있는 방법에 대해서는 optimality분석을 통해서 생각을 해봐야합니다.   
따라서 Binary Search의 Optimality를 분석하기 위해서는 Decision tree를 이용해서 분석합니다.    
Decision Tree의 경우 알고리즘과 input size가 주어지면 항상 같은 모양을 생성합니다.   
이때 이러한 Decision Tree를 만들기 위해서 아래와 같은 규칙을 만족합니다.   
- 각 노드는 0부터 n-1까지로 labeling할 수 있습니다.   
- Decision tree의 root는 임의의 Algorithm A가 가장 처음 비교하는 entry의 index로 설정합니다.  
- 찾으려는 K가 entry보다 작을때 다음에 비교할 entry의 index가 왼쪽 노드의 label이 됩니다.   
- 찾으려는 K가 entry보다 클때는 다음에 비교할 entry의 index가 오른쪽 노드의 label이 됩니다.   
- 더 비교할 entry가 없으면 중단합니다.   

![2021-09-22-1](https://user-images.githubusercontent.com/55619678/134375248-25382a25-2c21-40b1-b3f6-506b22eb2946.png)    
![2021-09-22-2](https://user-images.githubusercontent.com/55619678/134375253-6d80cc57-f028-4e8a-82a5-7690e85adfb4.png)     
위의 규칙을 만족하면서 binary search tree의 경우 decision tree를 위의 그림과 같이 나타낼 수 있습니다.   
또한 이전 수업시간에 배웠던 algorithm A, B, C의 경우도 위의 그림과 같이 나타낼 수 있습니다.    
위의 그림을 통해서 Decision Tree를 분석해보면 worst case의 경우는 root에서 부터 leaf 까지 가장 긴 path(노드의 수)가 되고 이를 p라고 부릅니다.   
이때 Decision Tree의 전체 node의 수를 파악해보면 최대 $N\le 1+2+4+...+2^{p-1}=2^p-1$ 입니다.   
따라서 이를 정리하면 $2^p\ge(N+1)$입니다.   
이를 통해 우리가 주장하는 바는 Decision tree의 총 노드의 수인 N은 항상 input size인 n보다 크거나 같다입니다.   
따라서 이를 증명하기 위해서는 **Contradiction(모순법))** 을 이용합니다. 바로 $N<n$이면 모순이다라는 것을 보이면 됩니다.    
이는 만약 임의의 index i를 제외하고 같은 값을 갖는 input array가 있을때 $N<n$이라면 decision tree의 경우 그 값에 대해서는 비교할 수 없는 상황이 나오게 됩니다. 그렇다면 실제로는 두 값이 다르지만 항상 같은 결과를 반환하게 됩니다. 따라서 $N\ge n$을 만족해야합니다.   
그렇게 Decision tree가 n보다 큰값을 갖는다는 것을 증명하고 나면 $2^p\ge(N+1)\ge(n+1)$로 나타낼 수 있습니다.   
이를 log를 취해서 계산하면 $p\ge log(n+1)$을 만족하고 p는 비교연산의 수로 정수이므로 올림해주게 되면 이는 최적임을 알 수 있습니다.   

## 2. Data Abstraction and Basic Data Structures    
Abstract Data Type은 데이터형을 추상화 하는 것으로 다른 말로는 데이터의 특징을 요약하는 의미를 가집니다.   
ADT는 Structure와 Function으로 나눌 수 있고 Structure는 데이터에 대한 선언, Function은 operation에 대한 정의를 말합니다.   
ADT를 사용하는 장점으로는 두가지가 있습니다.   
1. ADT는 algorithm의 correctness를 확인할 수 있습니다.   
2. ADT는 구현에 대한 performance analysis를 서로다른 설계와 구현의 개념으로 분리해서 생각할 수 있게 해줍니다.   

다음으로는 Tree Data Structure에서 사용하는 용어에 대해서 살펴보겠습니다.   
- Root : 부모가 없는 노드   
- Degree : 그 노드의 자식의 수   
- External node(leaf) : 자식이 없는 노드
- Internal node : 최소 하나의 자식이 있는 노드   
- Ancestor : root부터 x까지의 unique simple path에서 그 사이의 node들을 말한다. 자기 자신을 제외한 ancestor를 proper ancestor이라고 하기도 한다.   
- Descendant : Ancestor의 반대의 개념  
- Subtree : x가 root이고 그 descendant들로 이루어진 tree   
- Depth : 임의의 노드의 depth는 parent node의 depth + 1, root의 depth는 0으로 정의   
- Level : 같은 depth로 이루어진 노드들   
- Height : tree of height는 leaf들중 최대 깊이를 갖는 것을 말하며, 임의의 노드의 height는 그 노드가 root인 서브트리에서의 height   

Binary Tree ADT   
Binary Tree는 모든 노드가 최대 2개의 자식 노드를 갖는 tree를 말합니다.    
각각의 subtree는 binary tree의 성질을 모두 만족해야합니다.   
1. root노드를 가집니다.   
2. 노드를 기준으로 2개의 subtree를 L, R이라고 하는데 각각은 모두 binary tree입니다.   

여기서 binary tree의 3가지 특징이 있습니다.   
1. binary tree의 depth가 d일때 최대 노드의 수는 $2^d$입니다.  
2. binary tree의 height가 h일때 최소 노드의 수는 $2^{h+1}-1$를 가집니다.  
3. binary tree가 n개의 노드를 가질때 최소 height는 $log(n+1)-1$입니다.   

Stack    
Stack은 top이라고 불리는 가장 최근에 들어온 entry를 가르키는 것이 있으며 push, pop을 이용해 삽입 삭제연산을 수행합니다.   
이때 삽입삭제는 top에서만 이루어지며 이러한 구조를 last in, fisrt out(LIFO)라 합니다.   

Queue   
Queue는 rear라는 가장 마지막의 element를 나타내는 것과, front라는 가장 앞의 element를 나타내는 것을 가지며, enqueue, dequeue연산을 이용해 삽입, 삭제 연산을 수행합니다.   
이때 first in, first out(FIFO)로 수행합니다.   

Priority Queue    
Priority Queue는 데이터의 정보가 우선순위로 이용되는 것을 말합니다.   
이때 위의 Queue는 Priority Queue의 일종으로 데이터가 도착하는 시간을 우선순위로 보고 사용하는 방법입니다.    
이를 구현하기 위해서는 3가지 operation이 필요합니다.   
1. getMin, getMax : 가장 우선순위가 높은거 탐색
2. insert : 원소 삽입
3. removeMin, removeMax : 우선순위가 높은거 제거 하면서 꺼내기

Priority Queue는 구현하는 방법에 따라 위의 operation의 수행시간이 달라집니다.   
![2021-09-22-3](https://user-images.githubusercontent.com/55619678/134375255-5956c445-2ce2-4cef-99bb-b6e6951ccc90.png)    

Union-Find ADT for Disjoint Sets   
사용하는 application이 undirected graph에서 connected component를 찾을 때 사용하는 자료구조의 형태입니다.   
기본적으로는 union, find operation을 제공합니다.   
set-id라는 각 집합을 나타내는 key값을 가지는데 이 key값이 다를 경우 union연산을 수행합니다.   
Find연산은 한 element가 포함된 집합의 set-id를 찾는연산입니다.  

Dictionary ADT    
Dictionary는 [key, value]쌍으로 이루어진 자료구조를 말합니다. 이 key들은 정렬되어 있지는 않으며, hash, binary tree를 통해 구현하곤 합니다.   

## 4. Sorting    
Sorting은 다양한 문제의 sub problem으로써 사용되며 문제를 푸는데 도움이 되는 기본적인 알고리즘입니다.   
- Insertion Sort(삽입 정렬)   

삽입정렬은 데이터가 삽입될때 오름차순을 유지하도록 해주는 방법입니다.   
삽입정렬은 2개의 영역 sorted segment, unsorted segment로 나누어서 수행하며 unsorted segment에서 하나씩 빼서 sorted segment로 삽입해주는 방식으로 수행합니다.   
![2021-09-22-4](https://user-images.githubusercontent.com/55619678/134375256-c31cd5ba-edfc-4275-a5f5-4f57cadcca71.png)    
![2021-09-22-5](https://user-images.githubusercontent.com/55619678/134375260-73918455-2efc-4844-a3e9-c97912aab09c.png)    
![2021-09-22-6](https://user-images.githubusercontent.com/55619678/134375261-7ae6f452-9ad5-4ece-a104-de7b0315cd91.png)    
이때 loop invariant를 이용해서 correctness를 분석하는데 이 방법은 아래와 같습니다.  
1. Initialization(초기화) : 알고리즘이 loop에서 처음 loop를 돌기전에 참임을 보입니다.   
2. Maintenance(유지보수) : loop에서 어떤 iteration이 수행되고난 후에도 참임을 보입니다.    
3. Termination(종료) : 모든 iteration이 종료된 후에 전체적으로 참임을 보입니다.   
