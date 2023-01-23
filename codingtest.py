# 그리디 알고리즘 - 큰 수의 법칙 (92p 실전문제)
"""
N, M, K = map(int, input().split())
n = list(map(int, input().split()))

n.sort()
first = n[N-1]
second = n[N-2]
result = 0

# 단순하게 풀기
'''
while M != 0:
    for _ in range(K):
        result += first
        M -= 1
    result += second
    M -= 1
'''

# 큰수가 반복되는 횟수를 구해서 풀기
count = int(M / (K+1)) * K
count += M % (K+1)

result += first * count
result += second * (M-count)

print(result)
"""

# 숫자 카드게임
'''
n, m = map(int, input().split())

result = 0 #result초기화 필요
for _ in range(n):
    number = list(map(int, input().split()))
    min_number = min(number)
    result = max(min_number, result)

print(result)
# 나는 작은값들의 리스트를 만들고 그곳에서 제일 큰 수를 뽑을 걸 생각하고있엇는데.. 이럴수도 잇구만
'''

# 1이 될 때까지
'''
n, k = map(int, input().split())

count = 0
while n != 1:
    if n % k == 0:
        n = n // k
        count += 1
    else:
        n -= 1
        count += 1

print(count)
'''



# 구현 - 왕실의 나이트 (115p 실전문제)
# ** 다시 풀어보기 **
'''
M = input()
row = int(M[1]) #행
column = int(ord(M[0])) - int(ord('a')) + 1 #열
# ord 함수 -> 아스키코드 int형으로 반환

steps = [(-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)]

result = 0

for step in steps:  # 리스트 요소 자체를 for문 옆 변수에 넣어버릴 수 있다.
    Nrow = row + step[0]
    Ncolumn = column + step[1]

    if Nrow < 1 or Nrow > 8 or Ncolumn < 1 or Ncolumn > 8:
        continue
    else:
        result += 1

print(result)
'''

# 게임 개발
# ** 다시 풀어보기 **
# 방향을 설정해서 이동하는 문제 유형 -> dx, dy같은 별도의 리스트를 만들어 방향을 정하는 것이 효과적.
'''
N, M = map(int, input().split())

d = [[0] * M for _ in range(N)]
x, y, direction = map(int, input().split())
d[x][y] = 1

Map = []
for _ in range(N):
    Map.append(list(map(int, input().split())))

dx = [-1, 0, 1, 0]  #북동서남 방향정의
dy = [0, 1, 0, -1]

def turn_left():
    global direction  #전역변수를 함수 안에서도 쓰기위해서 global 사용
    direction -= 1
    if direction == -1:
        direction = 3

count = 1
trun_time = 0
while True:
    turn_left()
    nx = x + dx[direction]
    ny = y + dy[direction]

    if d[nx][ny] == 0 and Map[nx][ny] == 0:
        d[nx][ny] = 1
        x = nx
        y = ny
        count += 1
        trun_time = 0
        continue
    else:
        trun_time += 1
    if trun_time == 4:
        nx = x - dx[direction]
        ny = y - dy[direction]

        if Map[nx][ny] == 0:
            x = nx
            y = ny
        else:
            break
        trun_time = 0

print(count)
'''

# DFS 메서드 정의 - 깊이 우선 탐색, 재귀함수로 구현 가능
'''
def dfs(graph, v, visited):
    visited[v] = True  # 현재 노드 방문 처리
    print(v, end = ' ')
    for i in graph[v]:  # 현재 노드와 연결된 다른 노드를 재귀적으로 방문
        if not visited[i]:
            dfs(graph, i, visited)

graph = [  # 각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원 리스트)
    [], # 0번째라서 무시
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]

# 각 노드가방문된 정보를 리스트 자료형으로 표현(1차원 리스트)
visited = [False] * 9

# 정의된 DFS함수 호출
dfs(graph, 1, visited)
'''


# BFS 메서드 정의 - 너비 우선 탐색, 큐 사용
'''
from collections import deque

def bfs(graph, start, visited):
    queue = deque([start])
    visited[start] = True  # 현재 노드 방문처리

    while queue:
        v = queue.popleft()  # 큐에서 하나의 원소 뽑아 출력
        print(v, end=' ')

        for i in graph[v]:
            if not visited[i]:
                queue.append(i)  # 해당원소와 연결된, 아직 방문하지 않은 원소들을 큐에 삽입
                visited[i] = True


graph = [  # 각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원 리스트)
    [],  # 0번째라서 무시
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]

# 각 노드가방문된 정보를 리스트 자료형으로 표현(1차원 리스트)
visited = [False] * 9

bfs(graph, 1, visited)
'''

# 선택 정렬 - 처음부터 모두 훑으며 앞으로 보내기
"""
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
'''
for i in range(len(array)):
    min_n = array[i]
    for j in range(i+1, len(array)):
        if min_n > array[j]:
            min_n = array[j]
    array[i], min_n = min_n, array[i]  # 스와프는 두 변수의 '위치'를 변경하지 이렇게하면 숫자만 변경되고 원래 리스트는 그냥 그대로 다시시작 
'''

for i in range(len(array)):
    min_n = i
    for j in range(i+1, len(array)):
        if array[min_n] > array[j]:
            min_n = j
    array[i], array[min_n] = array[min_n], array[i]  # 스와프는 인덱스로 해야함

print(array)
"""


# 삽입 정렬 - 선택정렬에 비해 효율적. 정렬이 거의 되어있는 상황에서는 퀵정렬 알고리즘보다 더 효율적일 수 있다.
'''
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(1, len(array)):
    for j in range(i, 0, -1):  # range(start, end, step) start인덱스부터 end인덱스까지 step만큼 감소, 증가
        if array[j] < array[j-1]: 
            array[j], array[j-1] = array[j-1], array[j]
        else:
            break

print(array)
'''

# 퀵 정렬 - 가장 많이 사용되는 알고리즘. 데이터가 무작위로 입력되는 경우 효율적. 데이터의 특성을 파악하기 어렵다면 퀵정렬이 유리.
# 피벗 - 큰 숫자와 작은 숫자를 교환할 때, 교환하기 위한 기준
# 호어 분할 - 리스트에서 첫번째 데이터를 피펏으로 정함.
'''
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

def quick_sort(array, start, end):
    if start >= end:
        return
    pivot = start
    left = start + 1
    right = end
    while left <= right:
        while left <= end and array[left] <= array[pivot]:
            left += 1 
        while right > start and array[right] >= array[pivot]:
            right -= 1
        if left > right:
            array[right], array[pivot] = array[pivot], array[right]
        else:
            array[left], array[right] = array[right], array[left]
quick_sort(array, start, right - 1)
quick_sort(array, right + 1, end)

quick_sort(array, 0, len(array) - 1)
print(array)

# 파이썬 장점 살린 퀵 정렬 - 피벗과 데이터를 비교하는 연산횟수 증가로 시간면에서 조금 비효율적
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

def quick_sort(array):
    if len(array) <= 1:
        return array
    
    pivot = array[0]
    tail = array[1:]

    left_side = [x for x in tail if x <= pivot]
    right_side = [x for x in tail if x >= pivot]

    return quick_sort(left_side) + [pivot] + quick_sort(right_side)

    print(quick_sort(array))
'''

# 계수 정렬 - 특정한 조건이 부합할 때만 사용할 수 있지만 매우 빠른 정렬 알고리즘. 동일한 값의 데이터가 여러 개 등장할 때 적합.
# 조건 - 데이터의 크기 범위가 제한되어 정수 형태로 표현할 수 있을 때. 가장 큰 데이터와 가장 작은 데이터의 차이가 1000000을 넘지 않을 때.
'''
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8, 0, 5, 2]

count = [0] * (max(array)+1)

for i in range(len(array)):
    count[array[i]] += 1

for i in range(len(count)):
    for j in range(count[i]):
        print(i, end=' ')
'''

# 정렬라이브러리에서 key를 활용한 소스코드
'''
array = [('바나나', 2), ('사과', 5), ('당근', 3)]

def setting(data):
    return data[1]

result = sorted(array, key=setting)
print(result)
'''

N = int(input())
count = 0
m = 0
number = []

for _ in range(N):
    number.append(tuple(map(int, input().split())))

number = sorted(number, key=lambda number: number[0])
number = sorted(number, key=lambda number: number[1]) 

for num in number:
    a, b = num[0], num[1]
    if a < 0:
        break
    if a >= m:
        count += 1
        m = b

print(count)
