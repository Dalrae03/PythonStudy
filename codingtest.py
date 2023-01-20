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