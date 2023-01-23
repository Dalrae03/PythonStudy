## DFS 메서드 정의 
- 깊이 우선 탐색, 재귀함수로 구현 가능

```python
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
```
<br>
<br>

## BFS 메서드 정의 
- 너비 우선 탐색, 큐 사용
```py
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
```