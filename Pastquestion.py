# 유형별 기출
# I. 그리디

# 1. 모험가 길드
"""
N = int(input())
slist = list(map(int, input().split()))
result = 0
slist.sort()

# 최소 그룹 구한 것 같다...
'''
n = N-1

while n>=0:
    result += 1
    n -= slist[n]
'''

# 해답
count = 0
for i in slist:
    count += 1
    if count >= i:
        result += 1
        count = 0

print(result)
"""


# 2. 곱하기 혹은 더하기
"""
S = input()

# 내가 푼 방식 (후에 result가 0일경우는 오류가 난다. (근데 처음은 몰라도 후에 0이 될 경우가 있을까...?))
numbers = []
result = 0

'''
for i in S:
    i = int(i)
    numbers.append(i)

if numbers[0] == 0 or numbers[1] == 0 or numbers[0] == 0 or numbers[1] == 0:
    result = numbers[0] + numbers[1]
else:
    result = numbers[0] * numbers[1]


for i in range(2, len(numbers)):
    N = numbers[i]
    if N == 0 or N == 1:
        result += N
    else:
        result *= N


for i in S:
    i = int(i)
    numbers.append(i)

if numbers[0] <= 1 or numbers[1] <= 1:
    result = numbers[0] + numbers[1]
else:
    result = numbers[0] * numbers[1]


for i in range(2, len(numbers)):
    N = numbers[i]
    if N <= 1:
        result += N
    else:
        result *= N
'''

# 더 간단하고 좋은 답
result = int(S[0])

for i in range(1,len(S)):
    N = int(S[i])
    if result <= 1 or N <= 1:
        result += N
    else:
        result *= N


print(result)
"""


# 3. 문자열 뒤집기
# https://www.acmicpc.net/problem/1439
'''
S = input()

# 내 풀이

result = 0

first = S[0]
for i in range(len(S)-1):  #첫번째 수와 2번째 수가 다를경우도 고려해줘야해!
    s = S[i]
    if s == first and s != S[i+1]:
        result +=1

print(result)
'''


# 다른 풀이
'''
S = input()
count0 = 0
count1 = 0

if S[0] == '1':
    count0 += 1
else:
    count1 += 1

for i in range(len(S)-1):
    if S[i] != S[i+1]:
        if S[i+1] == '1':
            count0 += 1
        else:
            count1 += 1

print(min(count0,count1))
'''


# 4. 만들 수 없는 금액
"""
N = int(input())
numbers = list(map(int, input().split()))

# list를 가지고 서로 더해서 구할 수 있는 모든 수를 구한다음에 중복을 제거하고 자연수를 차례로 있는지 비교하려고했지만
# 몇개가 들어올지도 모르니 for문을 무작정 늘릴수도 없어서 개같이 실패....ㅋㅋㅋㅋ
'''
Snumbers =set(numbers)

for i in range(N):
    for j in range(i+1, N):
        Snumbers.add(numbers[i]+numbers[j])

for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            Snumbers.add(numbers[i]+numbers[j]+numbers[k])

print(Snumbers)
'''

# 답안
# 일단 list 오름차순 정렬 후 list의 가장 작은 수, target = 1부터 target - 1까지 모든 금액을 만들 수 있다고 가정하고 비교, 증가 시키기
# 더이상 list요소가 없거나, list요소가 target보다 클 경우 그 target이 만들 수 없는 최소 자연수
numbers.sort()

target = 1
for i in numbers:
    if target < i:
        break
    target += i

print(target)
"""


# 5. 볼링공 고르기
"""
'''
N, M = map(int, input().split())
balls = list(map(int, input().split()))
count = 0

for i in range(N):
    for j in range(i+1, N):
        if balls[i] != balls[j]:
            count += 1

print(count)
'''

# 다른 답안
N, M = map(int, input().split())
balls = list(map(int, input().split()))
result = 0

array = [0]*11

for i in balls:
    array[i] += 1

for i in range(1, M+1):
    N -= array[i]
    result += N * array[i]

print(result) 
"""


# 6. 무지의 먹방라이브
# https://programmers.co.kr/learn/courses/30/lessons/42891
"""
import heapq
List = [3,1,2]
k = 5

# 내 풀이
'''
def solution(food_times, k):
    time = 0
    answer = 0
    check = 0
    N = len(food_times)
    
    while time != k:
        for i in range(N):
            if food_times[i] != 0:
                food_times[i] -= 1
                time += 1
                answer = i
            else:
                check += 1
            if time == k:
                break

    if check == N:
        answer = -1
    # 여기서 문제. 다음으로 먹어야할 음식의 요소가 0일경우를 고려하지않고 무조건 다음 음식을 먹도록 설정했다... 
    # 이거 if문 또 분기해서 0아닌 수 나올때까지 for문돌려야할것같은데 그럼 코드가 너무 더러워지지않나...
    else:
        if answer == N-1:
            answer = 1
        else:
            answer += 1
    
    return answer

print(solution(List,k))
'''

# 답안
# 우선순위 큐(힙) 선행 필요 -> 복습요망

def solution(food_times, k):
    if sum(food_times) <= k:
        return -1

    q = []
    for i in range(len(food_times)):
        heapq.heappush(q,(food_times[i], i+1))

    sum_value = 0 #먹기위해 사용한 시간
    previous = 0 #직전에 다 먹은 음식 시간
    length = len(food_times)

    # 먹기위해 사용한 시간 + (현재의 음식 시간 - 이전 음식 시간) * 현재 음식 개수와 k비교
    # 제일 작은 음식을 없애기 위해 list돌았으니까 돈만큼 값을 빼줘야지
    while sum_value + ((q[0][0] - previous) * length) <= k:
        now = heapq.heappop(q)[0]
        sum_value += (now - previous) * length
        length -= 1
        previous = now #이전 음식 시간 재설정
    
    #남은 음식 중에서 몇 번째 음식인지 확인하여 출력
    result = sorted(q, key = lambda x: x[1]) #음식의 번호 기준으로 정렬
    return result[(k - sum_value) % length][1]
    

print(solution(List,k))
"""



# II. 구현

# 7. 럭키 스트레이트
# https://www.acmicpc.net/problem/18406
'''
N = input()
L = len(N)
l = int(L/2)
ln = 0
rn = 0

for i in range(l):
    ln += int(N[i])

for i in range(l, L):
    rn += int(N[i])

if ln == rn:
    print("LUCKY")
else:
    print("READY")
'''
'''
N = input()
L = len(N)
sum = 0

for i in range(L//2):
    sum += int(N[i])

for i in range(L//2, L):
    sum -= int(N[i])

if sum == 0:
    print("LUCKY")
else:
    print("READY")
'''


# 8. 문자열 재정렬
'''
N = input()
count = 0
sum = 0

for i in N:
    if ord('0') <= ord(i) and ord(i) <= ord('9'):
        count += 1
        sum += int(i)

n = sorted(N)

for i in range(count, len(N)):
    print(n[i], end='')
print(sum)
'''

# 다른 풀이
'''
data = input()
result = []
value = 0

for i in data:
    if i.isalpha():
        result.append(i)
    else:
        value += int(i)

result.sort()

if value != 0:
    result.append(value)

print(''.join(result))
'''

# 9. 문자열 압축
# https://school.programmers.co.kr/learn/courses/30/lessons/60057
'''
s = 'aabbaccc'

# 내 풀이 (망했고, 실패했지만... 해답의 단계와는 비슷했다...)
def solution(s):
    n = 1
    l = len(s)
    count = 0
    R = []
    answer = 1000

    while n <= l//2:
        a = s[0:n]
        for i in range(n, len(s)-n):
            b = s[i:i+n]
            if a == b:
                count += 1
            else:
                if count != 0:
                    R.append(count)
                    R.append(b)
                else:
                    R.append(b)
                a = s[i+n: i+n+n]
                i = i+n+n
        if answer > len(R):
            answer = len(R)
            count = 0
            R.clear()
        
        n += 1

    print(R)
    return answer

print(solution(s))
'''

# 해답
# range(start, stop, step) 3개의 매개변수를 사용
# 완전탐색이 가능하다. (범위가 1000으로 좁기 때문에)
# 리스트가 아닌 문자열 연산을 응용하였다.
'''
def solution(s):
    answer = len(s) #answer을 굳이 1000으로 안해도 된다. 초기 길이보다는 더 긴수를 갖지 못할테니까

    for step in range(1, len(s)//2+1):
        R ="" #리스트 초기화 하려했으면 여기서 했어야 했다...
        a = s[0:step]
        count = 1 #카운트 선언과 초기화도 여기서 했어야했다... 그것도 0이 아니라 1로 (근데 0으로 해도 이제 if문으로 잘 가를 수 있다고 본다)
        for i in range(step, len(s), step):
            b = s[i:i+step] #b로 따로 빼도 되고 안빼도 되고
            if a == b:
                count += 1
            else:
                R += str(count) + a if count >= 2 else a #if 조건부 표현식
                a = s[i:i+step] #문자열 인덱싱이 가지고있는 범위를 초과할 때, 오류가 나지않고 그냥 마지막 요소까지 다 출력을 한다.
                count = 1

        R += str(count) + a if count >= 2 else a #남은 문자열 처리
        answer = min(answer, len(R))

    return answer

print(solution(s))
'''


# 10. 자물쇠와 열쇠
# https://programmers.co.kr/learn/courses/30/lessons/60059
'''
key = [[0,0,0], [1,0,0], [0,1,1]]
lock = [[1,1,1], [1,1,0], [1,0,1]]

# 해답

# 90도 회전 함수
def rotate_a_matrix_by_90_degree(a):
    n = len(a)
    m = len(a[0])
    result = [[0] * n for _ in range(m)] #결과 2차원 리스트
    for i in range(n):
        for j in range(m):
            result[j][n-i-1] = a[i][j]
    return result

# 자물쇠 중간 부분이 모두 1인지 확인하는 함수
def check(new_lock):
    lock_length = len(new_lock) // 3
    for i in range(lock_length, lock_length*2):
        for j in range(lock_length, lock_length*2):
            if new_lock[i][j] != 1:
                return False
    return True

def solution(key, lock):
    n = len(lock)
    m = len(key)
    new_lock = [[0]*(n*3) for _ in range(n*3)]
    for i in range(n):
        for j in range(n):
            new_lock[i+n][j+n] = lock[i][j]

    for rotation in range(4):
        key = rotate_a_matrix_by_90_degree(key)
        for x in range(n*2):
            for y in range(n*2):
                for i in range(m):
                    for j in range(m):
                        new_lock[x+i][y+j] += key[i][j]
                if check(new_lock) == True:
                    return True
                for i in range(m):
                    for j in range(m):
                        new_lock[x+i][y+j] -= key[i][j]

    return False

print(solution(key, lock))
'''


# 11. 뱀
# https://www.acmicpc.net/problem/3190
# 내가 구현한 input정보들 받는 코드
# 뱀리스트를 만들어서 뱀이 길이가 늘어나도 뱀이 위치한 몸까지 다 기록하고자 했다.
# 망햇죠
'''
n = int(input())
M = [[0]*n for _ in range(n)]
apple = int(input())
for _ in range(apple):
    x, y = map(int, input().split())
    M[x-1][y-1] = 1

t = int(input())
trun = []
for _ in range(t):
    trun.append(list(input().split()))

snake = [[0,0]]
'''

# 해답
# 지도가 나와서 좌표'이동' 문제가 나오면 동서남북 이동거리 x,y리스트 만들어야한다
'''
n = int(input())
M = [[0]* (n+1) for _ in range(n+1)] #n+1만큼 해서 그냥 1부터 시작하게했다 / 0으로 해도 되는데 그럼 -1 다 따져야 해서 더 복잡해질듯
apple = int(input())
for _ in range(apple):
    x, y = map(int, input().split())
    M[x][y] = 1

t = int(input())
info = []
for _ in range(t):
    x, c = input().split()
    info.append((int(x), c))

dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]

def turn(direction, c):
    if c == "L":
        direction = (direction - 1) % 4
    else:
        direction = (direction + 1) % 4
    return direction

def simulate():
    x, y = 1, 1
    M[x][y] = 2 #뱀의 머리를 2로 표현
    direction = 0 #바라보는 방향 처음에는 동쪽
    time = 0
    index = 0 #다음에 회전할 정보
    q = [(x, y)] #뱀이 차지하고있는 위치정보
    while True:
        nx = x+dx[direction]
        ny = y+dy[direction]
        if 1 <= nx and 1 <= ny and nx <= n and ny <= n and M[nx][ny] != 2:
            if M[nx][ny] == 0:
                M[nx][ny] = 2
                q.append((nx, ny))
                px, py = q.pop(0)
                M[px][py] = 0
            if M[nx][ny] == 1:
                M[nx][ny] = 2
                q.append((nx, ny))
        else:
            time += 1
            break
        x, y = nx, ny
        time += 1
        if index < t and time == info[index][0]:
            direction = turn(direction, info[index][1])
            index += 1
    return time

print(simulate())
'''


# 12. 기둥과 보 설치
# https://school.programmers.co.kr/learn/courses/30/lessons/60061
# 망한 내 풀이 (오류잡지 못했고, 중간중간 print무더기는 오류 잡기위해서 리스트 내용 확인차 넣음)
'''
def solution(n, build_frame):
    coner = []
    answer = []

    for i in range(n+1):  #밑바닥 건설
        coner.append((i, 0))

    for j in build_frame:
        if j[2] == 0:
            if(j[0], j[1]) in coner:
                if j[3] == 1:
                    coner.append((j[0], j[1]+1))
                    answer.append([j[0], j[1], j[2]])
                    print(coner)
                    print(answer)
                    print('-' * 20)
                else:
                    coner.remove((j[0], j[1]))
                    coner.remove((j[0], j[1]+1)) #2,0,0을 없앨때 필요한 2,1코너도 없애버림... 그래서 2, 1이 없어서 list요소 오류가 난다.
                    answer.remove([j[0], j[1], j[2]])
                    print(coner)
                    print(answer)
                    print('-' * 20)
            else:
                continue
        else:
            if(j[0], j[1]) in coner or (j[0]+1, j[1]) in coner and (j[0], j[1]) in coner:
                if j[3] == 1:
                    if (j[0]+1, j[1]) in coner and (j[0], j[1]) in coner:
                        answer.append([j[0], j[1], j[2]])
                        print(coner)
                        print(answer)
                        print('-' * 20)
                    else:
                        coner.append((j[0]+1, j[1]))
                        answer.append([j[0], j[1], j[2]])
                        print(coner)
                        print(answer)
                        print('-' * 20)
                else:
                    coner.remove((j[0], j[1]))
                    coner.remove((j[0]+1, j[1])) #여기서 오류가 난다... not in list / 1, 1, 1, 0는 조건이 안맞아서 무시해야해.. 그래야 답이 맞아
                    answer.remove([j[0], j[1], j[2]])
                    print(coner)
                    print(answer)
                    print('-' * 20)
            else:
                continue
    
    return answer

print(solution(n, frame))
'''

# 해답
'''
n = 5
frame = [[0, 0, 0, 1], [2, 0, 0, 1], [4, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 1, 1], [3, 1, 1, 1], [2, 0, 0, 0], [1, 1, 1, 0], [2, 2, 0, 1]]


def check(answer):
    for x, y, stuff in answer:
        if stuff == 0:
            if y == 0 or [x - 1, y, 1] in answer or [x, y, 1] in answer or [x, y - 1, 0] in answer:
                continue
            return False
        else:
            if [x, y - 1, 0] in answer or [x + 1, y - 1, 0] in answer or [x + 1, y, 1] in answer and [x - 1, y, 1] in answer:
                continue
            return False
    return True



def solution(n, build_frame):
    answer = []

    for frame in build_frame:
        x, y, stuff, opreate = frame
        if opreate == 0:
            answer.remove([x, y, stuff])
            if check(answer) == False:
                answer.append([x, y, stuff])
        else:
            answer.append([x, y, stuff])
            if not check(answer):
                answer.remove([x, y, stuff])    
    
    return sorted(answer)


print(solution(n, frame))
'''


# 13. 치킨 배달

# 알고리즘은 맞은것 같은데 백준 1번째 예시는 잘 나오는데 나머지는 안나온다
# 백준에서 시간초과뜸...ㅠ
# https://www.acmicpc.net/problem/15686

'''
N, M = map(int, input().split())
Map = []

for i in range(N):
    Map.append(list(map(int, input().split())))

house = []
chicken = []

for i in range(N):
    for j in range(N):
        if Map[i][j] == 1:
            house.append((i, j))
        elif Map[i][j] == 2:
            chicken.append((i, j))

count = []
road = []

for i in range(len(chicken)):
    count.append(0)

for i in house:
    for j in chicken:
        R = abs(i[0]-j[0])+abs(i[1]-j[1])
        road.append(R)
    a = road.index(min(road))
    count[a] += 1
    road = []


while len(chicken) != M:
    a = count.index(min(count))
    count.append(a)
    chicken.append(a)


road = []
answer = 0

for i in range(len(chicken)):
    count.append(0)

for i in house:
    for j in chicken:
        R = abs(i[0]-j[0])+abs(i[1]-j[1])
        road.append(R)
    answer += min(road)
    road = []

print(answer)
'''

# 해답
'''
from itertools import combinations

N, M = map(int, input().split())
house = []
chicken = []
# house, chicken = [], [] 가능

for i in range(N):  #내가 했던것처럼 2차원리스트 만들고 해도 되는데 첫 반복문이 겹치기도 하고 계산속도도 감축할겸 합쳐버리고 2차원 리스트 안만들고
    Map = list(map(int, input().split()))
    for j in range(N):
        if Map[j] == 1:
            house.append((i, j))
        elif Map[j] == 2:
            chicken.append((i, j))

candidates = list(combinations(chicken, M))

def count(candidate):
    result = 0
    for hx, hy in house:  #한줄로 각자 따로 대입 가능
        temp = 1e9
        for cx, cy in candidate:
            temp = min(temp, abs(hx-cx)+abs(hy-cy))  #index이 따로 필요 없음
        result += temp
    return result

answer = 1e9
for c in candidates:
    answer = min(answer, count(c))

print(answer)
'''


# 14. 외벽점검
# https://school.programmers.co.kr/learn/courses/30/lessons/60062

# 해답
'''
from itertools import permutations

def solution(n, weak, dist):
    length = len(weak) #기본 한바퀴 (길이를 2배로 늘리는것은 허구의 길이기때문에 원래 길이 저장해둬야함)
    for i in range(length):
        weak.append(weak[i]+n)
    answer = len(dist) + 1  #친구 수보다 하나 더 많은 수로 초기화 (왜? -> 친구수의 최소값을 구해야하니까 그런것같은디... 마지막에 min쓸라고)
    for start in range(length):
        for friends in list(permutations(dist, len(dist))):
            count = 1
            posistion = weak[start] + friends[count - 1]  #출발 위치에서 순열의 0번째 친구가 끝까지 이동했을 경우의 위치
            for index in range(start, start + length):  # length는 기본 한바퀴
                if posistion < weak[index]:
                    count += 1
                    if count > len(dist):
                        break
                    posistion = weak[index] + friends[count - 1]
            answer = min(answer, count)
    if answer > len(dist):
        return -1
    
    return answer
    '''
        
# 큐를 사용한 구현 (다른 풀이 가져옴)
# 다음에 다시 이해하자... ㅏ어렵다...
# filter도 쓰네... 다음에 다시보자...
"""
from collections import deque

def solution(n, weak, dist):
    dist.sort(reverse=True)
    q = deque([weak])  #이게 무슨의미일까... -> deque([[1, 2, 3, 4, 5]])가 print되는데
    visited = set()  #집합 자료형
    visited.add(tuple(weak))
    for i in range(len(dist)):
        d = dist[i]
        for _ in range(len(q)):  #len = 1이야
            current = q.popleft()  #list뭉텅이 출력
            '''  #원래 풀이에는 없었지만 누군가가 반례를 하나 들어서 이거 추가하면 될듯 하면서 첨언함 (그래서 일단 적어둠)
            if d>=n-1: 
                return i+1
            '''
            for p in current:
                l = p
                r = (p + d) % n
                if l < r:
                    temp = tuple(filter(lambda x: x < l or x > r, current))
                else:
                    temp = tuple(filter(lambda x: x < l and x > r, current))

                if len(temp) == 0:
                    return (i + 1)
                elif temp not in visited:
                    visited.add(temp)
                    q.append(list(temp))
    return -1


# 뭐나오나 확인차 작성
weak = [1,2,3,4,5]
q = deque([weak])
visited = set()  #집합 자료형
visited.add(tuple(weak))
current = q.popleft()
print(current)
"""


# III. DFS/BFS 유형
# 15. 특정 거리의 도시 찾기
# https://www.acmicpc.net/problem/18352
'''
n, m, k, x = map(int, input().split())

graph = []

for _ in range(n+1):
    graph.append([])

for i in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)

visited = [False] * m
dist = [0] * n

# 아 구현 못하겠어,........ 하 대가리... 안굴러가...
def dfs(graph, k, x, visited, dist):
    if visited[k] != False:
        continue
    visited[k] = False
    for i in graph[k]:
        visited[i] = False
        dist[i] += 1
        dfs(graph, i, x, visited, dist)

    return dist

answer = []
answer = dfs(graph, k, x, visited, dist)

# 없으면 -1 print,... 
if answer != k:
    print(-1)
else:
    for i in range(len(answer)):
        if dist[i] == k:
            print(i)
'''
# 그래프에서 모든 간선의 비용이 동일할 때 'BFS'사용

# BFS란걸 알고난뒤 재풀이
# 답은 잘 나오는데... 시간초과뜸....
'''
from collections import deque

n, m, k, x = map(int, input().split())
graph = []

for _ in range(n+1):
    graph.append([])

for i in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)


def bfs(graph, k, x):
    dist = [300000] * (n+1)
    queue = deque()
    queue.append(x)
    count = 0
    while queue:
        a = queue.popleft()
        count += 1
        for i in graph[a]:
            queue.append(i)
            dist[i] = min(count, dist[i])
    return dist

d = bfs(graph, k, x)

# 이부분을 좀 줄일 수 있는 방법이 있을 것 같은데...
if k not in d:
    print(-1)
else:
    for i in range(len(d)):
        if d[i] == k:
            print(i)
'''

# 해답
# 이것도 시간초과가 뜨네... 미치겠네..... 아니 답인데 시간초과가 뜨는거 실화?
'''
import sys
from collections import deque

n, m, k, x = map(int, sys.stdin.readline().split())
graph = [[] for _ in range(n+1)]  #한 줄로 줄일 수 있음

for _ in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)

dist = [-1] * (n+1)
dist[x] = 0

qu = deque([x])
while qu:
    now = qu.popleft()
    for next in graph[now]:
        if dist[next] == -1:
            dist[next] = dist[now] + 1
            qu.append(next)

check = False
for i in range(1, n+1):
    if dist[i] == k:
        print(i)
        check = True

if check == False:
    print(-1)
'''


# 16. 연구소
# https://www.acmicpc.net/problem/14502  (백준에 등록은 하지 않았다)

# 해답
'''
n, m = map(int, input().split())
temp = [[0] * m for _ in range(n)]  #벽을 설치한 뒤의 리스트

data = []  #초기 맵 리스트
for _ in range(n):
    data.append(list(map(int, input().split())))

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

result = 0

def virus (x, y):  #근데 이거 재귀함수 넣는걸 좀 변경시켜서 (x+1, y), (x-1, y)이런식으로 넣어도 될 것 같다
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
    
        if nx >= 0 and nx < n and ny >= 0 and ny < m:
            if temp[nx][ny] == 0:
                temp[nx][ny] = 2
                virus(nx, ny)

def get_score():
    score = 0
    for i in range(n):
        for j in range(m):
            if temp[i][j] == 0:
                score += 1
    return score

def dfs(count):
    global result
    if count == 3:
        for i in range(n):
            for j in range(m):
                temp[i][j] = data[i][j]
        for i in range(n):
            for j in range(m):
                if temp[i][j] == 2:
                    virus(i, j)
        result = max(result, get_score())
        return

    for i in range(n):
        for j in range(m):
            if data[i][j] == 0:
                data[i][j] = 1
                count += 1
                dfs(count)
                data[i][j] = 0
                count -= 1

dfs(0)
print(result)
'''


# 17. 경쟁적 전염
# https://www.acmicpc.net/problem/18405

# bfs를 사용하여 할 수 있는 데까지 증식시키는 것은 했는데... 시간 타임정해서 출력하는건 못했다...
'''
from collections import deque

n, k = map(int, input().split())
# temp = [[0] * n for _ in range(n)]  필요할까...?

data = []
for _ in range(n):
    data.append(list(map(int, input().split())))

s, x, y = map(int, input().split())

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

q = deque()

for virus in range(1, k+1):
    for i in range(n):
        for j in range(n):
            if data[i][j] == virus:
                q.append((i, j, virus))

# q = sorted(q, key=lambda x:x[2]) -> 리스트로 변환돼


# 바이러스 개수는 한정되어있으니까 바이러스 종류가 바뀌는걸 카운트해서 한바퀴 다 돌면 시간이 하나 증가하도록 만들고자 했다(하지만 실패했다...)
fv = q[0][2]
length = len(q)
count = 0
t = 0

while q:
    X, Y, V = q.popleft()
    if fv != V:
        count += 1
        fv = V
    if t == s:
        break
    for i in range(4):
        nx = X + dx[i]
        ny = Y + dy[i]

        if nx < 0 or ny < 0 or nx >= n or ny >= n:
            continue
        if data[nx][ny] == 0:
            data[nx][ny] = V
            q.append((nx, ny, V))
    if count == length:
        t += 1
        count = 0
    

print(t)
print(data)
print(data[x-1][y-1])
'''

# 해답 (시간 관련과 내가 짰던 코드의 정보 받기과정을 줄인것 제외 그냥 복붙 -> 나름 선방했어! ^!^)
'''
from collections import deque

n, k = map(int, input().split())
graph = []  #전체 보드의 데이터
data = []

# 굳이 for문 한번 더 써서 시간낭비 하지말고 data받을 때 한번에 바이러스 위치도 같이 받자
# 시간도 큐 계산에 넣어버림으로써 내가 고민했었던 시간관련 계산 해결
for i in range(n):
    graph.append(list(map(int, input().split())))
    for j in range(n):
        if graph[i][j] != 0:
            data.append((graph[i][j], 0, i, j))
        
data.sort() #맨 처음 요소가 바이러스 숫자이기때문에 따로 lamda안해도 정렬됨
# sort는 리스트 정렬로 우리는 큐가 필요하니까 큐로 형변환 해야한다
q = deque(data)

s, x, y = map(int, input().split())

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

while q:
    V, S, X, Y = q.popleft()

    if S == s:
        break

    for i in range(4):
        nx = X + dx[i]
        ny = Y + dy[i]

        if nx < 0 or ny < 0 or nx >= n or ny >= n:
            continue
        if graph[nx][ny] == 0:
            graph[nx][ny] = V
            q.append((V, S+1, nx, ny))

    
print(graph[x-1][y-1])
'''



# 18. 괄호 변환
# https://school.programmers.co.kr/learn/courses/30/lessons/60058


# 구현은 했는데... 처음부터 '올바른 괄호 문자열'이 들어왔을 경우를 처리 못함...
# 기능은 비슷한데말이야...
"""
p = "(()())()"

def solution(p):
    if p == '':
        return p
    
    count = 0
    answer = ''
    check = 0
    S = 1
    u = ''
    v = ''
    plist = [p[0]]
    
    if p[0] == '(':
        check = 1

    # 나머지는 괜찮은데 인덱싱하는 방법에 문제가 있어서 처음부터 올바른 괄호 문자열이 들어왔을 경우 오류가 나는거였다
    # 시작이 ( 이고 스택이 비었을 경우 슬라이싱을 하지 않고 그냥 return해야하는데, S+1이라고 슬라이싱을 해버려서 (( / )())() 이렇게 두개로 나뉘어져버림
    for i in range(1, len(p)):
        if p[0] != p[i] and plist:
            plist.pop()
        elif p[0] == p[i]:
            plist.append(p[i])
        elif p[0] != p[i] and not plist:
            S = i - 1
            break

    u = p[:S+1]
    v = p[S+1:]
    print(u)
    print(v)
    U = ''

    if check == 1 and not plist:
        A = solution(v)
    else:
        answer = '('
        v = solution(v)
        answer = answer + v + ')'
        for i in range(1, len(u)-1):
            if u[i] == '(':
                U += ')'
            else:
                U += '('
        answer += U
        return answer

    answer = u + A
    
    return answer

print(solution(p))
"""


# 해답
"""
# '균형잡힌 괄호 문자열'의 인덱스 반환
def balanced_index(p):
    count = 0
    for i in range(len(p)):
        if p[i] == '(':
            count += 1
        else:
            count -= 1
        if count == 0:
            return i


# '올바른 괄호 문자열'인지 판단
def check_proper(p):
    count = 0  #왼쪽 괄호의 개수
    for i in p:
        if i == '(':
            count += 1
        else:
            if count == 0:
                return False
            count -= 1
    return True

'''
def solution(p):
    answer = ''

    if p == '':
        return p
    
    count = 0
    check = 0
    S = 1
    u = ''
    v = ''
    plist = [p[0]]
    

    for i in range(1, len(p)):
        if p[0] != p[i] and plist:
            plist.pop()
            count += 1
        elif p[0] == p[i]:
            plist.append(p[i])
            count += 1
        elif p[0] != p[i] and not plist:
            S = i - 1
            break

    u = p[:S+1]
    v = p[S+1:]

    U = ''

    if check_proper(u):
        answer = u + solution(v)
        

    else:
        answer = '('
        v = solution(v)
        answer = answer + v + ')'
        for i in range(1, len(u)-1):  
            if u[i] == '(':
                U += ')'
            else:
                U += '('
        answer += U
    
    return answer
'''

def solution(p):
    count = 0
    answer = ''
    if p == '':
        return answer

    index = balanced_index(p)
    u = p[:index+1]
    v = p[index+1:]

    # 이 밑부분은 내가 작성한걸로 해도 ok 
    if check_proper(u):
        answer = u + solution(v)
    else:
        answer = '('
        answer += solution(v)
        answer +=')'
        u = list(u[1:-1])
        for i in range(len(u)):
            if u[i] == '(':
                u[i] = ')'
            else:
                u[i] = '('
        answer += "".join(u)

print(solution(p))
"""


# 19. 연산자 끼워 넣기
# https://www.acmicpc.net/problem/14888

# 잘 나오긴 하는데 백준에서 시간초과 뜬다 -> 당연하지,.. for문에 if까지 겁나 써댔으니가...
"""
from itertools import permutations
from collections import deque

n = int(input())
numbers = list(map(int, input().split()))
e = list(map(int, input().split()))
E = ['+', '-', '*', '//']
s = []  #개수까지 고려 해서 넣은 연산자
result = []  #모든 계산값 담기

for i in range(4):
    for j in range(e[i]):
        s.append(E[i])

q = deque(permutations(s, n-1))

while q:
    m = 1
    S = q.popleft()
    x = numbers[0]
    for i in S:
        if m < n:
            y = numbers[m]
            if i == '+':
                x = x + y
                m += 1
            elif i == '-':
                x = x - y
                m += 1
            elif i == '*':
                x = x * y
                m += 1
            elif i == '//':
                # 양수로 바꾼 뒤 몫을 취하고, 그 몫을 음수로 바꾼 것과 같다 (음수를 양수로 나눌 때)
                if x < 0 and y > 0:
                    x = -x
                    x = x // y
                    x = -x
                    m += 1
                else:
                    x = x // y
                    m += 1
    result.append(x)

print(max(result))
print(min(result))
"""

    
# 해답
# 아니 책에서는 해답으로나왔는데 답안으로나왔는데 백준에서는 틀렸데요 -> Max와 Min의 초기화 값인 1e9를 수정해야한다

# 1e9는 실수다. 문제에서 결과값은 -1e9 <= res <= 1e9이기 때문에 최댓값, 최솟값이 1e9, -1e9라면 Max, Min이 갱신되지 않아 실수로 출력된다
# int(1e9)로 정수로 변환해주거나 1e9 + 1과 같이 문제에서 나올 수 없는 값을 초기값으로 사용해 주어야 한다.
# -1e9, 1e9까지 범위라서 만약 1e9가 나온다면 갱신이 되지 않으니 1e9까지 포함시키는 더 큰 수로 초기화를 해야한다 라는것 같다.
# 백준에서는 이런 오류가 종종 난다고 하는데 그래서 초기부터 초기화를 1e10으로 잡는 습관을 가지자 
'''
n = int(input())
data = list(map(int, input().split()))
add, sub, mul, div = map(int, input().split())

Max = -1e10
Min = 1e10

def dfs(i, now):
    global Max, Min, add, sub, mul, div
    if i == n:
        Max = max(Max, now)
        Min = min(Min, now)
    else:
        # 각각의 연산자에 대한 재귀 함수 호출이 별도로 이루어짐
        if add > 0:
            add -= 1
            dfs(i+1, now + data[i])
            add += 1  # 다른 경우의 수에서 해당 연산자를 다시 사용할 수 있도록 함
        if sub > 0:
            sub -= 1
            dfs(i+1, now - data[i])
            sub += 1
        if mul > 0:
            mul -= 1
            dfs(i+1, now * data[i])
            mul += 1
        if div > 0:
            div -= 1
            dfs(i+1, int(now / data[i]))  #int(now/data[i])와 //(몫만 걸러내는 연산자)와 다르네...? 이게 그 세부조건(C++연산)에 맞냐 아니냐를 가르는 것같은데...
            div += 1
            
dfs(1, data[0])

print(Max)
print(Min)
'''


# 20. 감시 피하기
# https://www.acmicpc.net/problem/18428 (백준에 등록하지는 않았다.)

#temp없이, 있이 둘다 해봣는데 함수 호출뒤에 아무것도 안일어나.. 오류도 안나고 yes,no print도 안되고... 왜지?
# 문제는 있겠지... 아 악 근데 오류는 떠야지ㅠ
'''
N = int(input())
data = []
temp = [['X'] * N for _ in range(N)]  #이게필요할까...?
# print(temp)

for _ in range(N):
    data.append(list(input().split()))
# print(data)

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

# 이게 문제인것 같긴 한데...
def check(data):
    for i in range(N):
        for j in range(N):
            if data[i][j] == 'T':
                for k in range(4):
                    while 0 <= i and i < N and 0 <= j and j < N:
                        nx = i + dx[k]
                        ny = j + dy[k]
                        if data[nx][ny] == 'O':
                            break
                        elif data[nx][ny] == 'S':
                            return False
                        i = nx
                        j = ny
                        
    return True

def dfs(count):
    if count == 3:
        for i in range(n):
            for j in range(m):
                temp[i][j] = data[i][j]
        if check(temp):
            print("YES")
        else:
            print("NO")
    return

    for i in range(N):
        for j in range(N):
            if data[i][j] == 'X':
                data[i][j] == 'O'
                count += 1
                dfs(count)
                data[i][j] = 'X'
                count -= 1

dfs(0)
'''


# 해답
# DFS안쓰고 조합 라이브러리 썼어.... 난 DFS로 푸는게 알고싶은데...
"""
from itertools import combinations

N = int(input())
data = []
teacher = []
space = []

for i in range(N):
    data.append(list(input().split()))
    for j in range(N):
        if data[i][j] == 'T':
            teacher.append((i, j))
        if data[i][j] == 'X':
            space.append((i, j))
        
def check (x, y, direction):
    if direction == 0:
        while y >= 0:
            if data[x][y] == 'S':
                return True
            if data[x][y] == 'O':
                return False
            y -= 1
    if direction == 1:
        while y < N:
            if data[x][y] == 'S':
                return True
            if data[x][y] == 'O':
                return False
            y += 1

    if direction == 2:
        while x >= 0:
            if data[x][y] == 'S':
                return True
            if data[x][y] == 'O':
                return False
            x -= 1

    if direction == 3:
        while x < N:
            if data[x][y] == 'S':
                return True
            if data[x][y] == 'O':
                return False
            x += 1
    return False

def process():
    for x, y in teacher:
        for i in range(4):
            if check(x, y, i):
                return True
    return False

find = False

for wall in combinations(space, 3):
    for x, y in wall:
        data[x][y] = 'O'
    if not process():
        find = True
        break
    for x, y in wall:
        data[x][y] = 'X'

if find:
    print("YES")
else:
    print("NO")
"""



# 21. 인구이동
# https://www.acmicpc.net/problem/16234 (백준에 등록은 하지 않음)

# 기능은 대충 구현을 했는데...
# 문제 1. 각 연결된 나라끼리 계산을 한 것이 아니라 총 모든 나라와 (국경선이 열려있지 않아도) 평균을 계산함
# 문제 2. DFS/BFS사용을 안했는데...?
"""
n, l, r = map(int, input().split())

data = []
for _ in range(n):
    data.append(list(map(int,input().split())))


dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
R = 0
'''
sub = []  #각 요소들의 좌표와 빼기 정보
count = 0  #해당 좌표들의 개수
check = []  #해당되는 좌표들


for i in range(n):
    for j in range(n):
        temp = [(i, j), -1, -1, -1, -1]
        for k in range(4):
            nx = i + dx[k]
            ny = j + dy[k]
            if 0 <= nx and nx < n and 0 <= ny and ny < n:
                temp[k+1] = abs(data[i][j] - data[nx][ny])
        sub.append(temp)



for i in sub:
    for j in range(1, 5):
        if l <= i[j] and i[j] <= r:
            count += 1
            check.append(i[0])
            break



for i in check:
    x, y = i[0], i[1]
    result += data[x][y]

result = result // count

for i in check:
    x, y = i[0], i[1]
    data[x][y] = result
'''

            
while True:
    result = 0  #더해서 나눈 계산한 숫자
    sub = []  #각 요소들의 좌표와 빼기 정보
    count = 0  #해당 좌표들의 개수
    check = []  #해당되는 좌표들
    for i in range(n):
        for j in range(n):
            temp = [(i, j), -1, -1, -1, -1]
            for k in range(4):
                nx = i + dx[k]
                ny = j + dy[k]
                if 0 <= nx and nx < n and 0 <= ny and ny < n:
                    temp[k+1] = abs(data[i][j] - data[nx][ny])
            sub.append(temp)

    for i in sub:
        for j in range(1, 5):
            if l <= i[j] and i[j] <= r:
                count += 1
                check.append(i[0])
                break
    print(check)
    if not check:
        break
    else:
        for i in check:
            x, y = i[0], i[1]
            result += data[x][y]

        result = result // count

        for i in check:
            x, y = i[0], i[1]
            data[x][y] = result
        print(data)
        R += 1

print(R)
"""

# 해답
# 한 나라에서부터 '연결된' 나라들의 국경선을 알아야하니까 BFS가 적합
"""
from collections import deque
n, l, r = map(int, input().split())

data = []
for _ in range(n):
    data.append(list(map(int,input().split())))

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
result = 0

def process(x, y, index):
    united = []
    united.append((x, y))
    q = deque()
    q.append((x, y))
    union[x][y] = index  #현재 연합의 번호 할당
    summary = data[x][y]
    count = 1
    while q:
        x, y = q.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0 <= nx < n and 0 <= ny < n and union[nx][ny] == -1:
                if l <= abs(data[nx][ny] - data[x][y]) <= r:
                    q.append((nx, ny))
                    union[nx][ny] = index
                    summary += data[nx][ny]
                    count += 1
                    united.append((nx, ny))
    for i, j in united:
        data[i][j] = summary // count
    return

total_count = 0

while True:
    union = [[-1] * n for _ in range(n)]
    index = 0
    for i in range(n):
        for j in range(n):
            if union[i][j] == -1:
                process(i, j, index)
                index += 1
    if index == n * n:
        break
    total_count += 1

print(total_count)
"""



# 22. 블록 이동하기
# https://school.programmers.co.kr/learn/courses/30/lessons/60063  (프로그래머스에 등록은 하지 않음)

# 와 감이 1도 안잡힌다... 역시 난도 3이라서 그런가...
# 해답
"""
from collections import deque

board = [[0, 0, 0, 1, 1],[0, 0, 0, 1, 0],[0, 1, 0, 1, 1],[1, 1, 0, 0, 1],[0, 0, 0, 0, 0]]


def get_next_pos(pos, board):
    next_pos = []
    pos = list(pos)
    pos1_x, pos1_y, pos2_x, pos2_y = pos[0][0], pos[0][1], pos[1][0], pos[1][1]
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    for i in range(4):
        nx_pos1_x, nx_pos1_y, nx_pos2_x, nx_pos2_y = pos1_x + dx[i], pos1_y + dy[i], pos2_x + dx[i], pos2_y + dy[i]
        if board[nx_pos1_x][nx_pos1_y] == 0 and board[nx_pos2_x][nx_pos2_y] == 0:
            next_pos.append({(nx_pos1_x, nx_pos1_y), (nx_pos2_x, nx_pos2_y)})
    
    if pos1_x == pos2_x:
        for i in [-1, 1]:
            if board[pos1_x + i][pos1_y] == 0 and board[pos2_x + i][pos2_y] == 0:
                next_pos.append({(pos1_x, pos1_y), (pos1_x + i, pos1_y)})
                next_pos.append({(pos2_x, pos2_y), (pos2_x + i, pos2_y)})
    elif pos1_y == pos2_y:
        for i in [-1, 1]:
            if board[pos1_x][pos1_y + i] == 0 and board[pos2_x][pos2_y + i] == 0:
                next_pos.append({(pos1_x, pos1_y), (pos1_x, pos1_y + i)})
                next_pos.append({(pos2_x, pos2_y), (pos2_x, pos2_y + i)})
    # print(next_pos)
    return next_pos



def solution(board):
    n = len(board)
    new_board = [[1] * (n+2) for _ in range(n+2)]
    for i in range(n):
        for j in range(n):
            new_board[i+1][j+1] = board[i][j]

    q = deque()
    visited = []
    pos = {(1, 1), (1, 2)}
    q.append((pos, 0))
    visited.append(pos)
    while q:
        pos, time = q.popleft()
        if (n, n) in pos:
            return time
        
        for next_pos in get_next_pos(pos, new_board):
            if next_pos not in visited:
                q.append((next_pos, time + 1))
                visited.append(next_pos)
    return 0



print(solution(board))
"""



# IV. 정렬
# 23. 국영수 (백준에 등록하진 않았다)
# https://www.acmicpc.net/problem/10825
# 와... 꽤 까다롭네... 이걸 못푼다고...? 실버4인데...?

# 국어대로 계수정렬해서 받은다음에 list의 요소 개수가 2가 넘으면 영어로 정렬, 그뒤는 다시 생각해보려고했는데, 아닌것 같아... 영어 sort정렬도 인덱스 3개 필요해...
"""
N = int(input())
student = []
score = [[] for _ in range(101)]

for _ in range(N):
    t = tuple(input().split())
    score[int(t[1])].append(t)

for i in range(100, 1, -1):
    if len(score[i]) >= 2:
        score[i].sort(key=score[i][2])
    elif score[i]:
        print(score[i][0][0])
    else:
        continue
"""

# 해답
"""
N = int(input())
student = []

for _ in range(N):
    student.append(input().split())  #굳이 튜플로 안바꿔도 되긴 되더라. 이렇게 입력하면 리스트로 받아서 2차원 리스트 된다

student.sort(key= lambda x: (-int(x[1]), int(x[2]), -int(x[3]), x[0]))  #잘익혀두기.. 미쳤어... lambda익혀두자...
# sort 메서드는 key를 사용하여 조건을 설정할 수도 있고, 정렬시키면 기본적으로 튜플(리스트)를 구성하는 원소 순서에 맞게 정렬 시킨다. (그냥 기본적으로 앞이 같으면 뒤 원소, 뒤원소 이렇게 넘어간다는 소리)

for i in student:
    print(i[0])
"""


# 24. 안테나
# https://www.acmicpc.net/problem/18310 (백준에 등록은 하지 않았다)


# 백준에서 시간 초과 떠......... 환장하겠넹 ^!^
"""
N = int(input())

numbers = set(map(int, input().split()))

numbers = list(numbers)
numbers.sort()
N = len(numbers)

Min = 1e10
result = 0

for i in range(N):
    temp = 0
    for j in range(N):
        temp += abs(numbers[i] - numbers[j])
    if Min > temp:
        Min = temp
        result = numbers[i]

print(result)
"""

# 해답
# 대가리를 좀 더 굴려보자... 너무 쉬워서 좀 해탈한걸...
# 굳이 일일이 다 계산할 필요 없고, 다른 방식, 관점으로 생각하면 더 쉽게 풀 수 있었어... 모든 받은 정보들은 쓸모없지 않아...
"""
N = int(input())

numbers = lsit(map(int, input().split()))
numbers.sort()

print(numbers[(n - 1) // 2])
"""


# 25. 실패율
# https://school.programmers.co.kr/learn/courses/30/lessons/

N = 5
stages = [2, 1, 2, 6, 2, 4, 3, 3]


# 항상 예외처리 주의
# 처음에 런타임 에러 - '모든 수를 0으로는 나눌수 없다' 나왔다
# 조건 잘 읽고 예외처리 확실하게... 런타임에러에도 여러 종류가 있다.
# '스테이지에 도달한 유저가 없는 경우 해당 스테이지의 실패율은 0으로 정의한다'는 조건 -> 런타임에러 핵심 해결책이었다
"""
def solution(N, stages):
    count = [0 for _ in range(N+2)]

    for i in stages:
        count[i] += 1

    temp = {}

    for i in range(1, N+1):
        sum = 0
        for j in range(i, N+2):
            sum += count[j]
        if sum == 0:
            temp[i] = 0
        else:
            temp[i] = count[i] / sum

    result = sorted(temp.items(), key= lambda x: x[1], reverse= True)
    answer = []

    for i in result:
        answer.append(i[0])

    return answer
"""

# 다른 답안
"""
def solution(N, stages):

    answer = []
    length = len(stages)

    for i in range(1, N+1):
        count = stages.count(i)

        if length == 0:
            fail = 0
        else:
            fail = count/length
        
        answer.append((i, fail))
        length -= count

    answer = sorted(answer, key= lambda x: x[1], reverse= True)

    answer = [i[0] for i in answer]
    return answer


print(solution(N,stages))
"""


# 26. 카드 정렬하기
# https://www.acmicpc.net/problem/1715

# 답은 잘 나오는데 시간초과 뜬다... 어쩐지 이게 골4인데 이렇게 쉽게 풀릴리가 없다고 생각했어ㅋㅋㅋㅋ 하.........
# 시간 초과 뜨는 이유: 매번 배열을 정렬하고 0번째 값을 빼주는 것으로는 시간초과가 불가피 => 매번 정렬 하지 말아야한다... 그럼... 조합?
"""
N = int(input())
numbers = []
result = 0

for _ in range(N):
    numbers.append(int(input()))

for _ in range(N-1):
    numbers.sort()
    fir = numbers.pop(0)
    sec = numbers.pop(0)
    temp = fir + sec
    numbers.append(temp)
    result += temp

print(result)
"""
# 우선순위 큐 -> 원소를 넣다 빼는것만으로도 정렬된 결과를 얻을 수 있다 => sort로 인한 시간초과를 해결할 수 있다.
# 리스트의 pop과 append, sort를 heapq에 맞게 변형만 시켰다. 나름 알고리즘생각은 나쁘지 않았나봐
"""
import heapq

N = int(input())
numbers = []
result = 0

for _ in range(N):
    heapq.heappush(numbers, int(input()))

while len(numbers) != 1:
    fir = heapq.heappop(numbers)
    sec = heapq.heappop(numbers)
    temp = fir + sec
    heapq.heappush(numbers, temp)
    result += temp

print(result)
"""


# V. 이진 탐색
# 27. 정렬된 배열에서 특정 수의 개수 구하기
# 조건 1. 1 <= N <= 1,000,000 -> N의 수가 엄청 많고, 원소 값, 숫자의 범위가 넓다 => 이진탐색으로 풀어야 할 가능성이 있다
# 조건 2. O(log N)의 시간복잡도 알고리즘으로 풀어라

# 내가 푼 답인데, 추가적인 해답으로 bisect을 사용해서 그냥 함수로 묶어버린 해답이 있었다.
"""
from bisect import bisect_left, bisect_right

n, x = map(int, input().split())
numbers = list(map(int, input().split()))

result = bisect_right(numbers, x) - bisect_left(numbers, x)

if result == 0:
    print(-1)
else:
    print(result)
"""

# 추가적인 해답
# 이진탐색으로 문제 풀기
# 정렬되어서 수열이 들어오기 때문에 마지막원소 위치와 첫원소 위치의 차이가 해당 원소의 개수가 된다
# 원소의 개수를 세는 함수, 첫번째 원소 위치 찾는 함수, 마지막 원소 위치 찾는 함수 총 3개의 함수를 구현한다
"""
def count_by_value(array, value):
    n = len(array)

    # 마지막이 n-1인 이유는 인덱스가 매개변수로 요구하기 때문
    a = first(array, value, 0, n-1)

    if a == None:
        return 0

    b = last(array, value, 0, n-1)

    if b == None:
        return 0

    return b - a + 1


def first(array, target, start, end):
    if start > end:
        return None
    mid = (start+end) // 2
    # 0을 넣는 이유는 mid가 끝과 끝일때 고려
    if (mid == 0 or array[mid-1] < target) and array[mid] == target:
        return mid
    # 작거나 같을 때로 나누는 것 주의
    elif array[mid] >= target:
        return first(array, target, start, mid-1)
    else:
        return first(array, target, mid+1, end)


def last(array, target, start, end):
    if start > end:
        return None
    mid = (start+end) // 2
    # n-1을 넣는 이유는 mid가 끝과 끝일때 고려
    if (mid == n-1 or array[mid+1] > target) and array[mid] == target:
        return mid
    # 클때만 고려하는 것 주의 (같은 것이 이제 오른쪽의 끝 원소일 수 있기 때문)
    elif array[mid] > target:
        return last(array, target, start, mid-1)
    else:
        return last(array, target, mid+1, end)

n, x = map(int, input().split())
numbers = list(map(int, input().split()))

result = count_by_value(numbers, x)

if result == 0:
    print(-1)
else:
    print(result)
"""

# 28. 고정점 찾기
# O(logN)의 시간 복잡도로 만들어야한다 -> 이진탐색, 이진탐색 라이브러리 bisect 사용해야할 듯 (근데 어떻게...?)
# => 인덱스 값 > 요소 값 -> 오른쪽, 더 큰 부분 탐색
# => 인덱스 값 < 요소 값 -> 왼쪽, 더 작은 부분 탐색
"""
n = int(input())
numbers = list(map(int, input().split()))

def binary_search(numbers, start, end):
    if start > end:
        return None
    mid = (start + end) // 2
    if numbers[mid] == mid:
        return mid
    elif mid > numbers[mid]:
        return binary_search(numbers, mid+1, end)
    else:
        return binary_search(numbers, 0, mid-1)

result = binary_search(numbers, 0, n-1)

if result != None:
    print(result)
else:
    print(-1)
"""

# 29. 공유기 설치
# https://www.acmicpc.net/problem/2110  백준에 등록은 하지 않음

# 답은 나오는데 백준에서 메모리 초과 나옴...ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 하놔ㅠ
"""
from bisect import bisect_right
from itertools import combinations

n, c = map(int, input().split())
house = []
for _ in range(n):
    x = int(input())
    y = bisect_right(house, x)
    house.insert(y, x)

result = list(combinations(house, c))

answer = -1e10
for i in result:
    temp = 1e10
    for j in range(c-1):
        t = i[j+1] - i[j]
        temp = min(temp, t)
    answer = max(answer, temp)

print(answer)
"""


# 해답
# 각 집의 좌표가 (탐색 범위가) 최대 10억 이므로 이진 탐색을 이용하면 문제를 해결할 수 있다.
"""
n, c = map(int, input().split(' '))

array = []
for _ in range(n):
    array.append(int(input()))
array.sort()

start = 1
end = array[-1] - array[0]  # 가능한 최대 거리
result = 0

while(start <= end):
    mid = (start + end) // 2  # mid는 가장 인접한 두 공유기 사이의 거리(gap)를 의미
    value = array[0]
    count = 1
    # 현재의 mid의 값을 이용해 공유기를 설치
    for i in range(1, n):  # 앞에서부터 차근차근 설치
        if array[i] >= value + mid:
            value = array[i]
            count += 1
    if count >= c:
        start = mid + 1
        result = mid  # 최적의 결과를 저장
    else:
        end = mid -1

print(result)
"""


# 30. 가사 검색 (고난도 별 3개...)
# https://school.programmers.co.kr/learn/courses/30/lessons/60060  프로그래머스에 등록하진 않음


# 뭐라도 해보려는 발악... 결국 못했지만... 아니 알고리즘 생각도 안나... 두 리스트의 개수가 2이상 10000이하라서 양이 꽤 많단말이지... 단어 길이들도 길어...
# 그러려면 이진탐색써야할 것 같은데... 이진탐색을 어떻게 적용시킬지 생각이 나질 않아요...ㅋ..ㅋㅋ.ㅋ.ㅋ.ㅋㅋㅋㅋㅋ
words = ["frodo", "front", "frost", "frozen", "frame", "kakao"]
queries = ["fro??", "????o", "fr???", "fro???", "pro?"]

"""
def solution(words, queries):
    # 일단은 단어들 중복제거
    words = set(words)
    words = list(words)

    N = len(queries)
    wordsi = 0
    queriesj = 0

    for i in range(N):
        index = 0
        T = ""
        for l in queries[i]:
            if l != '?':
                T += l
                if index == 0:
                    index = queries[i].index(l)
        if len(words[wordsi]) == len(queries[i]) and T in queries[i] and queries[i][index] == words[wordsi][index]:
            count += 1  #근데 이거 하려면 words리스트를 while이나 아무튼 순차 탐색해야하는데 그럼 이거 취지도 안맞고~ 타임오버나고~ 와... 어카냐....

    answer = []
    return answer
"""

# 해답
# bisect라이브러리를 계속 숫자에 한 글자만 사용해서 문자도 되는 걸 잊고있었다. 문자가 뭉텡이로 주어진 것에는 항상 이유가 있을텐데...
# 문자 묶음을 가지고 bisect라이브러리를 사용할 수 있다.
# 문자인것을 사용하여 문자 아스키코드 제일 낮은 a와 높은 z를 이용해서 개수를 구할 수 있는 방법이 있다.
# 정렬요소 만큼 리스트 크기를 만들어서 정렬하는 방법 계수정렬(카운팅 정렬) 사용
# 카운팅 정렬은 데이터의 크기 범위가 제한되어 정수 형태로 표현할 수 있을 때, 가장 큰 데이터와 작은 데이터의 차이가 1,000,000을 넘지 않을 때, 동일한 값의 데이터가 여러 개 등장할 때 적합

from bisect import bisect_left, bisect_right

def count_by_range(a, left_value, right_value):
    right_index = bisect_right(a, right_value)
    left_index = bisect_left(a, left_value)
    return right_index - left_index

# 모든 단어를 길이마다 나누어서 저장하기
array = [[] for _ in range(10001)]
reversed_array = [[] for _ in range(10001)]

def solution(words, queries):
    answer = []
    for i in words:
        array[len(i)].append(i)
        reversed_array[len(i)].append(i[::-1])
    
    for i in range(10001):  # 이진 탐색을 수행하기 위해 각 단어 리스트 정렬 수행
        array[i].sort()
        reversed_array[i].sort()
    
    for i in queries:
        if i[0] != '?':  # 접미사
            res = count_by_range(array[len(i)], i.replace('?', 'a'), i.replace('?', 'z'))
        else:  # 접두사
            res = count_by_range(reversed_array[len(i)], i[::-1].replace('?', 'a'), i[::-1].replace('?', 'z'))
        answer.append(res)

    return answer

print(solution(words, queries))
