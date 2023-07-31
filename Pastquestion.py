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
"""
S = input()

# 내 풀이
'''
result = 0

frist = S[0]
for i in range(1, len(S)-1):
    s = S[i]
    if s == frist and s != S[i+1]:
        result +=1

print(result)
'''

# 다른 풀이
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
"""


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

# 내가 구현한 input정보들 받는 코드
# 뱀리스트를 만들어서 뱀이 길이가 늘어나도 뱀이 위치한 몸까지 다 기록하고자 했다.
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

