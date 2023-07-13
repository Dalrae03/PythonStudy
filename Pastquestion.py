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





