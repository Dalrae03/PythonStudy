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