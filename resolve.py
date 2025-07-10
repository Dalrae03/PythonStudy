# 알고리즘 유형별 기출 문제 다시풀기



# 그리디 알고리즘 --------------------------------------------------------------------------

# 01. 모험가 길드
# 최댓값을 만드는거니까 최대한 한 그룹에 많은 인원을 넣어야함
# 분명 다시 풀었는데, 기록이 안남아있어서 일단 해답을 가지고옴...
'''
N = int(input())
slist = list(map(int, input().split()))
result = 0
slist.sort()

count = 0
for i in slist:
    count += 1
    if count >= i:
        result += 1
        count = 0

print(result)
'''


# 02. 곱하기 혹은 더하기 (04/10)
# 처음 다시 풀 때 이랬던것같다... 첫원소는 result에 이미 들어있어가 했었다.
'''
NList = input()
result = 0


for i in NList:
    n = int(i)
    if n==0 or n==1:
        result += n
    else:
        result *= n

print(result)
'''

# 다시푼 것 고친거
'''
NList = input()
result = int(NList[0])


for i in range(1, len(NList)):
    n = int(NList[i])
    if result<= 1 or n <= 1:
        result += n
    else:
        result *= n


print(result)
'''


# 03. 문자열 뒤집기 (04/12)
# 첫 수 기준으로 첫수와 다음수가 전환되는 개수를 세는 것을 핵심으로 둠
# 백준 1439 - 뒤집기
'''
S = input()
count = 0

for i in range(len(S)-1):
    if S[i] == S[0] and S[i+1] != S[0]:
        count += 1

print(count)
'''


# 04. 만들 수 없는 금액 (07/09)
'''
N = int(input())
money = list(map(int, input().split()))
money.sort()

# relst에 현 리스트의 수 + 모든 덧셈 경우의 수를 넣기 => 이걸 어떻게 하느냐가 문제
# 중복 제거 나열 -> set or tuple?
# 빈 수를 찾아서 출력 -> for로 하나씩 수를 증가시키면서 맞춰보기? (시간이나 메모리 제한이 괜찮을까?)
# => 애초에 이런 접근이 아님...

# 해답
# 작은 수 부터 차례로 누적 덧셈을 하면서 원소들의 대소를 비교하여 판단.
# target보다 원소의 수가 크면 만들 수 없는 수라는 판단 -> 이걸 어캐하는데...................
# target - 1 까지의 수를 만들 수 있다 라는 정의.

target = 1

for i in money:
    # 만들 수 없는 금액을 찾았을 때 반복 종료료
    if target < i:
        break
    target += i

print(target)

'''


# 05. 볼링공 고르기 (07/10)
# 딱 내가 처음 풀었던 방식으로 풀었음... M을 사용하지 않고 2중 for문을 사용하는 방식...
'''
N, M = map(int, input().split())
Balls = list(map(int, input().split()))
count = 0

for i in range(N-1):
    Balls[i]
    for j in range(i+1, N):
        if Balls[i] != Balls[j]:
            count += 1

print(count)
'''

# 다른 방법
# 다른 효과적 해결 방법. 볼링공의 각 무게마다 개수를 세서 하는 방법.
# M이 최대 10이라서 가능
N, M = map(int, input().split())
Balls = list(map(int, input().split()))

array = [0] * 11
for x in Balls:
    array[x] += 1

result = 0
for i in range(1, M+1):
    N -= array[i]
    result += array[i] * N

print(result)