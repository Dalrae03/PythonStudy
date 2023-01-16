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




