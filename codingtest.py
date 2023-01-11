# 그리디 알고리즘 - 큰 수의 법칙 (92p 실전문제)
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





