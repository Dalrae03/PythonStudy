'''
name = input()
print(name)
print(type(name))
'''

# 백준 1008
'''
a, b = input().split(' ')
a, b = map(int,input().split(' ')) # 맵함수 사용
a = int(a)
b =int(b)
print(a/b)
'''

# 백준 3003
'''
chess = [1,1,2,2,2,8]
List = []

List = list(map(int, input().split(' ')))

# print(List)

# print(type(List[0]))

for i in range(len(chess)):
    List[i] = chess[i]- List[i]

print(List[0], List[1], List[2], List[3], List[4], List[5])
'''

# 2차원 리스트 배열
'''
n, m = map(int, input().split())

#1
mylist=[0 for _ in range(n)]  
for i in range(n):
	mylist[i]=list(map(int, input().split()))  
'''

# 백준 10828
'''
# ppt - 큰따옴표, 숫자에 따옴표 사용안함, if사용할때 조건부 표현식 사용-> stack 자체를 안에 무엇인가가 있다는 뜻으로 사용
import sys
N = int(sys.stdin.readline().rstrip())  #input은 시간초과
stack =[]

for _ in range(N):
    input = sys.stdin.readline().rstrip().split()
    order = input[0]
    if order == "push":
        stack.append(input[1])
    elif order == "pop":
        if (not stack):
            print(-1)
        else:
            print(stack.pop())
    elif order == "size":
        print(len(stack))
    elif order == "empty":
        print(0 if stack else 1)
    elif order == "top":
        print(stack[-1] if stack else -1)
'''

# 백준 9012
'''
# ppt
def solve(parens):
    stack =[]

    for paren in parens:
        if len(stack) != 0:
            if paren == ')':
                stack.pop()
                continue
        elif len(stack) == 0:
            if paren == ')':
                print("NO")
                return
        stack.append(paren)
    if len (stack) == 0:
        print("YES")
    else:
        print("NO")

T = int(input())

for _ in range(T):
    parens = list(input().rstrip())
    solve(parens)


# for else문 사용
import sys
N = int(input())
stack = []

for _ in range(N):
    input = sys.stdin.readline().strip()
    stack.clear()
    for i in input:
        if i == '(':
            stack.append(i)
        else:
            if stack:
                stack.pop()
            else:
                print("NO")
                break
    else:   # else가 왜 있어야할까..?
        if not stack:
            print("YES")
        else:
            print("NO")
'''


# 백준 4949

# 짝을 이루는 두 괄호가 있을 때, 그 사이에 있는 문자열도 균형이 잡혀야 한다.
# -> 여기에 꽂혀서.. 나는 괄호 앞의 띄어쓰기도 봐야하는줄... 그것만 엄청보고 종료조건을 보지않았다...
# 무조건 for쓸 생각만 했어... 주어진 반복횟수가 없는데도...
# 주어진 반복횟수가 없다면 whlie을 생각하자.
# 프린트는 잘 되는데 정답이 안떠... 왤까..? 왜틀렸냐고오옥 짱나네 => ( 하나만 넣었을경우 no가 나와야하는데 yes가 나와... 그래서 틀림.. 수정요망...
'''
while True:
    sentances = input()
    stack = []

    if sentances == '.':
        break

    for sentance in sentances:
        if sentance == '(' or sentance == '[':
            stack.append(sentance)

        elif sentance == ')':
            if len(stack) !=0 and stack[-1] == '(':
                stack.pop()
            else:
                print("no")
                break

        elif sentance == ']':
            if len(stack) !=0 and stack[-1] == '[':
                stack.pop()
            else:
                print("no")
                break

    if len(stack) == 0:
        print('yes')
'''
# 밑의 if문으로 len(stack) == 0 을 사용한다면 )가 먼저왔을 경우에 stack이 빈상태라 no와 yes가 함께 나온다.
# 따라서 len(stack) == 0 을 사용하려면 ),]가 먼저왔을때 stack에 무언갈 넣을 수 있어야한다...

# 수정완...
'''
while True:
    sentances = input()
    stack = []
    i = 0

    if sentances == '.':
        break

    for sentance in sentances:
        if sentance == '(' or sentance == '[':
            stack.append(sentance)

        elif sentance == ')':
            if len(stack) !=0 and stack[-1] == '(':
                stack.pop()
            else:
                i = 1
                break

        elif sentance == ']':
            if len(stack) !=0 and stack[-1] == '[':
                stack.pop()
            else:
                i = 1
                break

    if i == 0 and len(stack) == 0:
        print("yes")
    else:
        print("no")
'''
# true false처럼 i 를 0과 1로 판단한것


# 수정 완...
'''
while True:
    sentances = input()
    stack = []

    if sentances == '.':
        break

    for sentance in sentances:
        if sentance == '(' or sentance == '[':
            stack.append(sentance)

        elif sentance == ')':
            if len(stack) !=0 and stack[-1] == '(':
                stack.pop()
            else:
                stack.append(')')
                break

        elif sentance == ']':
            if len(stack) !=0 and stack[-1] == '[':
                stack.pop()
            else:
                stack.append(']')
                break

    if len(stack) == 0:
        print('yes')

    else:
        print('no')
'''


# 인터넷 카피
'''
while True :
    a = input()
    stack = []

    if a == "." :
        break

    for i in a :
        if i == '[' or i == '(' :
            stack.append(i)
        elif i == ']' :
            if len(stack) != 0 and stack[-1] == '[' :
                stack.pop() # 맞으면 지워서 stack을 비워줌 0 = yes
            else : 
                stack.append(']')
                break
        elif i == ')' :
            if len(stack) != 0 and stack[-1] == '(' :
                stack.pop()
            else :
                stack.append(')')
                break
    if len(stack) == 0 :
        print('yes')
    else :
        print('no')
'''


# 백준 1874
'''
N = int(input())
stack = [0] # +-할 숫자 스택
M = [] # 입력할 숫자들 리스트
j = 0
result = []

for _ in range(N):  # 입력숫자 리스트를 만듦
    M.append(int(input()))

for i in range(N):
    while stack[-1] != M[i]:
        if stack[-1] < M[i]:
            j += 1
            result.append('+')
            stack.append(j)

        elif stack[-1] > M[i]:
            del result[:]
            result.append('NO')
            break

    if stack[-1] == M[i]:
        result.append('-')
        stack.pop()

    else:
        break

for i in range(len(result)):
    print(result[i])
'''

# 백준 11866 (복습)
'''
from collections import deque
import sys

n, k = map(int, input().rstrip().split())
N = deque()
list = []

for i in range(n):
    N.append(i+1)

for _ in range(len(n)-1):
    for _ in range(k-1):
        N.append(N.popleft())
    b = N.popleft()
    list.append(b)
    
print("<", end ='')
for i in range(N-1):
    print(list[i], end = ', ')
print(list[-1], end = '>')
'''

# 백준 5597
'''
student = []
for i in range(30):
    student.append(i+1)

N = []

for i in range(28):
    N.append(int(input()))

for i in range(28):
    for j in range(30):
        if N[i] == student[j]:
            del student[j]
            break

print(student[0])
print(student[1])
'''

# 백준 11047 (그리디 알고리즘 - 동전 0)
# 첫번째 if문(if M < K:)있어서 틀렸다고 나옴... 왜...?
'''
N, K = map(int, input().split())

money = []

for _ in range(N):
    money.append(int(input()))

money.reverse()
count = 0

for M in money:
    if M <= K:  # if문의 유무의 문제가 아니라 이상 이하의 문제... 천원 남아있을때 천원짜리를 사용해야하니까... 부등호 추가
        count += K // M
        K = K % M

    if K == 0:
        break

print(count)
'''

# 첫번째 if문 없앤 버전인데 이건 또 맞았데... 왜? if문의 유무의 차이가 머야...?
'''
N, K = map(int, input().split())

money = []

for _ in range(N):
    money.append(int(input()))

money.reverse()
count = 0

for M in money:
    count += K // M
    K = K % M

    if K == 0:
        break

print(count)
'''

# 백준 1931
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
    if a >= m:
        count += 1
        m = b

print(count)
'''

# 백준 11399
'''
N = int(input())
result = 0
p = 0
number = list(map(int, input().split()))

number.sort()

for i in number:
    p += i
    result += p

print(result)
'''

# 백준 1920
# 시간 초과. 하지만 결괏값은 맞게 나옴. 아무래도 이진탐색을 사용해야할 듯
'''
N = int(input())
A = list(map(int, input().split()))
A.sort()
M = int(input())
m = list(map(int, input().split()))

for i in m:
    if i in A:
        print('1')
    else:
        print('0')
'''
'''
def search (A, i, start, end):
    while (start <= end):
        mid = (start + end) // 2
        if A[mid] > i:
            end = mid -1
        elif A[mid] < i:
            start = mid + 1
        else:
            return 1
    return None

N = int(input())
A = list(map(int, input().split()))
A.sort()
M = int(input())
m = list(map(int, input().split()))

for i in m:
    result = search(A, i, 0, N-1)
    if result == 1:
        print('1')
    else:
        print('0')
'''

# 백준 10816
# 런타임 에러... 왜자꾸나오냐 돌겟다진짜............
'''
N = int(input())
get = sorted(list(map(int, input().split())))
M = int(input())
List = list(map(int,input().split()))

def search (L, i, start, end):
    if start > end:
        return 0
    mid = (start + end) // 2
    if L[mid] < i:
        return search(L, i, mid+1, end)
    elif L[mid] > i:
        return search(L, i, start, mid-1)
    else:
        return cnt.get(i)

cnt = {}
for i in get:
    if i in cut:
        cnt[i] += 1
    else:
        cnt[i] = 1

for i in List:
    print(search(get, i, 0, len(get)-1), end=' ')
'''
'''
N = int(input())
get = list(map(int, input().split()))
M = int(input())
List = list(map(int,input().split()))

cnt = {}
for i in get:
    if i in cnt:
        cnt[i] += 1
    else:
        cnt[i] = 1

result =[0] * M
for i in range(M):
    if List[i] in cnt:
        result[i] = cnt[List[i]]

for count in result:
    print(count)
'''

# 런타임 에러
'''
N = int(input())
get = list(map(int, input().split()))
M = int(input())
List = list(map(int,input().split()))
count = 0

def search (L, i, start, end):
    while (start <= end):
        mid = (start + end) // 2
        if L[mid] < i:
            start = mid +1
        elif L[mid] > i:
            end = mid -1
        else:
            return L[start:end+1].count(i)
    return 0

get.sort()

for i in List:
    result = (search(get, i, 0, N))
    print(result, end=' ')
'''
# 런타임 에러
# 메모리제한 초과로 런타임에러 뜰 수 있다.
'''
N = int(input())
get = sorted(list(map(int, input().split())))
M = int(input())
List = list(map(int,input().split()))

def binary(n, N, start, end):
    if start > end:
        return 0
    m = (start+end)//2
    if n == N[m]:
        return N[start:end+1].count(n)
    elif n < N[m]:
        return binary(n, N, start, m-1)
    else:
        return binary(n, N, m+1, end)

n_dic = {}
for n in N:
    start = 0
    end = len(N) - 1
    if n not in n_dic:
        n_dic[n] = binary(n, N, start, end)

print(' '.join(str(n_dic[x]) if x in n_dic else '0' for x in M ))
'''

# 런타임 에러
'''
N = int(input())
get = list(map(int, input().split()))
M = int(input())
List = list(map(int,input().split()))
count = 0

def search (L, i, start, end):
    while (start <= end):
        mid = (start + end) // 2
        if L[mid] < i:
            start = mid +1
        elif L[mid] > i:
            end = mid -1
        else:
            return L[start:end+1].count(i)
    return 0

get.sort()

for i in List:
    result = (search(get, i, 0, N))
    print(result, end=' ')
'''


# 백준 4673
'''
all_numbers = set(range(1, 10001))
numbers = set()

def d ():
    for i in range(1, 10001):
        for j in str(i):
            i += int(j)
        numbers.add(i)

d ()

List = sorted(all_numbers-numbers)

for i in List:
    print(i)
'''

# 백준 2587
'''
number = []

for i in range(5):
    number.append(int(input()))

number.sort()
num = 0

for i in number:
    num += i

print(int(num / 5))
print(number[2])
'''

# 백준 2563
'''
N = int(input())
W_paper = []
paper = [[0]*100 for _ in range(100)]

for _ in range(N):
    a, b = list(map(int,input().split()))

    for i in range(a, a+10):
        for j in range(b, b+10):
            paper[i][j] = 1

count = 0
for i in range(100):
    for j in range(100):
        if paper[i][j] == 1:
            count += 1

print(count)
'''

# 백준 2750
'''
N = int(input())
numbers = []

for _ in range(N):
    numbers.append(int(input()))

numbers.sort()

for i in numbers:
    print(i)
'''

# 백준 10989
# 메모리 초과 -> 왜..? 계수정렬쓰라해서 썼는데...
# -> for문 안에서 append를 사용하면 메모리 재할당이 이루어져서 메모리를 효율적으로 사용할 수 없다.
# 따라서 메모리 초과가 일어난 듯 싶다.
# sys 안썼더니 이번엔 시간초과나서 sys로 수정
'''
import sys

N = int(sys.stdin.readline())
count = [0] * 10001

for _ in range(N):
    count[int(sys.stdin.readline())] += 1

for i in range(10001):
    if count[i] != 0:
        for j in range(count[i]):
            print(i)
'''
'''
import sys
N = int(sys.stdin.readline())
numbers = []

for _ in range(N):
    numbers.append(int(sys.stdin.readline()))

count = [0] * (max(numbers)+1)

for i in numbers:
    count[i] += 1

for i in range(len(count)):
    for j in range(count[i]):
        print(i)

# 인터넷 검색, 딕셔너리 사용
N = int(input())
numbers = []

for _ in range(N):
    numbers.append(int(input()))

frequency = {}

for num in numbers:
    if num not in frequency:
        frequency[num] = 1
    else:
        frequency[num] += 1

for num, count in frequency.items():
    for _ in range(count):
        print(num)
'''


# 백준 1181
'''
N = int(input())
word = []

for _ in range(N):
    W = input()
    if W not in word:
        word.append(W)
    else:
        continue

word.sort()
word.sort(key=len)

for i in word:
    print(i)
'''

'''
List = {}

for _ in range(N):
    W = input()
    List[W] = len(W)

L = sorted(List.items(), key = lambda x :x[1])

print(L)
'''

# 백준 1427
'''
N = input()
numbers = []

for i in N:
    numbers.append(int(i))

numbers.sort(reverse=True)

for i in numbers:
    print(i, end='')
'''

# 백준 11650
'''
N = int(input())
number = []

for _ in range(N):
    number.append(list(map(int, input().split())))

M = sorted(number, key = lambda x : x[1])
L = sorted(M, key = lambda x : x[0])

for i in range(N):
    print(L[i][0], end=' ')
    print(L[i][1])
'''

# 백준 11651
'''
N = int(input())
number = []

for _ in range(N):
    number.append(list(map(int, input().split())))

L = sorted(number, key = lambda x : x[0])
M = sorted(L, key = lambda x : x[1])

for i in range(N):
    print(M[i][0], end=' ')
    print(M[i][1])
'''

# 백준 10814
'''
N = int(input())
List = []

for _ in range(N):
    List.append(list(input().split()))


for i in range(N):
    List[i][0] = int(List[i][0])

L = sorted(List, key = lambda x : x[0])

for i in range(N):
    print(L[i][0], end=' ')
    print(L[i][1])
'''

# 백준 10872
'''
N = int(input())

def fac (N):
    if N <= 1:
        return 1
    else:
        return (N * fac(N-1))

print(fac(N))
'''

# 백준 10870
'''
N = int(input())

def fnumber (N):
    if N == 0:
        return 0
    elif N == 1:
        return 1
    else:
        return (fnumber(N-1) + fnumber(N-2))

print(fnumber(N))
'''

# 백준 25501
'''
T = int(input())
S = []
count = 0

for _ in range(T):
    S.append(input())

def recursion(s, l, r, count):
    count += 1
    if l >= r:
        return 1, count
    elif s[l] != s[r]:
        return 0, count
    else:
        return recursion(s, l+1, r-1, count)

def isPalindrome(s):
    return recursion(s, 0, len(s)-1, count)

for i in range(T):
    x, y = isPalindrome(S[i])
    print(x, end=' ')
    print(y)
'''

# 백준 1269
'''
N, M = map(int, input().split())
A = set()
B = set()

A.update(input().split())
B.update(input().split())

a = A - B
b = B - A

print(len(a)+len(b))
'''

# 백준 1764
'''
N, M = map(int, input().split())
n = []
m = []

for _ in range(N):
    n.append(input())

for _ in range(M):
    m.append(input())

A = set(n) & set(m)
B = list(A)
B.sort()

print(len(B))
for i in range(len(B)):
    print(B[i])
'''

# 백준 14425
'''
# 프린트는 잘 되는데 틀렷데......... 왤까....
N, M = map(int, input().split())
n = []
m = []

for _ in range(N):
    n.append(input())

for _ in range(M):
    m.append(input())

A = set(n) & set(m)
B = list(A)

print(len(B))
'''
'''
N, M = map(int, input().split())
n = set()
count = 0

for _ in range(N):
    n.add(input())


for _ in range(M):
    a = input()
    if a in n:
        count += 1

print(count)
'''

# 백준 1541
'''
import sys
numbers = sys.stdin.readline().rstrip().split('-')
number = []

for i in numbers:
    count = 0
    num = i.split('+')
    for j in num:
        count += int(j)
    number.append(count)

result = number[0]

for i in range(1, len(number)):
    result -= number[i]

print(result)
'''

# 백준 2805
# 메모리 초과...
'''
import sys

N, M = map(int, sys.stdin.readline().split())
wood = list(map(int, sys.stdin.readline().split()))

a = max(wood)
b = min(wood)

def search ( ma, mi, M, List ):
    mid = (a + b) // 2
    cut = [0] * len(List)
    for i in List:
        if i >= mid:
            cut.append(i - mid)
        else:
            cut.append(0)
    all = sum(cut)
    if all > M:
        search(ma, mid, M, List)
    elif all < M:
        search(mid, mi, M, List)
    else:
        return mid

result = search(a, b, M, wood)

print(result)
'''
'''
import sys

N, M = map(int, sys.stdin.readline().split())
wood = list(map(int, sys.stdin.readline().split()))

a = max(wood)  # end
b = 1  # start -> 1이 아닌 min(wood)로 하면 틀렸다고한다. 아마 min보다 더 작은경우도 있을 수 있으니까 그런것같다.

while b <= a:
    mid = (a + b) // 2
    log = 0
    for i in wood:
        if i >= mid:
            log += i - mid
    
    if log >= M:
        b = mid + 1
    else:
        a = mid - 1

print(a)
'''

# 백준 2108
'''
from collections import Counter
import sys

N = int(sys.stdin.readline().rstrip())
numbers = []

for _ in range(N):
    numbers.append(int(sys.stdin.readline().rstrip()))

numbers.sort()

cnt = Counter(numbers).most_common(2)

print(round(sum(numbers) / N))
print(numbers[(N//2)])
if len(numbers) > 1:
    if cnt[0][1] == cnt[1][1]:
        print(cnt[1][0])
    else:
        print(cnt[0][0])
else:
    print(cnt[0][0])
print(max(numbers)-min(numbers))
'''

# 백준 24060
'''
import sys

N, K = map(int, sys.stdin.readline().rsplit())
A = list(map(int, sys.stdin.readline().split()))
result = []

def merge_sort(arr):
    if len(arr) < 2:
        return arr

    mid = (len(arr)+1) // 2
    low_arr = merge_sort(arr[:mid])
    high_arr = merge_sort(arr[mid:])

    merged_arr = []
    l = h = 0
    while l < len(low_arr) and h < len(high_arr):
        if low_arr[l] < high_arr[h]:
            merged_arr.append(low_arr[l])
            result.append(low_arr[l])
            l += 1
        else:
            merged_arr.append(high_arr[h])
            result.append(high_arr[h])
            h += 1

    while l < len(low_arr):
        merged_arr.append(low_arr[l])
        result.append(low_arr[l])
        l += 1   
    while h < len(high_arr):
        merged_arr.append(high_arr[h])
        result.append(high_arr[h])
        h +=1

    return merged_arr

merge_sort(A)

if len(result) >= K:
    print(result[K-1])

else:
    print(-1)
'''

# 백준 13305
'''
import sys

N = int(input())
R = list(map(int, sys.stdin.readline().split()))
M = list(map(int, sys.stdin.readline().split()))

money = []
result = 0

low = M[0]
for i in range(len(R)):
    result += R[i] * low
    if M[i+1] < low:
        low = M[i+1]

print(result)
'''

# 백준 1966
# list는 popleft가 안돼 일단 인풋에서 이미 queue를 list로 바꿔버림...
'''
from collections import deque
import sys

test = int(input())

for _ in range(test):
    N, M = map(int, sys.stdin.readline().split())
    qu = deque(list(map(int, sys.stdin.readline().split())))
    count = 0

    while len(qu) != 0:
        m = max(qu)
        frist = qu.popleft()
        M -= 1

        if frist == m:
            count += 1
            if M < 0:
                print(count)
                break

        else:
            qu.append(frist)
            if M < 0:
                M = len(qu) - 1      
'''
# 참고전과 비슷하게 풀긴했던 내코드... 하지만 답이 1도 안나오고 받기만 잘받음... 도대체 어디가 문제일까...
# 그냥 인덱스를 쓰지말고 popleft가 조금 더 나을것같긴하다..ㅎㅎㅎㅎ
'''
from collections import deque
import sys

test = int(input())

for _ in range(test):
    N, M = map(int, sys.stdin.readline().split())
    qu = deque(list(map(int, sys.stdin.readline().split())))
    count = 0

    while len(qu) != 0:
        m = max(qu)
        # 맨앞자리가 최대값이 아닌경우
        if qu[0] != m :
            qu.append(qu.popleft())
            M -= 1
            if M < 0:
                M = len(qu) - 1
        # 앞자리가 최댓값이고 우리가 알고싶은 수일 경우
        elif m == qu[0] and M == 0:
            qu.popleft()
            count += 1
            print(count)
            break
        # 최댓값이지만 우리가알고싶은 수가 아닐경우
        else:
            qu.popleft()
            count += 1
'''

# 백준 11382
'''
numbers = list(map(int, input().split()))
print(sum(numbers))
'''

# 백준 25314
'''
N = int(input())

R = N // 4

for _ in range(R):
    print('long', end=' ')

print('int')
'''

# 백준 10810
'''
N, M = map(int, input().split())
baskit = [0 for _ in range(N)]

for _ in range(M):
    frist, last , ball = map(int, input().split())
    for i in range(frist, last+1):
        baskit[i-1] = ball

for i in baskit:
    print(i, end=' ')
'''

# 백준 11478
'''
S = input()
result = []

# 부분 문자열 구햇음.......
for i in range(len(S)):
    j = 1 + i
    start = 0
    while j <= len(S):
        result.append(S[start:j])
        j += 1
        start += 1

R = list(set(result))
print(len(R))
'''

# 백준 10798
'''
import sys
words = []
result =[]
max = 0

for _ in range(5):
    N = sys.stdin.readline().rstrip()
    words.append(N)
    len(N)
    if max < len(N):
        max = len(N)

for i in range(max):
    for j in range(5):
        if (len(words[j])-1) < i:
            pass
        else:
            result.append(words[j][i])

for i in range(len(result)):
    print(result[i], end='')
'''

# 백준 25206 - 너의 평점은
'''
import sys
subject = []
score = 0
S = 0

for i in range(20):
    subject.append(list(sys.stdin.readline().rstrip().split()))

for i in range(20):
    if subject[i][2] == 'P':
        pass
    elif subject[i][2] == 'A+':
        score += float(subject[i][1]) * 4.5
        S += float(subject[i][1])
    elif subject[i][2] == 'A0':
        score += float(subject[i][1]) * 4
        S += float(subject[i][1])
    elif subject[i][2] == 'B+':
        score += float(subject[i][1]) * 3.5
        S += float(subject[i][1])
    elif subject[i][2] == 'B0':
        score += float(subject[i][1]) * 3
        S += float(subject[i][1])
    elif subject[i][2] == 'C+':
        score += float(subject[i][1]) * 2.5
        S += float(subject[i][1])
    elif subject[i][2] == 'C0':
        score += float(subject[i][1]) * 2
        S += float(subject[i][1])
    elif subject[i][2] == 'D+':
        score += float(subject[i][1]) * 1.5
        S += float(subject[i][1])
    elif subject[i][2] == 'D0':
        score += float(subject[i][1]) * 1
        S += float(subject[i][1])
    elif subject[i][2] == 'F':
        score += float(subject[i][1]) * 0
        S += float(subject[i][1])    

result = score / S
print(round(result, 6))
'''

# 백준 10813 - 공 바꾸기
'''
N, M = map(int, input().split())

basket = [i + 1 for i in range(N)]

for _ in range(M):
    i, j = map(int, input().split())
    basket[i-1], basket[j-1] = basket[j-1], basket[i-1]

for i in basket:
    print(i, end=' ')
'''


# 백준 10811 - 바구니 뒤집기 (와 이걸 못품... / chatgpt검색 참고)
'''
N, M = map(int, input().split())

basket = [i + 1 for i in range(N)]

for _ in range(M):
    i, j = map(int, input().split())
    # 리스트 슬라이싱으로 푸는 방법
    basket[i-1:j] = basket[i-1:j][::-1]
    # reversed함수로 푸는 방법
    basket[i-1:j] = list(reversed(basket[i-1:j]))


for i in basket:
    print(i, end=' ')
'''


# 백준 1546 - 평균
'''
N = int(input())
score = list(map(int, input().split()))

M = max(score)
entire = 0

for i in score:
    entire += i / M * 100
    
print(entire/N)
'''


# 백준 27866 - 문자와 문자열
'''
S = input()
i = int(input())

print(S[i-1])
'''


# 백준 2743 - 단어 길이 재기
'''
S = input()
print(len(S))
'''


# 백준 11718 - 그대로 출력하기
# 백준의 입력값을 보고 세 줄을 한꺼번에 입력을 해야한다고 생각했었음. (한줄에 출력 하나씩이더라...)
# 그래서 엔터와 공백으로 입력이 끝났다는걸 알려야하는데 그걸 어떻게 알리지에서 막혀버림.
# 아래 코드는 첫번째 한 줄만 입력받고 프린트함. -> 결론 못풀었다는거 (망함...)
'''
import sys
S = sys.stdin.read().rstrip()
print(S)
'''
# 입력값이 없어지는 상황에 발생하는 에러를 사용해야한다 -> 몇번의 입력이 주어지는지 정해지지 않기 때문. -> break가 핵심이겠구나! while도 쓸수도! 라는 생각을...해야함...
# sys.stdin.readline()은 사용할 수 없다. EOF에러를 받을 때, 오류를 발생시키지 않고, 빈 문자열을 리턴하기 때문이다.
# (input은 EOF받을 때 EOF에러 일으킴킴)
# https://www.acmicpc.net/board/view/57241
# EOF - End Of File의 줄임말. 입력값이 없어지는 상황을 받아줌
'''
while True:
    try:
        print(input())
    except EOFError:
        break
'''

# 백준 9086 - 문자열
'''
T = int(input())
Slist = []

for i in range(T):
    Slist.append(input())

for i in range(T):
    print(Slist[i][0], Slist[i][len(Slist[i])-1], sep='')
'''

# 백준 1439 - 뒤집기
'''
S = input()
count = 0

for i in range(len(S)-1):
    if S[i] == S[0] and S[i+1] != S[0]:
        count += 1

print(count)
'''


# 백준 2444 - 별찍기
'''
N = int(input())

for i in range(1, N+1):
    star = '*' * (2*i-1)
    print(star.rjust(N+i-1, ' '))

for i in range(N-1, 0, -1):
    star = '*' * (2*i-1)
    print(star.rjust(N+i-1, ' '))
'''


# 백준 10988 - 팰린드롬인지 확인하기
# 와... 문자열 뒤집는 방법을 다시 익혀야함...
# 문자열은 바로 .reverse()사용이 안된다 -> 리스트로 변환하고 join을 사용해서 다시 결합해야함
# 기본적인 문자열 뒤집기 방법을 잊으면 다시 이 문제 보기.
'''
S = input()
S_reverse = S[::-1]

if S == S_reverse:
    print(1)
else:
    print(0)
'''


# 백준 28278 - 스택 2
# 이렇게 했는데 틀렸데... 왜? (솔직히 실버4인데 이게 맞으면 너무 쉬운편이긴한데... 왜 틀렸을까... 백준은 이유좀 알려줫음좋겟다.. 진심으로다가....)
# 출력 답안은 정해진대로 잘 나오는디....

# N = int(input()) -> 이걸 사용하면 시간초과로 틀렸다고 나온데... (아니그럼 시간초과라고해야지 틀렸습니다 왜띄우는데)
# 결론, 핵심 기능코드는 맞는데 받는게 문제였다...
# 나중에 출력값 잘 나오는데도 계속 틀렸다하고 채점시간이 오래걸리면 input을 sys.stdin.readline().rstrip()으로 바꿔보자
'''
N = int(input())
stack = list()

for _ in range(N):
    order = input()
    x = int(order[0]) #굳이 int로 안바꿔도 되긴해

    if x == 1:
        stack.append(int(order[2]))
    elif x == 2:
        if not stack:
            print(-1)
        else:
            print(stack.pop())
    elif x == 3:
        print(len(stack))
    elif x == 4:
        if not stack:
            print(1)
        else:
            print(0)
    else:
        if not stack:
            print(-1)
        else:
            print(stack[-1])
'''

# input을 sys사용한것으로, 판단을 굳이 int형이 아닌 문자열로
# 와 이것도 틀렸다는데 그 이유가 order[2]로 파싱하는게 잘못됨... -> 이렇게 하면 넣고 싶은 정수가 2자리가 넘어갈때 첫번째 숫자만 담겨서... (ex. 10을 넣었을 때 1만 들어감)
# 그래서 시간초과가 아닌 틀렸다가 뜨는거였고...
# split() 공백 기준으로 명령어를 나눠서 처리해야함
'''
import sys

N = int(sys.stdin.readline().rstrip())
stack = list()

for _ in range(N):
    order = sys.stdin.readline().rstrip().split() #split추가 -> 공백기준으로 리스트로 변경

    if order[0] == '1':
        stack.append(int(order[1])) #이것도 굳이 int로 전환이 필요없긴해
    elif order[0] == '2':
        if not stack:
            print(-1)
        else:
            print(stack.pop())
    elif order[0] == '3':
        print(len(stack))
    elif order[0] == '4':
        if not stack:
            print(1)
        else:
            print(0)
    else:
        if not stack:
            print(-1)
        else:
            print(stack[-1])
'''



# 백준 12789 - 도키도키 간식드리미
# 밑에는 실패한 코드. 할 수 있을것같았는데... 아래 코드로 하면 Sad가 나와서 실패.
# next와 stack의 비교를 stack[-1]로 해서 index error를 내지 않기 위해 ' '로 초기화를 했지만 (물론 이 방법은 매우매우 안좋다는걸 안다.)
# 백준 예시로 돌렸을 때 6번째 반복에서 마지막의 elif에 걸려 ' '를 pop할 수 있기 때문에 pop을 해버리고 빈 스택과 i의 조건이 맞아져
# Sad가 출력이 되어버리는 것...
# stack을 무조건 빈 것으로 초기화시켜서 판단하는데 사용하는 것이 당연한 것인디... 나는 혹시했지...
# 그리고 map함수를 까먹어서 일일히 int로 전환해야하는 방법을 사용했다.

'''
import sys

N = int(input())
student = sys.stdin.readline().rstrip().split()  # 리스트들로 존재 할 것임.
next = 1
stack = [' ']
i = 0

while True:
    if next != int(student[i]) and next != stack[-1]:
        stack.append(int(student[i]))
        i += 1

    elif next == int(student[i]) and next != stack[-1]:
        i += 1
        next += 1

    elif next == stack[-1] and (next != int(student[i]) or i == N):
        stack.pop()
        next += 1


    if stack[-1] == ' ' and i == N:
        print('Nice')
        break
    elif stack[-1] != ' ' and i == N:
        print('Sad')
        break
'''

# 인터넷 검색 해답
# 내가 걱정했던 stack[-1]의 인덱스 오류를 그냥 첫 요소를 stack안에 바로 넣어버리면서 해결해버림
# 그리고 일단 무조건적으로 stack에 넣고 stack에서 비교를 함.
# 발상의 전환이 필요!!!!!
'''
import sys

N = int(input())
student = list(map(int, sys.stdin.readline().rstrip().split()))  # 정수형 리스트들로 존재 할 것임
next = 1
stack = []  # list 초기화시 아무것도 없는 조건이라면 무조건 초기화도 빈 것으로. (' ' 뭐 이딴 초기화 금지)

for i in student:
    stack.append(i)
    while stack and stack[-1] == next:
        stack.pop()
        next += 1

if stack:
    print('Sad')
else:
    print('Nice')
'''



# 백준 28279 - 덱 2
'''
from collections import deque
import sys

N = int(sys.stdin.readline().rstrip())
dq = deque()

for _ in range(N):
    order = sys.stdin.readline().rstrip().split()

    if order[0] == '1':
        dq.appendleft(order[1])

    elif order[0] == '2':
        dq.append(order[1])

    elif order[0] == '3':
        if not dq:
            print(-1)
        else:
            print(dq.popleft())

    elif order[0] == '4':
        if not dq:
            print(-1)
        else:
            print(dq.pop())

    elif order[0] == '5':
        print(len(dq))

    elif order[0] == '6':
        if not dq:
            print(1)
        else:
            print(0)

    elif order[0] == '7':
        if dq:
            print(dq[0])
        else:
            print(-1)

    elif order[0] == '8':
        if dq:
            print(dq[-1])
        else:
            print(-1)
'''

# 백준 2346 - 풍선 터트리기
# 코드가 좀 매우매우 더럽고 구리지만 답은 잘 나오는데 틀렸데... 아니 왜????????????? (솔직히 스택, 덱, 큐 중에서 하나 써야하긴하는데(의무는아니지만) 아무것도 안쓰긴함....)
# i = (i + move) % N 이 부분을 계산하고 넘어가는데, 이제 넘어가는 요소들 사이에 0이 있을 수 있기때문에 틀림
# -> 왜냐면, 살아있는 풍선로만 개수를 세서 넘어가야하기 때문.
"""
N = int(input())
numbers = list(map(int, input().split()))
i = 0
result = []

# 이동은 하는데 터진 풍선은 빼고 이동해야함. -> 0으로 초기화를하자. 종이에 0은 적혀있지않다고했으니까.
for _ in range(N):
    if numbers[i] != 0:
        result.append(i+1)
        move = numbers[i]
        numbers[i] = 0
        i = (i + move) % N
    else:
        while numbers[i] == 0:
            if move < 0:
                i -= 1
                if i < 0:
                    i = N-1

            else:
                i += 1
                if i > (N-1):
                    i = 0
        result.append(i+1)
        move = numbers[i]
        numbers[i] = 0
        i = (i + move) % N

result = map(str, result)
print(' '.join(result))
"""


# 인터넷 검색 해답
# enumerate와 rotate를 알았다면 이걸 풀 수 있었을 텐뎅...
'''
import sys
from collections import deque

# input이라는 이름의 전역 변수를 새로 정의 (기존의 내장함수 input()은 사용 X, readline끝에 괄호 없애야함)
input = sys.stdin.readline

N = int(input())
dq = deque(enumerate(map(int, input().split())))
result = []

while dq:
    idx, move = dq.popleft()
    result.append(idx + 1)

    if move > 0:
        dq.rotate(-(move-1))  #move그대로 돌면 우리가 예상했던 번째의 수도 넘겨버림
    else:
        dq.rotate(-move)


print(' '.join(map(str, result)))
'''

# 백준 24511 - queuestack
# 출력도 잘 나오고 sys도 썼는데 시간초과라고~~~?~?~?~? 말도안돼ㅠㅠㅠㅠㅠㅠㅠㅠㅠ
# 근데 굳이 stack은 넣었다가 안빼도 괜찮긴해.... 왜냐면 넣은거 그대로 뺄거니까.
# 와 스택부분 pass써도 시간초과나옴 아 에바야~~~~~~~~~~~~~
'''
import sys
from collections import deque

input = sys.stdin.readline

N = int(input())
Type = list(map(int, input().split()))
numbers = list(map(int, input().split()))
M = int(input())
input_numbers = list(map(int, input().split()))

qu_stack = []
result = []

for i in range(N):
    qu = deque()
    qu.append(numbers[i])
    qu_stack.append((Type[i], qu))

for i in range(M):
    inumber = input_numbers[i]
    for j in range(N):
        if qu_stack[j][0] == 0: #큐일때
            qu_stack[j][1].append(inumber)
            inumber = qu_stack[j][1].popleft()
        
        else: #스택일때
            pass

    result.append(inumber)

print(' '.join(map(str, result)))  
'''
#join은 문자열만 다룰 수 있는 함수이기 때문에 무조건 str전환 필요.
# result가 문자열 리스트라면 바로 print(' '.join(result))사용 가능


# 굳이 qu_stack리스트의 원소를 튜플로 묶지말아보자
# 얘도 시간초과뜸................
'''
import sys
from collections import deque

input = sys.stdin.readline

N = int(input())
Type = list(map(int, input().split()))
numbers = list(map(int, input().split()))
M = int(input())
input_numbers = list(map(int, input().split()))

qu_stack = []
result = []

for i in range(N):
    qu = deque()
    qu.append(numbers[i])
    qu_stack.append(qu)

for i in range(M):
    inumber = input_numbers[i]
    for j in range(N):
        if Type[j] == 0: #큐일때
            qu_stack[j].append(inumber)
            inumber = qu_stack[j].popleft()
        
        else: #스택일때
            pass

    result.append(inumber)

print(' '.join(map(str, result)))
'''


# 큐 있는 부분만 큐만들면 되긴해
# 아니 얘도 시간초과임 for문을 하나로 줄였는데 시간초과임. queue도 필요한것만 만들었는데 시간초과임 이게 맞아?
'''
import sys
from collections import deque

input = sys.stdin.readline

N = int(input())
Type = list(map(int, input().split()))
numbers = list(map(int, input().split()))
M = int(input())
input_numbers = list(map(int, input().split()))

result = []

for i in range(M):
    inumber = input_numbers[i]
    for j in range(N):
        if Type[j] == 0: #큐일때
            qu = deque()
            qu.append(numbers[j])
            qu.append(inumber)
            inumber = qu.popleft()
            numbers[j] = qu.popleft()  #여기를 popleft가 아니라 넣은 수로 바꿔보자, 어차피 새로 넣는 수가 남는거잖아
        
        else: #스택일때
            pass

    result.append(inumber)

print(' '.join(map(str, result)))
'''


# 어차피 새로 넣는 수가 남는거라서 두 수를 교체하면 됨 굳이 deque안쓰고.
# 이것도 시간초관데... 그냥 이중for문이 문제인건가 싶긴함...
'''
import sys

input = sys.stdin.readline

N = int(input())
Type = list(map(int, input().split()))
numbers = list(map(int, input().split()))
M = int(input())
input_numbers = list(map(int, input().split()))

result = []

for i in range(M):
    inumber = input_numbers[i]
    for j in range(N):
        if Type[j] == 0: #큐일때
            inumber, numbers[j] = numbers[j], inumber
        
        else: #스택일때
            pass

    result.append(inumber)

print(' '.join(map(str, result)))
'''

# 인터넷 검색해답
# 역시 2중 for문이 문제였음...하...
# stack은 영향을 주지 않는다는 건 알았는데 그걸 아예 생략해서 queue에 해당하는 것만 모아서 계산해도 되는거였음. (왜 하나를 생각하고 둘은 생각 못하니...)
# 그니까 pass로 넘기지말고 애초에 pass를 쓸 여지도 주지 말자는 것.
'''
import sys
from collections import deque

input = sys.stdin.readline

N = int(input())
Type = list(map(int, input().split()))
numbers = list(map(int, input().split()))
M = int(input())
input_numbers = list(map(int, input().split()))

result = []

qu = deque()
for i in range(N):
    if Type[i] == 0:
        qu.append(numbers[i])

for i in range(M):
    qu.appendleft(input_numbers[i])
    print(qu.pop(), end=' ')
'''



# 백준 19532 - 수학은 비대면 강의 입니다.
# 브루트포스 형식 문제 - 모든 경우를 다 따지며 해 찾기. 완전탐색(순차탐색, BRS, DFS)
'''a, b, c, d, e, f = map(int, input().split())'''

"""
# 나는 연립방정식 더해서 변수 하나 제거 하고 나머지 변수앞 계수로 나눠서 구하는거라고 알고있는데 냅다 그냥 더해버리면 안되겠지요~? 하...
A = a+d
B = b+e
C = c+f

# 아니 나는 'x와 y가 각각 -999 이상 999 이하의 정수인 경우만 입력으로 주어짐이 보장'된데서 이렇게 짰는데 겁나많이 나오네 -부터해가지고..ㅋㅋㅋㅋ
for i in range(-999, 1000):
    for j in range(-999, 1000):
        if (A * i) + (B * j) == C:
            print(i, j)
            break


# 근데 이거 경우의 수 찾아서 바로 멈추게 하려고 함수로 감싸서 return으로 중지시킨다고해도 
# 바로 처음찾는 경우의 수가 -798, 999 던데 예시 결과값이 아니던데 예시결과값은 2 -1이던데
# 근데 이런게 브루트포스 아니야..? 경우의 수 다찾는거...
# 우뜨케 2 -1을 딱 뽑을수 있어? 예제 출력처럼..? (모르겟답..........아직 나는 멀엇나보다.......)
def find_solution():
    for i in range(-999, 1000):
        for j in range(-999, 1000):
            if (A * i) + (B * j) == C:
                print(i, j)
                return

find_solution()
"""

# 검색한 브루트포스 답안
# 이렇게 푸는게 맞는데 if를 잘못썼음... (그냥 연립방정식 푸는 방식을 잊었고, 브루트포스라서 그냥 냅다 직관적으로 해도됐었음)
'''
for i in range(-999, 1000):
    for j in range(-999, 1000):
        if (a * i) + (b * j) == c and (d * i) + (e * j) == f:
            print(i, j)
            break
'''



# 백준 2839 - 설탕 배달
# 거스름돈 문제와 비슷한데... 큰수인 5부터 나누고 이제 나머지값을 3이 나누면 된다고 생각했는데,
# 6, 3, 11같은경우... 3으로 먼저 나누는게 좋음... 3으로 나누면 답도나옴... 5로 먼저나누면 답 안나오는디...
# 와 11은 3을 끝까지 나눠선 안돼... 3을 3번이 아니라 2번 나누어야해.....

# 나는 3먼저 나누는 경우, 5먼저 나누는 경우 두개를 나누어서 나온 결과값을 비교하면 답이 나올거라 생각했는데
# 각 수를 끝까지 나눠선 안되는 경우가 존재 (ex. 11 -> 3을 2번 나누고 5를 한번해야 답이 3나옴. -1이 아님.)
'''
N = int(input())

def suger_deliver(N, a, b):
    answer = N // a
    N %= a
    answer += N // b
    N %= b

    if N != 0:
        answer = -1
    
    return answer

first = suger_deliver(N, 5, 3)
second = suger_deliver(N, 3, 5)

if first >= 0 and second >= 0:
    print(min(first, second))
else:
    print(max(first, second))
'''
# 어떻게 11를 입력했을때의 답을 구할 수 있을까?

# 검색한 해답
# 적은 봉투로 나누기 위해서는 큰 수로 나누는게 답.
# 하지만 5, 3 봉투 모두 다 쓰는 경우의 수도 존재하기 때문에, 5로 딱 나누어 떨어지지 않는다면,
# 3을 "하나씩" 빼가면서 5로 나뉘어지는지 확인해봐야한다.
# => 냅다 3으로 싹다 나누면 안된다는 소리다 (그리고 이게 11이 들어왔을 때 풀 수 있는 키워드)

'''
N = int(input())

answer = 0
while N >= 0:
    if N % 5 == 0:
        answer += N // 5
        print(answer)
        break
    N -= 3
    answer += 1
else:
    print(-1)
'''

# while문을 좀 더 자주 떠올려보고 사용할 수 있도록 하자.



# 백준 1654 - 랜선 자르기
'''
K, N = map(int, input().split())  #필요 랜선개수 N
line = []

for i in range(K):
    line.append(int(input()))

All = sum(line)
end = All // N
start = 1
result = 0  #중간 저장 변수로 최적해를 찾아도 하단의 else나 if로 영향 못받게 함

while start <= end:
    answer = 0
    mid = (start + end) // 2
    
    for i in line:
        answer += (i // mid)
    
    if answer < N:
        end = mid - 1

    else:
        result = mid
        start = mid + 1

print(result)  
'''
# 처음한 mid를 프린트 하면 예제1 을 입력했을 경우 - 200이 아닌 201출력... 아무래도 마지막의 else에 영향을 받은듯 하다...
# 그렇다고 중간에 N과 같을 경우를 넣는다면 그것이 최대 길이인지 장담을 못하지 않나 
# -> gpt의 도움을 받아 중간 변수를 넣는 방향으로 해결! 이것만 수정했듬. 전체적인 알고리즘은 나쁘지 않았던 것 같다!



# 백준 2110 - 공유기 설치
# C개의 공유기. N개의 집에 적당히 설치해서, 가장 인접한 두 공유기 사이의 거리를 최대로 하기.

# 최대한 먼 거리를 유지하려면 끝과 끝에 공유기 확정. 나머지 공유기들의 위치는 가운데에서 고루 분배.
# 끝과 끝수를 더해서 공유기 설치하게되면 나올 칸수로 나누고 그 길이와 가깝게 유사한 애를 고르는 것.
'''
N, C = map(int, input().split())  #N - 집 개수, C - 공유기의 개수
house = []

for i in range(N):
    house.append(int(input()))

house.sort()
average = (house[0] + house[-1]) // (C - 1)  #끝과 끝집 사이의 고른 간격이 나옴. -> 고르게 분포시키기 위해서 최대의 간격일지도.

start = 0
end = N-1
result = 0

# 이걸 공유기 - 2만큼 반복해야할텐디요....
while start <= end:
    mid = (start + end) // 2

    if average < house[mid] - house[start]:
        end = mid - 1
        
    elif average == house[mid] - house[start]:
        result = mid
        break

    else:
        result = mid
        start = mid + 1

print(result)
print(house[result] - house[0])  #이러면 안되지요... result 가 가운데중 1번째가 아닐테니까요....
'''
# 나는 실패하고 말았더...

# 40분 고민하면서 풀고 해답 검색했다... (사실 40분 넘었을지도)
# 해답
# 1. 좌표입력 받은 다음에 정렬 <- 이건 했음
# 2. start = 1, end = houst[-1] - house[0] (시작값: 최소 거리, 최대값: 최대 거리)
# 3. 앞 집 부터 공유기 설치
# 4. 설치할 수 있는 공유기 개수가 c개를 넘어가면 더 넓게 설치할 수 있다는 이야기 = 설치거리를 mid + 1 로 설정. 앞 집부터 다시 설치
# 5. c개를 넘어가지 않는다면 더 좁게 설치해야 한다는 이야기 이므로 mid - 1로 설정.

# 약간 모든 경우의 수를 구하는 것마냥 하는 것 같음.
'''N, C = map(int, input().split())  #N - 집 개수, C - 공유기의 개수
house = []

for i in range(N):
    house.append(int(input()))

house.sort()

start = 1
end = house[-1] - house[0]
answer = 0


while start <= end:
    mid = (start + end) // 2
    current = house[0]
    count = 1  #설치 공유기 count

    for i in range(1, len(house)):
        if house[i] >= current + mid:  #중간 길이와 앞 집 좌표 더하기
            count += 1
            current = house[i]

    if count >= C:  #공유기 개수보다 많으면, 더 넓게 설치 가능
        start = mid + 1
        answer = mid
    else:  #더 좁게 설치해야함
        end = mid - 1


print(answer)
'''



# 백준 1300 - K번째 수 -------------------------------------------------- 다시풀어보기... 역시 골드레벨 너무 어렵다.......
# 일반적으로 n*n배열 만들어서 정렬해서 이진탐색으로 구하기에는 너무 직관적으로 쉽고 타임오버할 것 같고 그렇게 쉬운 방법으로 해결하는 문제는 아닌 것 같음... (골드 1인데...)
# 가운데 제곱수는 한번씩 나오고 나머지는 두번씩 나온다. 근데 가운데 제곱수 1과의 곱으로 4부터 3개씩 나올 수 있음. (다른애들도)
# 최대 수는 N * N. 최소는 1
# 가운데 제곱수는 N이 자신의 수가 되면 3개, 아니면 하나. 나머지는 2개씩

'''
N = int(input())
k = int(input())

# 최대 최소 인덱스 값 (실 배열 내의 값도 가능함.)
strat = 1
end = N * N
numbers = []

# 근데 이거 N*N배열 만들어 푸는거랑 뭐가달라 직관적, 원시적인데 이거 타임오버 날 것 같은데
for i in range(1, N+1):
    for j in range(1, N+1):
        numbers.append((i*j))

numbers.sort()

while strat <= end:
    mid = (strat + end) // 2

    if mid == k:
        break
    elif mid > k:
        end = mid - 1
    else:
        strat = mid + 1


print(numbers[mid])  #mid와 k의 값이 똑같음
'''
# 아니 근데 생각해보니까 이러면 그냥 바로 unmbers[k]하지 굳이 이진탐색을 쓸 필요가 없잖아.
# 하... 이진탐색을 써서 생각의 전환이라는걸 해보자....
# N = 3일때 1, 9의 중간 값 5는 없어. 애초에 N이 3까지 밖에없기때문. -> 이걸 이용하는 걸지도.
# 인덱스 5는 3이긴함. 근데 수가 커질수록 중간값이 N이 아님. -> 5부터 그럼
# 40분이상을 고민해봤는데 모르겠다...


# 해답
# 이분탐색으로 어떤 수보다 작은 자연수의 곱(i*j)이 몇개인지 알아낼 것.
# A보다 작은 숫자가 몇 개인지 찾아내면 A가 몇번째 숫자인지 알 수 있다.
# ex) 10*10에서 20보다 작거나 같은 수를 생각해보면, 20을 행으로 나눈 몫의 값들이 개수가 되는 것을 알 수 있다. -> 이것을 응용!
# 해당 임의의 숫자(인덱스가 아니라 실제 값, mid)보다 작거나 같은 숫자들을 전부 찾아줌으로써 mid가 몇 번째 위치한 숫자인지 알아내기 가능 
# 이분탐색과 약간의 점화식을 생각해야했다.
'''
N = int(input())
k = int(input())

# 최대 최소 인덱스 값 (실 배열 내의 값도 가능함.)
start = 1
end = k  #k번째 수는 k보다 클 수 없다.
numbers = []



while start <= end:
    mid = (start+end) // 2
    temp = 0

    for i in range(1, N+1):
        temp += min(mid//i, N) #1같은 경우가 열의 숫자 N을 초과하는 경우도 있기 때문에 min을 사용하여 걸러내기를 해아함
        # mid이하의 i의 배수 or 최대N

    if temp >= k: #같은 수가 중복이 되는 경우가 있기 때문에 >= 이상으로 구해야함. (12233444이렇게 겹쳐있을때, 실 구해야하는건 첫번째 4인데, 4보다 작은 수의 개수는 첫번째 4든 두번째 4든 똑같으니까 두번째 4가 선택될 수도있다는 그런예시)
        answer = mid
        end = mid - 1
    else:
        start = mid +1
    
print(answer)
'''



# 백준 2720 - 세탁소 사장 동혁
# 평범한 거스름돈 문제랄까
'''
N = int(input())
unit = [25, 10, 5, 1]
money = []
result = []

for i in range(N):
    money.append(int(input()))

for i in money:
    r = []
    for j in unit:
        temp = i // j
        i %= j
        r.append(temp)
    result.append(r)

for i in result:
    for j in i:
        print(j, end=' ')
    print()
'''    
# print 할 때 print('\n')을 사용하면 추가로 한 줄 더 띄워서 빈 줄이 생김 (총 2번의 엔터)
# 줄바꿈을 원한다면 print('\n') 대신 print()만 사용하기



# 백준 18870 - 좌표압축
# Xi를 좌표 압축한 결과 X'i의 값은 Xi > Xj를 만족하는 서로 다른 좌표 Xj의 개수와 같아야 한다
'''
N = int(input())
numbers = list(map(int, input().split()))  #공백으로 요소들을 각각의 요소로 리스트 만드는 것 다시 숙지하기

s_numbers = sorted(numbers)  #s_unmbers는 정렬이 된 상태
result = {}  #후에 딕셔너리 정리 필요. 오랜만에 쓰려니까 기억이 잘 안난다...
order = 0

for i in range(N):
    if s_numbers[i] not in result.keys():
        result[s_numbers[i]] = order
        order += 1

for i in numbers:
    print(result[i], end=' ')
'''



# 백준 7785 - 회사에 있는 사람
'''
N = int(input())
members = {}
leave = []

for i in range(N):
    a, b = input().split()
    members[a] = b


for i in members.keys():
    if members[i] == 'enter':
        leave.append(i)

leave.sort()
for i in range(len(leave)):
    print(leave.pop())
    '''



# 백준 2903 - 중앙 이동 알고리즘
'''
N = int(input())
result = 2**N + 1

print(result*result)
'''



# 백준 1037 - 약수
'''
N_count = int(input())
numbers = list(map(int, input().split()))

numbers.sort()
print(numbers[0]*numbers[-1])
'''



# 백준 1193 - 분수 찾기
'''
N = int(input())

temp = 1
t = 1

while temp < N:
    t += 1
    temp += t

if t % 2 == 0:  #짝수일때
    a = temp - N
    print(f"{t - a}/{1 + a}")

else:  #홀수일때
    a = N - (temp - t) - 1
    print(f"{t - a}/{1 + a}")
'''



# 백준 2501 - 약수 구하기
'''
N, K = map(int,input().split())
count = 0

for i in range(1, N+1):
    if N % i == 0:
        count += 1
    
    if count == K:
        print(i)
        break

else:  #모든 for문에 걸리지 않는다면 실행하기를 하기 위해서 else사용
    print(0)
'''



# 백준 1978 - 소수 구하기
'''
N = int(input())
numbers = list(map(int, input().split()))
result = 0

for i in numbers:
    count = 0
    for j in range(1, i+1):
        if i % j == 0:
            count += 1
    if count == 2:
        result += 1

print(result)
'''



# 백준 2581 - 소수
'''
M = int(input())
N = int(input())
num = 0
numbers = []

def find_decimal(R):
    count = 0
    for j in range(1, R+1):
        if R % j == 0:
            count += 1
    if count == 2:
        return 1
    return 0 


for i in range(M, N+1):
    if find_decimal(i):
        num += i
        numbers.append(i)
    

if num == 0:
    print(-1)


else:
    print(num)
    print(min(numbers))
'''



# 백준 5086 - 배수와 약수
'''
while True:
    
    N, M = map(int, input().split())
    if N == 0 and M == 0:
        break

    if M % N == 0:
        print("factor")
    elif N % M == 0:
        print("multiple")
    else:
        print("neither")
'''



# 백준 9506 - 약수들의 합
'''
def find_measure(R):
    numbers = []
    for j in range(1, R):
        if R % j == 0:
            numbers.append(j)
    return numbers

while True:
    
    M = int(input())
    if M == -1:
        break

    numbers = find_measure(M)
    if sum(numbers) == M:
        print(f'{M} = {numbers[0]}', end='')
        for i in range(1, len(numbers)):
            print(f' + {numbers[i]}', end='')
        print('')
        
    else:
        print(f'{M} is NOT perfect.')
'''

    

# 백준 27433 - 팩토리얼
'''
N = int(input())

def factorial(N):
    if N == 1 or N == 0:
        return 1
    else:
        return factorial(N-1) * N


print(factorial(N))
'''



# 백준 27323 - 직사각형
'''
A = int(input())
B = int(input())

print(A*B)
'''



# 백준 15439 - 베라의 패션
'''
N = int(input())
print(N*(N-1))
'''



# 백준 10101 - 삼각형 외우기
'''
angle = []
for i in range(3):
    angle.append(int(input()))

if sum(angle) != 180:
    print("Error")

elif (angle[0] == 60 and angle[1] == 60) or (angle[1] == 60 and angle[2] == 60) or (angle[0] == 60 and  angle[2] == 60):
    print("Equilateral")

elif (angle[0] == angle[1]) or (angle[1] == angle[2]) or (angle[0] == angle[2]):
    print("Isosceles")
else:
    print("Scalene")
'''



# 백준 1085 - 직사각형에서 탈출
'''
x, y, w, h = map(int,input().split())
distance = [x, y, w-x, h-y]
print(min(distance))
'''



# 백준 3009 - 네 번째 점
'''
from collections import Counter

dot = []
for i in range(3):
    a, b = map(int, input().split())
    dot.append((a,b))

f_dot = []
s_dot = []

for i in dot:
    f_dot.append(i[0])
    s_dot.append(i[1])

f_count = Counter(f_dot)
s_count = Counter(s_dot)

for i, j in f_count.items():
    if j == 1:
        a = i
for i, j in s_count.items():
    if j == 1:
        b = i

print(a, b)
'''



# 백준 15894 - 수학은 체육과목 입니다.
'''
n = int(input())

if n == 1:
    print(4)
else:
    print(n*3 + (n-1) + 1)
'''



# 백준 1620 - 나는야 포켓몬 마스터 이다솜
'''
N, M = map(int, input().split())

pokemon = {}

for i in range(1, N+1):
    name = input()
    pokemon[str(i)] = name
    pokemon[name] = i

for j in range(M):
    question = input()
    print(pokemon[question])
'''



# 백준 11653 - 소인수분해
# 소수 리스트를 뽑아서 받은 수를 싹 소수 리스트를 돌려가면서 소인수 한 수 리스트를 받은다음에 sort로 정렬후 하나씩 출력
# -> 그리고 이렇게 하고 시간초과를 받음ㅋㅋ 소수를 모두구한다음에 나누는건 비효율적이래...

# 하단은 시간초과 받은 내 코드...
'''
# 소수 판별 알고리즘은 검색의 도움을 받음... 이미 내가 소수판별 알고리즘을 몇개 짰지만 계속 for 돌리는 거라 너무 오래걸릴 것 같았기 때문...
import math

N = int(input())
decimal = []

if N == 1:
    print('')

for j in range(2, N+1):
    for i in range(2, int(math.sqrt(j))+1):
        if j % i == 0:
            break
    else:
        decimal.append(j)

# decimal차례대로 한번씩 나눠보고 0되면 append, 그리고 다시 새 수를 decimal차례로 나눔
while True:
    for i in decimal:
        if N % i == 0:
            print(i)
            N //= i
            break
    else:
        break
'''

# 해답 (with. claude)
'''
N = int(input())

if N == 1:
    pass

else:
    i = 2
    # i*i는 sqrt()랑 같은 의미
    while i * i <= N:
        while N % i == 0:
            print(i)
            N //= i
        i += 1

    if N > 1:
        print(N)
'''



# 백준 11050 - 이항 계수1
# 이항 계수 구하는 식: n! / (k!(n-k)!)
# -> 팩토리얼을 구현하면 될 듯하다.
# 근데 이러니까 런타임 에러남... 재귀함수로 풀면 안되려나봐 -> 그래서 for문으로 바꿧듬
"""
N, K = map(int, input().split())

'''
def factorial(n):
    if n == 1:
        return 1
    else:
        return n*factorial(n-1)
'''

def factorial(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

print(factorial(N)//(factorial(K)*factorial(N-K)))
"""



# 백준 5073 - 삼각형과 세 변
'''
while True:
    length = list(map(int, input().split()))

    if length[0] == 0:
        break

    elif (sum(length)-max(length)) <= max(length):
        print("Invalid")

    elif length[0] == length[1] == length[2]:
        print("Equilateral")

    elif (length[0] == length[1]) or (length[2] == length[1]) or (length[0] == length[2]):
        print("Isosceles")

    else:
        print("Scalene")
'''



# 백준 24723 - 녹색 거탑
# 1층은 1+1, 2층은 1+2+1, 3층은 1+3+3+1, 4층은 1+4+6+4+1 로 각각의 두 요소를 두개씩 더한게 점점 늘어나고 그것들의 총합이 내려오는 경우의 수.
# input으로 주어질 N이 5이하기 때문에 재귀나 반복으로 누적해서 해도 시간 초과가 안날 것 같다.
# 그래서 하고있긴한데... 코드를 처음부터 다시짜야할 듯...
# 각 층마다 생길 떨어질 공간의 경우의 수를 요소로 list만들어서 sum을 할 생각
# 더 세세한 방법이나 각각의 정보는 굿노트에 필기한 것 참고. 나중에 다시시도할 예정

N = int(input())
num = [1, 1]


if N != 1:
    for j in range(N-1):
        temp = []
        temp.append(1)
        for i in range(len(num)-1):
            temp.append(num[i]+num[i+1])
        temp.append(1)
        num[:] = temp


print(sum(num))



















