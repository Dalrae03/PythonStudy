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