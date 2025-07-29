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




