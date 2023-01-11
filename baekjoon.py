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