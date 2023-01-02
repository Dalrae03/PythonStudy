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
chess = [1,1,2,2,2,8]
List = []

List = list(map(int, input().split(' ')))

# print(List)

# print(type(List[0]))

for i in range(len(chess)):
    List[i] = chess[i]- List[i]

print(List[0], List[1], List[2], List[3], List[4], List[5])