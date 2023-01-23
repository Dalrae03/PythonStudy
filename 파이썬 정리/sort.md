## 선택 정렬 
- 처음부터 모두 훑으며 앞으로 보내기
```py
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
'''
for i in range(len(array)):
    min_n = array[i]
    for j in range(i+1, len(array)):
        if min_n > array[j]:
            min_n = array[j]
    array[i], min_n = min_n, array[i]  # 스와프는 두 변수의 '위치'를 변경하지 이렇게하면 숫자만 변경되고 원래 리스트는 그냥 그대로 다시시작 
'''

for i in range(len(array)):
    min_n = i
    for j in range(i+1, len(array)):
        if array[min_n] > array[j]:
            min_n = j
    array[i], array[min_n] = array[min_n], array[i]  # 스와프는 인덱스로 해야함

print(array)
```

<br>
<br>

## 삽입 정렬 
- 선택정렬에 비해 효율적. 정렬이 거의 되어있는 상황에서는 퀵정렬 알고리즘보다 더 효율적일 수 있다.
```py
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(1, len(array)):
    for j in range(i, 0, -1):  # range(start, end, step) start인덱스부터 end인덱스까지 step만큼 감소, 증가
        if array[j] < array[j-1]: 
            array[j], array[j-1] = array[j-1], array[j]
        else:
            break

print(array)
```

<br>
<br>

## 퀵 정렬 
- 가장 많이 사용되는 알고리즘. 데이터가 무작위로 입력되는 경우 효율적. 데이터의 특성을 파악하기 어렵다면 퀵정렬이 유리.

피벗 - 큰 숫자와 작은 숫자를 교환할 때, 교환하기 위한 기준

호어 분할 - 리스트에서 첫번째 데이터를 피펏으로 정함.
```py
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

def quick_sort(array, start, end):
    if start >= end:
        return
    pivot = start
    left = start + 1
    right = end
    while left <= right:
        while left <= end and array[left] <= array[pivot]:
            left += 1 
        while right > start and array[right] >= array[pivot]:
            right -= 1
        if left > right:
            array[right], array[pivot] = array[pivot], array[right]
        else:
            array[left], array[right] = array[right], array[left]
quick_sort(array, start, right - 1)
quick_sort(array, right + 1, end)

quick_sort(array, 0, len(array) - 1)
print(array)

# 파이썬 장점 살린 퀵 정렬 - 피벗과 데이터를 비교하는 연산횟수 증가로 시간면에서 조금 비효율적
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

def quick_sort(array):
    if len(array) <= 1:
        return array
    
    pivot = array[0]
    tail = array[1:]

    left_side = [x for x in tail if x <= pivot]
    right_side = [x for x in tail if x >= pivot]

    return quick_sort(left_side) + [pivot] + quick_sort(right_side)

    print(quick_sort(array))
```

<br>
<br>

## 계수 정렬 
- 특정한 조건이 부합할 때만 사용할 수 있지만 매우 빠른 정렬 알고리즘. 동일한 값의 데이터가 여러 개 등장할 때 적합.

조건 - 데이터의 크기 범위가 제한되어 정수 형태로 표현할 수 있을 때. 가장 큰 데이터와 가장 작은 데이터의 차이가 1000000을 넘지 않을 때.
```py
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8, 0, 5, 2]

count = [0] * (max(array)+1)

for i in range(len(array)):
    count[array[i]] += 1

for i in range(len(count)):
    for j in range(count[i]):
        print(i, end=' ')


# 정렬라이브러리에서 key를 활용한 소스코드
array = [('바나나', 2), ('사과', 5), ('당근', 3)]

def setting(data):
    return data[1]

result = sorted(array, key=setting)
print(result)
```