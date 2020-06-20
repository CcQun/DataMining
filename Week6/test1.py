all = [[1, 2, 5], [2, 4], [2, 3], [1, 2, 4], [1, 3], [2, 3], [1, 3], [1, 2, 3, 5], [1, 2, 3]]

# sup = [0, 0, 0, 0, 0]
sup = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for i in all:
    for j in range(1, len(sup) + 1):
        if j in i:
            sup[j] += 1

print(sup)

fre = {}
for i in range(1, len(sup) + 1):
    if sup[i] > 2:
        fre[i] = sup[i]

print(fre)



