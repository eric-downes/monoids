

ordX = len(X)
ordY = len(Y)
assert ordX <= ordY
parts = {i:i for i in range(ordX)}
for i in range(ordX):
    for j in range(ordX):
        k = Y[i, j]
        if k in parts:
            assert X[i,j] == parts[k]
        else:
            parts[k] = X[i,j]
    
