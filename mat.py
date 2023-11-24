from monoids import *

MatMon = list[NDArray[float]]

def is_matmon(m:MatMon) -> bool:
    try:
        matmon_cayley(m, True))
        return True
    except ValueError:
        return False

def freeze(x:NDArray[int]) -> tuple[tuple[int,...], ...]:
    return tuple(tuple(row) for row in x)

def unital_square_table(m:NDArray[int]
                        ) -> tuple[NDArray[int], list[int|None]]:
    assert is_square(m)
    idx = list(range(len(m)))
    if (i := get_identity_idx(m)) is not None:
        if not i: return m, idx
        jdx = idx[i:i+1] + idx[:i] + idx[i+1:]
        return m[jdx].T[jdx].T, jdx
    else:
    mm = np.zeros(shape=(lmm := len(m) + 1, lmm), dtype=int)
    one = np.arange(lmm, dtype=int)
    mm[0] = one
    mm.T[0] = one
    mm[1:].T[1:] = m.T
    return mm, [None] + idx

def matmon_cayley(m:MatMon,
                  raise_on_unknown:bool = True
                  ) -> tuple[NDArray[int], list[int|None]]:
    k = len(m)
    sm = {freeze(x): i for i, x in enumerate(m)}
    t = np.ndarray(size=(k,k), dtype=int)
    for i,x in enumerate(m):
        for j,y in enumerate(m):
            if not (p := sm.get(freeze(z := x @ y), None)):
                if raise_on_unknown:
                    raise ValueError(f'x@y = {z} not in list')
                p = k
                k += 1
            t[i,j] = p
    mc, perm = unital_square_table(t)
    dim = max([len(x) for x in m])
    return mc, [np.eye(dim) if i is None else m[i] for i in perm]

def matmon_tensor_product(M:MatMon, N:MatMon) -> MatMon:
    # need to invert mtr, ntr?
    mc, mtr = matmon_cayley(M)
    nc, ntr = matmon_cayley(N)
    mnc, mntr = direct_product(mc, nc)
    rep = [kronecker_product(mtr[m], ntr[n]) for m, n in mntr]
    return mnc, rep
    
