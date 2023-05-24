from magmas, monoids import *

def commutator(x:int, y:int, M:NDArray[int], S:NDArray[int]) -> int:
    xy = M[x,y]
    yx = M[y,x]
    negyx = np.argmin(S[yx])
    return S[xy, negyx]

def is_lie_algebra(B:NDArray[int], K:NDArray[F], V:NDArray[F],
                   field_sum:Binop[F] = op.add,
                   field_mul:Binop[F] = op.mul) -> bool:
    # this should always work -- see http://jan-kotrbaty.cz/data/bt.pdf (p.68 Thm 4.7 -David)
    # V -- basis vectors in the Chevalley basis including 0-vector (mat of field elem)
    # B -- proposed lie bracket up to proportionality (magma of ints)
    # K -- proportionality constants (matrix of field elements)
    for i in range(len(B)):
        ri = B[i]
        for j in range(len(B)):
            rj = B[j]
            lhs = B[B[i,j]]
            Vij = V[ri[rj]]
            nVji = field_mul(-1, V[rj[ri]])
            rhs = field_mul(K[i,j], field_sum(Vij, nVji))
            if (lhs != rhs).any(): return False
    return False

def is_discrete_lie_algebra(B:NDArray[int], S:NDArray[int]) -> bool:
    # this only works for discrete Lie Algebras over finite sets
    # bc otherwise addition not closed
    if np.size(B) != np.size(S): return False
    if not is_magma(B): return False
    if not (is_abelian(S) and is_group(S)): return False
    for i in range(len(B)):
        ri = B[i]
        for j in range(len(B)):
            rj = B[j]
            negrjri = np.argmin(S[rj[ri]], 1)
            rhs = S[*zip(ri[rj], negrjri)]
            lhs = B[B[i,j]]
            if (lhs != rhs).any(): return False
    return True


