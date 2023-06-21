
def partial_homo(G:NDArray[int], L:NDArray[int]) -> dict[int,int]:
    # always possible for any semigroup G and magma L
    l = len(L)
    sigma = {i:i for i in range(l)}
    for i in range(l):
        for j in range(l):
            if (g := G[i,j]) in sigma:
                assert sigma[g] == L[i,j]
            else:
                sigma[g] = L[i,j]
    return sigma

# attempt sigma respects inverses
def inverse_preserving(G, L, sigma) -> None:
    # assuming sigma o rlift = id_L
    l = len(L)
    lg = len(G)
    for i in range(lg):
        inv = np.argmin(G[i])
        if {i, inv} & sigma.keys():
            if inv in sigma:
                assert L[sigma[i], sigma[inv]] == 0
            else:
                sigma[inv] = np.argmin(L[sigma[i]])
    return
            
        
