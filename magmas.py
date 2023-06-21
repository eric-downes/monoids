from utils import *

def is_action(A:NDArray[int], rows:bool = True) -> bool:
    f = ident if rows else (lambda x: x.T)
    return A.max() < f(A).shape[1]

def greens_preorders(A:NDArray[int]) -> dict[str,list[set[int]]]:
    assert is_magma(A)
    pre = dict.fromkeys(['l','r','lr'])
    pre['l'] = list(map(set, A))
    pre['r'] = list(map(set, A.T))
    pre['lr'] = [l & r for l,r in zip(pre['l'], pre['r'])]
    return pre
        
def magma_section(a:NDArray[int], subset:Iterator[int]
                  ) -> tuple[NDArray[int], list[int], bool]:
    elems = sorted(subset)
    assert max(elems) < min(a.shape)
    a = a[elems].T[elems].T
    for i,e in enumerate(elems):
        a[a==e] = i
    return a, elems, 0 <= a.min() and a.max() < len(elems)

def is_square(a: NDArray[T]) -> bool:
    return a.ndim == 2 and a.shape[0] == a.shape[1]

def is_abelian(a: NDArray[T]) -> bool:
    return is_square(a) and (a.T == a).all()
commutes = is_abelian

def is_endo(a: NDArray[int]) -> bool:
    return np.issubdtype(a.dtype, np.integer) \
        and (a < max(a.shape)).all() and (0 <= a).all()

def is_unital(a: NDArray[int]) -> bool:
    rone = np.arange(a.shape[1])
    rhas = (rone == a).all(1)
    cone = np.arange(a.shape[0])    
    chas = (cone == a.T).all(1)
    n = min(len(rhas), len(chas))
    if (rhas[:n] * chas[:n]).any():
        return True
    return False
has_unit = is_unital

def potency(a: NDArray[int]) -> bool:
    idem = True
    uni = True
    a0 = a[0,0]
    for i in range(len(a)):
        idem &= a[i,i] == i
        uni &= a[i,i] == a0
        if not (idem or uni):
            break
    return {'idempotent': idem, 'unipotent': uni}

def is_bijection(row: NDArray[int]) -> bool:
    return set(row) == set(range(len(row)))

def is_left_cancellative(a: NDArray[int]) -> bool:
    return pd.DataFrame(a).apply(is_bijection, axis = 1).prod()

def is_right_cancellative(a: NDArray[int]) -> bool:
    return is_left_cancellative(a.T)

def is_latin_square(a: NDArray[int]) -> bool:
    try:
        assert is_left_cancellative(a), 'not l cancel'
        assert is_right_cancellative(a), 'not r cancel'
        return True
    except AssertionError as e:
        print(e)
        return False
is_cancellative = is_latin_square
has_inverses = is_latin_square

def is_magma(a: NDArray[int]) -> bool:
    return is_square(a) and is_endo(a)

def is_quasigroup(a: NDArray[int]) -> bool:
    return is_magma(a) and is_latin_square(a)

def is_loop(a: NDArray[int]) -> bool:
    return is_quasigroup(a) and is_unital(a)

def left_power_assoc_hlpr(i:int, i_to_nmk:int, a:NDArray[int], k:int) -> bool:
    if not k:
        return True
    if a[(i_to_1pnmk := a[i_to_nmk, i]), i] != a[i_to_nmk, a[i, i]]:
        return False
    return pahlpr(i, i_to_1pnmk, a, k - 1)

def is_left_power_assoc_upto(a: NDArray[int], pwr: int = 3) -> bool:
    # rewrite using @ft.lru_cache to be more efficient
    assert pwr >= 0
    for i in range(len(a)):
        if not left_pwr_assoc_hlpr(i, i, a, pwr):
            return False
    return True

def invert(bij:Sequence[int]) -> pd.Series:
    ser = pd.Series(bij, name = 'vals').reset_index().set_index('vals')
    return ser.sort_index()['index']

def submagma(G:NDArray[int],
             gens:list[int],
             elements_only:bool = False) -> tuple[NDArray[int], list[int]]:
    assert is_magma(G)
    helems = list(set(gens))
    orderH = 0
    H = G[helems].T[helems].T
    helems = list(set(H.ravel()))
    while orderH - (orderH := len(helems)):
        helems = list(set((H := G[helems].T[helems].T).ravel()))
    if elements_only or orderH == len(G):
        return H, helems
    gi_to_hi = invert(helems)
    return np.array([gi_to_hi[row] for row in H]), helems



def fingerprint(x):
    return hash(tuple(x))

NAMES = {(1,0,0):'Unital Magma',
         (1,1,0):'Monoid',
         (1,0,1):'Loop',
         (1,1,1):'Group'}

Endo = list[int]

class UnitalMagma:
    # checks if the endomorphisms can be interpreted as a magma
    def __init__(self, table:List[Endo]):

        # necessary test for inverses
        hashes = {}
        self.inverses = True
        l = len(table)
        for i in range(l):
            hashes.add( fingerprint(table[i]) )
            if len(table[i]) != l:
                raise TypeError(f'argument must be square; else not a binary operator')
            if self.inverses and len(set(table[i])) != l:
                self.inverses = False

        # associativity test & unit-disqualification
        self.assoc = True
        units = set(range(l))
        for i, j in product(range(l), repeat = 2):
            if table[i][j] != i: units.discard(j)
            if self.assoc and fingerprint( table[i] * table[j] ) not in hashes:
                self.assoc = False
                
        # set or adjoin the unit
        if units:
            self.unit = min(units)
        else:
            self.unit = l
            table[l] = Endo(range(l+1))
            for i in range(l):
                table[i].append(i)
                
        # inverses sufficiency test
        if self.inverses:
            for i, row in pd.DataFrame(table).T.iterrows():
                if len(set(row)) != len(table):
                    self.inverses = False
                    break
        self.what = NAMES[(1, self.assoc, self.inverses)]

