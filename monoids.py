from functools import partial, lru_cache
from typing import TypeVar, Callable
from hashlib import blake2b
import sys
import re

from pprint import pprint

from magmas import *

T = TypeVar('T')
Unop = Callable[[T],T]
RowId = bytes|tuple[int,...]
Rows = dict[RowId, int]
HashFcn = Callable[[T], RowId]
Pair = tuple[T,T]
Sint = set[int]

class NovelRow(Exception): pass
    
@dataclass
class RowMonoid:
    row_closure: NDArray[int]
    monoid_table: NDArray[int]
    row_map: Rows
    magma_order: int # m.row_closure[:m.magma_order] is original magma
    labels: list[str] = None

def endo_orbit(endo:NDArray[int]) -> dict[tuple[int,...], int]:
    assert is_endo(endo)
    seen = {tuple(x := np.arange(len(endo))) : (c := 0)}
    while c < len(seen):
        seen.setdefault(tuple(x := endo[x]), c := c + 1)
    return seen

def adjoin_identity(a:NDArray[int]) -> NDArray[int]:
    # somewhat fragile... need to make more robust typed relabelling system 
    assert (n := a.shape[0]) == a.shape[1] and len(a.shape) == 2
    rone = np.arange(a.shape[1])
    rhas = (rone == a).all(1)
    chas = (rone == a.T).all(1)
    if (rhas * chas).any():
        return a
    a = np.where(a < 0, a - 1, a + 1) 
    a = np.vstack((rone.reshape((1, a.shape[1])), a))
    cone = np.arange(a.shape[0])
    return np.hstack((cone.reshape((a.shape[0], 1)), a))
    
def adjoin_negatives(a: NDArray[int]) -> NDArray[int]:
    # so we can handle octonions
    # we assume -1 associates: -xy= -1 * xy = x * (-y) 
    assert a.shape[0] == a.shape[1] and len(a.shape) == 2
    if (0 <= a).all(): return a
    a = adjoin_identity(a)
    nega = -a
    n = len(a)
    nega[nega == 0] = n
    nega[nega == -n] = 0
    a = np.hstack((np.vstack((a, nega)), np.vstack((nega, a))))
    a = np.where(0 <= a, a, (abs(a) + n) % (2 * n))
    return a
    
def double_rows(a : NDArray[T]) -> NDArray[T]:
    delta = np.empty(dtype = a.dtype, shape = a.shape)
    delta.fill(np.iinfo(a.dtype).max)
    return np.vstack((a, delta))

def default_count(gseq:str, gstrs:str = None) -> int:
    if gstrs: return len(re.findall(f'[{gstrs}]', gseq))
    return gseq.count('.')

def compose_and_record(a: NDArray[int],
                       i: int,
                       j: int,
                       dok: DoK[int],
                       gens: dict[int,str],
                       rows: Rows,
                       prog: dict[str, int],
                       count: Callable = default_count,
                       verbose: bool = False,
                       raise_on_novel: bool = False) -> NDArray[int]:
    n = prog['n']
    ak = a[i][ a[j] ] #even on large a, a[i] takes ns; prob dont need to cache
    k = rows.setdefault(row_hash(ak), n)
    dok[(i,j)] = k
    gseq = gens[i] + '.' + gens[j] # assoc -> parens not needed
    if (s:= gens.get(k,None)):
        gens[k] = min(gseq, s, key = count)
    else:
        gens[k] = gseq
    if k == n:
        if raise_on_novel:
            raise NovelRow()
        a[n] = ak
        n += 1
        if n == a.shape[0]:
            a = double_rows(a)
        prog['n'] = n
    if verbose:
        print(a[i], ' o ', a[j], ' = ',  ak)
    return a

def row_closure(a:NDArray[int],
                labels:list[str] = [],
                raise_on_novel:bool = False,
                verbose:bool = False
                ) -> tuple[NDArray[int], DoK[int], Rows, dict[int,str]]:
    if labels: assert len(labels) == len(a)
    rows = {}
    gens = {}
    dok = {}
    for i,r in enumerate(a):
        rows[row_hash(r)] = i
        gens[i] = labels[i] if labels else str(i)
    prog = {'n': (n := a.shape[0])}
    gstrs = ''.join(set(re.findall('\w', ''.join(labels))))
    count = partial(default_count, gstrs = gstrs)
    fcn = partial(compose_and_record, count = count,
                  dok = dok, rows = rows, prog = prog, gens = gens,
                  verbose = verbose, raise_on_novel = raise_on_novel)
    app = Applicator(fcn)
    a = app.square(double_rows(a), n)
    a = double_rows(a)
    while n != prog['n']:
        app.extend(a, (n := prog['n']))
        print(f"a now has {prog['n']} rows")
    return a[:n], dok, rows, [gens[i] for i in range(len(gens))]

def is_associative(a: NDArray[int]) -> bool:
    try: row_closure(a, verbose = False, raise_on_novel = True)
    except NovelRow: return False
    return True
associates = is_associative

def row_monoid(a: NDArray[int], labels:list[str] = None,
               verbose:bool = False) -> RowMonoid:
    # no user-friendly relabelling: identity might be row 45
    assert np.issubdtype(a.dtype, np.integer)
    assert len(a.shape) == 2
    assert (a < max(a.shape)).all() and (0 <= a).all()
    if labels:
        assert len(labels) == len(a)
    else:
        labels = {}
    if a.shape[1] <= a.max():
        a = a.T
    n_original_rows = a.shape[0]
    a, dok, rows, labels = row_closure(a, labels = labels, verbose = verbose)
    for i, label in enumerate(labels):
        labels[i] = re.sub('^\((.+)\)$', '\g<1>', label)
    order = len(rows)
    m = np.zeros(dtype=int, shape=(order, order))
    for (i,j),k in dok.items():
        m[i,j] = k
    m = adjoin_identity(m)
    return RowMonoid(a, m, rows, n_original_rows, labels)

def is_semigroup(a: NDArray[int]) -> bool:
    return is_magma(a) and is_associative(a)

def is_monoid(a: NDArray[int]) -> bool:
    return is_unital(a) and is_semigroup(a)

def is_group(a: NDArray[int]) -> bool:
    return is_loop(a) and is_associative(a)

@lru_cache
def _orbit(i:int, ai:tuple[int]) -> set[int]:
    j = i
    seen = {i:None}
    for _ in range(len(ai)):
        j = ai[j]
        if j in seen: break
        seen[j] = None
    return seen.keys()

def cyclic_endomonoid(f:Unop[T], ident:T) -> dict[T,tuple[int,T]]:
    d = {}
    i = 1
    d[ident] = (i, y := f(ident))
    while y not in d:
        d[y] = (i := i + 1, yp := f(y))
        y = yp
    return d

def group_orbit(i:int, a:NDArray[int], outyp:type = list) -> Sequence[int]:
    return outyp(_orbit(i, tuple(a[i])))

def commutators(a: NDArray[int]) -> list[set[int]]:
    assert is_group(a)
    # depends on the group identity being given index 0...
    comms = []
    for i, row in enumerate(a):
        # ij/(ji)
        c = set()
        for j, ij in enumerate(row):
            c.add(G[ij, G[G[j,i]].argmin()])
        comms.append(c)
    return comms

def quotient(g:NDArray[int], helems:set[int]
             ) -> tuple[NDArray[int], list[int]]:
    # implicitly assuming <h> is a group...
    assert not (qord := divmod(gord := len(g), len(helems)))[1]
    hmap = list(helems)
    assert is_abelian(g[hmap].T[hmap].T)
    assert is_group(g)
    gi_to_qi = {c:0 for c in helems}
    qmap = [0]
    for i in range(gord):
        if i not in gi_to_qi:
            coset = set(g[i, hmap])
            gi_to_qi |= {c:len(qmap) for c in coset}
            qmap.append(min(coset))
    lol = []
    for equiv in qmap:
        lol.append([gi_to_qi[e] for e in g[equiv, qmap]])
    return np.array(lol), qmap

def cyclic_group(n:int) -> NDArray[int]:
    lol = [list(range(n))]
    for _ in range(n - 1):
        lol.append( lol[-1][1:] + lol[-1][0:1] )
    return np.array(lol)

def homutator(G:NDArray[int],
              H:NDArray[int],
              f:dict[int,int]) -> tuple[dict[Pair[int], Pair[int]], Pair[Sint]]:
    u = {}
    img = (set(), set())
    for i in range(len(G)):
        hinv = np.argmin(H[f[i]])
        inv = np.argmin(G[i])
        for j in range(len(G)):            
            hjnv = np.argmin(H[f[j]])
            jnv = np.argmin(G[j])
            prodimg = f[G[i,j]]
            img[0].add(pre := H[prodimg, H[f[jnv], f[inv]]])            
            img[1].add(post := H[prodimg, H[hjnv, hinv]])
            u[(i,j)] = (pre, post)
    return u, img

def prod_hlpr(): pass

def direct_product(G, H) -> NDArray[int]:
    AB = sorted([G, H], key = lambda x: len(x))
    ordc = (orda := len(A := AB[0])) * (ordb := len(B := AB[1]))
    tmp = np.empty(shape = (ordc, ordc, 2), dtype = int)
    k = 0
    for a in range(orda):
        for b in range(ordb):
            tmp[a*b:(a+1)*b, k*b:(k+1)*b, 0] = A[a,k]
            tmp[a*b:(a+1)*b, k*b:(k+1)*b, 1] = B
        k += 1
    emap = {tuple(pair):i for pair, i in zip(tmp[0], range(ordc))}
    C = np.empty(shape = (ordc, ordc), dtype = int)
    for i in range(ordc):
        for j in range(ordc):
            C[i,j] = emap[tuple(tmp[i,j])]
    return C

class Aut:
    has_outer = True

class Inn(Aut):
    has_outer = False
    def __init__(self, G:NDArray[int]):
        assert is_group(G)
        order = len(G)
        inn = {}
        for p in range(order):
            q = np.argmin(px := G[p])
            pxq = px[ G.T[q] ]
            inn.setdefault(tuple(pxq), set()).add(p)
        arr = np.array(shape = (len(inn), order), dtype=int)
        labels = []
        reps = []
        for v in inn.values():
            reps += [min(v)]
            labels += [str(rep)]
        assert len(set(reps)) == len(reps)
        for r, row in zip(reps, inn.keys()):
            arr[r] = np.array(row)            
        self.row_group = row_monoid(arr, labels = labels)
        self.table = self.row_group.monoid_table

def semidirect_product(G:NDArray[int], H:NDArray[int], phi:NDArray[int]) -> NDArray[int]:
    assert is_group(G) and is_group(H)
    assert phi.shape[0] == G.shape[0] and phi.shape[1] == H.shape[0]
    assert (phi[0] == np.arange(phi.shape[1])).all()
    for row in phi:
        d = {i:j for i, j in enumerate(row)}
        _, img = homutator(H,H,d)
        assert len(img[0] | img[1]) == 1
    AB = sorted([G, H], key = lambda x: len(x))
    ordc = (orda := len(A := AB[0])) * (ordb := len(B := AB[1]))
    tmp = np.empty(shape = (ordc, ordc, 2), dtype = int)
    for a in range(orda):
        tmp[:, a*ordb:(a+1)*ordb, 1] = np.tile(B[phi[a]], (orda, 1))
    for c in range(ordc):
        a = c // ordb
        tmp[c, :, 0] = np.repeat(A[a], ordb)
    print(tmp)
    emap = {tuple(pair):i for pair, i in zip(tmp[0], range(ordc))}
    C = np.empty(shape = (ordc, ordc), dtype = int)
    for i in range(ordc):
        for j in range(ordc):
            C[i,j] = emap[tuple(tmp[i,j])]
    return C

    AB = sorted([G, H], key = lambda x: len(x))
    A = invert_group(AB[0])
    return direct_product(A, AB[1])


        
        


'''
def left_magma_pow(i:int, pwr:int, a:NDArray[int], check:bool = False) -> int:
    # only well defined for power-assoc magmas:
    assert k > 0, "undef for generic magma"
    if check:
        assert left_power_assoc_hlpr(i, i, a, pwr)
    if pwr == 1:
        return i
    return pos_pow_hlpr(i, i, a, pwr - 1)

def magma_pow(i:int, pwr:int, a:NDArray[int],
              left:bool = True, check:bool = False) -> int:
    return left_magma_pow(i, pwr, a if left else a.T, check)[0]

def group_pow(i:int, pwr:int, a:NDArray[int], check:bool = False) -> int:
    if check:
        assert is_group(a)
    # also works for a power associative magma
    if pwr < 0:
        i = a[i].argmin() # 0 is always ident here; finds the inverse
        pwr = abs(pwr)
    if not pwr:
        return 0
    return left_magma_pow(i, pwr, a, check = False)


def generate_labels(action:NDArray[int], rows:bool = True) -> list[str]:
    # we already do this in row_monoid...
    # either create the full Green's preorder on fcns or
    # just focus on the Green's preorder in magmas
    s = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNPQRSTUVWXYZ"
    if not rows: action = action.T
    assert action.max() < action.shape[1]
    k = 0
    lookup = {tuple(act):i for i, act in enumerate(action)}
    greens = {}
    semigroup = set()
    while lookup:
        orbit = endo_orbit(action[k])
        semigroup |= orbit.keys()
        for act in op.and_(*sorted([lookup.keys(), orbit.keys()], key = len)):
            greens[lookup[act]] = (k, orbit[act])
            lookup.pop(act)
        k += 1
 ...
    orbits = sorted([(i, endo_orbit(act)) for i, act in enumerate(action)],
                    key = lambda x: len(x[1]))
    closed = reduce(lambda x,y: x[1].keys()|x[2].keys(), orbits)
    labels = [''] * len(action)
    c = 0
    while orbits:
        i, orbit = orbits.pop()
        labels[i] = s[c := c + 1]
        if not (closed := closed - orbit.keys()):
            break
        orbits = sorted([orb - orbit.keys() for orb in orbits], key = lambda x: len(x[1]))
    return labels


it seems like for a one-generator abelian group <a|..>, we should
be able to form a monoid just knowing information about a*a...


def ring_extension(a: NDArray[int]) -> NDArray[int]:
    if not is_abelian(a) or not is_group(a):
        return None
    m = np.zeros(dtype = int, shape = np.r_[a.shape]+1)

shared = a.keys() & b.keys()
afst = min(a[list(shared)], key = lambda t: t[0])
bfst = min(b[list(shared)], key = lambda t: t[0])
usea = bfst <= afst
for x in (
    np.array(a.keys())

for sorted(cyclic_monoids, key = )
'''
