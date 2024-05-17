# test
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

class NovelRow(Exception): pass
    
@dataclass
class RowMonoid:
    row_closure: NDArray[int]
    monoid_table: NDArray[int]
    row_map: Rows
    magma_order: int # m.row_closure[:m.magma_order] is original magma
    labels: list[str] = None


def adjoin_identity(a:NDArray[int]) -> NDArray[int]:
    # somewhat fragile... need to make more robust typed relabellling system 
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
    while max(i,j) >= a.shape[0]:
        a = double_rows(a)
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
        while n >= a.shape[0]:
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
    gstrs = ''.join(set(re.findall(r'\w', ''.join(labels))))
    count = partial(default_count, gstrs = gstrs)
    fcn = partial(compose_and_record, count = count,
                  dok = dok, rows = rows, prog = prog, gens = gens,
                  verbose = verbose, raise_on_novel = raise_on_novel)
    app = Applicator(fcn)
    a = app.square(double_rows(a), n)
    a = double_rows(a)
    while n != prog['n']:
        a = app.extend(a, (n := prog['n']))
        print(f"a now has {prog['n']} rows; a.shape={a.shape}")
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

is_monoid_hom = is_homomorphism

def min_generators(M:NDArray[int]) -> set[int]:
    orbit = {}
    inv = {}
    for i in range(len(M)):
        if i not in inv: # exists y in M; xy = z ==> orbit(z) <= orbit(x)
            orbit[i] = (orb := _orbit(i, M[i]))
            for j in orb:
                inv.setdefault(j, set()).add(i)
    img = set()
    gens = set()
    key = lambda x: len(x[1])
    for i, cangen in sorted(inv.items(), key = key)
        if i in img: continue
        if len(dgen := cangen - gens) == 1:
            dorb = orbit[j := dgen.pop()]
        else:
            j, dorb = max([(j, set(orbit[j]) - img) for j in dgen], key = key)
        img.update(dorb)
        gens.update(j)
    for j in gens:
        

'''
def min_generators(M:NDArray[int]) -> set[int]:
    orbits = {}
    closure = {}
    candidates = {}
    maxlen = 0
    for i in range(len(M)):
        if i not in closure:
            orbits[i] = (orb := set(_orbit(i, M[i])))
            if (m := len(orb)) > maxlen:
                maxlen = m
                js = orb
            else: js = orb - closure
            for j in js:
                closure[j] = i
    for j, orb in sorted(orbits.items(), key = lambda x: len(x[1])):
        

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
