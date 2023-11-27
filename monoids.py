from functools import partial, lru_cache
from typing import TypeVar, Callable
from hashlib import blake2b
import sys
import re

from sympy.core.symbol import Symbol
from sympy.core.mul import Mul
from pprint import pprint

from magmas import *

T = TypeVar('T')
Unop = Callable[[T],T]
RowId = bytes|tuple[int,...]
Rows = dict[RowId, int]
HashFcn = Callable[[T], RowId]
LoT = list[tuple[T,T]]

Grp = NDArray[int] # square, group
Action = NDArray[int] # rectangle, ea row injective if grp act

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
    #assert is_abelian(g[hmap].T[hmap].T)
    #assert is_group(g)
    gi_to_qi = {c:0 for c in helems}
    qmap = [0]
    for i in range(gord):
        if i not in gi_to_qi:
            coset = set(g[i, hmap])
            for c in coset:
                y = len(qmap)
                if x := gi_to_qi.get(c, None):
                    raise ValueError(f'cosets not mut excl; {c} |-> {x}, {y}')
                gi_to_qi[c] = y
            qmap.append(min(coset)) # min(coset) is the coset representative
    lol = []
    for equiv in qmap:
        lol.append([gi_to_qi[e] for e in g[equiv, qmap]])
    return np.array(lol), qmap

def cyclic_group(n:int) -> NDArray[int]:
    lol = [list(range(n))]
    for _ in range(n - 1):
        lol.append( lol[-1][1:] + lol[-1][0:1] )
    return np.array(lol)

def det(G:NDArray[int], method:str = 'lu') -> Mul:
    # lu ftw; don't even bother with other methods
    from sympy import factor, Matrix, I
    ident = [Symbol(f'x_{i}') for i in range(len(G))]
    m = [[ident[j] for j in G[i]] for i in range(len(G))]
    M = Matrix(m) # for some reason this needs to be on its own line!
    return M.det(method = method)

def general_action_product(a:NDArray[int],
                           b:NDArray[int],
                           calc_xtr:bool = True,
                           ) -> tuple[NDArray[int], LoT[int]|None, LoT[int]]:
    # setup coords
    if calc_xtr:
        xtrans = {t:i for i,t in enumerate(zip(
            np.repeat(np.arange(a.shape[0]), b.shape[0]),
            np.tile(np.arange(b.shape[0]), a.shape[0]) ))}
    else:
        xtrans = None
    ytrans = {t:i for i,t in enumerate(zip(
        np.repeat(np.arange(a.shape[1]), b.shape[1]),
        np.tile(np.arange(b.shape[1]), a.shape[1]) ))}
    # setup data
    shp = tuple(a.shape[i] * b.shape[i] for i in range(2)) + (3,)
    ab = np.ndarray(shape = shp, dtype = int)
    ab[:,:,0] = np.repeat(np.repeat(a, b.shape[0], 0), b.shape[1], 1)
    ab[:,:,1] = np.tile(b, a.shape)
    return ab, xtrans, ytrans

def direct_action_product(a:NDArray[int],
                          b:NDArray[int],
                          calc_xtr:bool = True,
                          ) -> tuple[NDArray[int], LoT[int]|None, LoT[int]]:
    ab, xtrans, ytrans = general_action_product(a, b, calc_xtr)
    # apply and return, assuming left actions
    def fcn(z:np.array) -> None:
        z[-1] = ytrans[tuple(z[:-1])]
    np.apply_along_axis(fcn, 2, ab)
    return ab[:,:,2], xtrans, ytrans
        
def direct_product(a:NDArray[int],
                   b:NDArray[int]) -> tuple[NDArray[int], LoT[int]]:
    assert is_square(a) and is_square(b)
    ab, _, ytrans = direct_action_product(a, b, False)
    # apply and return, assuming left actions
    def fcn(z:np.array) -> None:
        z[-1] = ytrans[tuple(z[:-1])]
    np.apply_along_axis(fcn, 2, ab)
    return ab[:,:,2], ytrans

def verify_act(G:Grp, act:Action, H:Grp) -> bool:
    '''
    G -- source monoid
    act -- monoid action of G on H
    H -- target monoid
    verifies the arguments satisfy
    act : G --> End(H)
    '''
    if (len(G), len(H)) != act.shape:
        return False
    for g, phi in enumerate(act):
        lhs = phi[ G[g] ]
        rhs = H[ phi[g] ][ phi ]
        if (lhs != rhs).any():
            return False
    return True

def sdp(G:Grp, H:Grp, phi:Action) -> tuple[Grp, list[int]]:
    assert verify_act(G, phi, H)
    tr = {}
    z = []
    GH = np.ndarray(shape = [(leng := len(G)) * (lenh := len(H))] * 2, dtype = int)
    for i, (g, h) in enumerate(zip(
            np.repeat(np.arange(leng), lenh),
            np.tile(np.arange(lenh), leng) )):
        z.append(t := (g,h))
        tr[t] = i
    for i, (g, h) in enumerate(z):
        aut = phi[g]
        t0 = np.repeat(G[g], lenh)
        t1 = np.tile(H[h][aut], leng)
        GH[i] = np.array([tr[t] for t in zip(t0, t1)])
    return GH, tr

def central_elems(a:NDArray[int]) -> NDArray[int]:
    assert is_square(a)
    return np.nonzero((a == a.T).all(0))

def central_product(H:NDArray[int], K:NDArray[int],
                    phi: dict[int,int] = None,
                    ) -> tuple[NDArray[int], LoT[int]]:
    ''' https://en.wikipedia.org/wiki/Central_product
    The external central product is constructed from two groups H and K,
    two subgroups H1 <= Z(H) and K1 <= Z(K), and a group isomorphism
    phi: H1 <--> K1.  The external central product is the quotient of the
    direct product H x K by the normal subgroup
      N = {(h,k) in H1 x K1 and phi(h) * k = 1_K}
    ... except we implement for loops
    '''
    assert is_loop(H) and is_loop(K)
    if phi is None:
        phi = {}
        h1_elem = central_elems(H)
        k1_elem = central_elems(K)
        assert len(h1_elem) == len(k1_elem)
        phi = {h: np.argmin(K[k]) for h,k in zip(h1_elem, k1_elem)}
    else:
        h1_elem, k1_elem = zip(sorted(phi.items(), key = lambda x: x[0]))
        assert len(set(k1_elem)) == len(phi)
    HxK, trans = direct_product(H, K)
    denom_elem = [trans[(h,k)] for h,k in phi.items()]
    HoK, qmap = quotient(HxK, denom_elem)
    # need to compose qmap with trans somehow...
    raise NotImplementedError()


'''
    from sympy.core.power import Pow
    out = []
    while factors:
        f0 = factors.pop()
        while isinstance(f := factor(f0, extension = [I]), Mul|Pow):
            factors.extend(f.args)
        out.append(f)
    return out


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
