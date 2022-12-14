from typing import TypeVar, Callable, Iterator
from dataclasses import dataclass
from functools import partial
from hashlib import blake2b
import sys

from numpy.typing import NDArray
from pandas import DataFrame
from pprint import pprint
import numpy as np

T = TypeVar('T')
DoK = dict[tuple[T, T], T]
RowId = bytes|tuple[int,...]
Rows = dict[RowId, int]

class NovelRow(Exception): pass
    
@dataclass
class RowMonoid:
    row_closure: NDArray[int]
    monoid_table: NDArray[int]
    row_map: Rows
    magma_order: int # m.row_closure[:m.magma_order] is original magma

class Applicator:
    def __init__(self,
                 fcn : Callable[[NDArray[T], int, int], NDArray[T]]):
        self.f = fcn
        self.lo = 0
    def square(self, a: NDArray[T], until: int) -> NDArray[T]:
        a0 = a
        for i,j in iprod(range(self.lo, until), range(self.lo, until)):
            a = self.fcn(a0, i, j)
        self.lo = until
        return a
    def extend(self, a: NDArray[T], until: int) -> NDArray[T]:
        for i,j in iprod(range(0, self.lo), range(self.lo, until)):
            a = self.fcn(a, i, j)
            a = self.fcn(a, j, i)
        return self.square(a, until)

def adjoin_identity(a:NDArray[int]) -> NDArray[int]:
    # somewhat fragile... need to make more robust typed relabellling system 
    assert (n := a.shape[0]) == a.shape[1] and len(a.shape) == 2
    rone = np.arange(a.shape[1])
    rhas = (rone == a).all(1)
    chas = (rone == a.T).all(1)
    if (rhas * chas).any():
        return a
    a = np.where(a < 0, a - 1, a + 1) 
    a = np.vstack((rone, a))
    cone = np.arange(a.shape[0])
    return np.hstack((cone, a))
    
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
    
def iprod(itera:Iterator[T], iterb:Iterator[T]) -> Iterator[tuple[T,T]]:
    for a in itera:
        for b in iterb:
            yield a,b

def double_rows(a : NDArray[T]) -> NDArray[T]:
    delta = np.empty(dtype = a.dtype, shape = a.shape)
    delta.fill(np.iinfo(a.dtype).max)
    return np.vstack((a, delta))

def row_hash(r : np.array) -> tuple[int,...]|bytes:
    if len(r) <= 100:
        return tuple(r)    
    # works so long as row is C-contiguous; otherwise consider tuple(r)
    return blake2b(r).digest()

def compose_and_record(a: NDArray[int],
                       i: int,
                       j: int,
                       dok: DoK[int],
                       rows: Rows,
                       prog: dict[str, int],
                       verbose: bool = False,
                       raise_on_novel: bool = False) -> NDArray[int]:
    n = prog['n']
    ak = a[i][ a[j] ] #even on large a, a[i] takes ns; prob dont need to cache
    k = rows.setdefault(row_hash(ak), n)
    dok[(i,j)] = k
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
                raise_on_novel:bool = False,
                verbose:bool = False) -> tuple[NDArray[int], DoK[int], Rows]:
    dok = {}
    rows = {row_hash(r):i for i,r in enumerate(a)}
    prog = {'n': (n := a.shape[0])}
    fcn = partial(compose_and_record,
                  dok = dok, rows = rows, prog = prog, verbose = verbose, raise_on_novel = raise_on_novel)
    app = Applicator(fcn)
    a = app.square(double_rows(a), n)
    a = double_rows(a)
    while n != prog['n']:
        app.extend(a, (n := prog['n']))
        print(f"a now has {prog['n']} rows")
    return a[:n], dok, rows

def row_monoid(a: NDArray[int], verbose:bool = False) -> RowMonoid:
    # no user-friendly relabelling: identity might be row 45
    assert np.issubdtype(a.dtype, np.integer)
    assert len(a.shape) == 2
    assert (a < max(a.shape)).all() and (0 <= a).all()
    n0 = a.shape[0]
    a, dok, rows = row_closure(a, verbose = verbose)
    order = int(np.round(np.sqrt(len(dok))))
    if row_hash(np.arange(order)) not in rows:
        # adjoin an identity if one is not already present in the closure
        e = order
        order += 1
        for j in range(order):
            dok[(e, j)] = j
            dok[(j, e)] = j
    m = np.ndarray(dtype = a.dtype, shape = (order, order))
    for (i,j),k in dok.items():
        m[i,j] = k
    return RowMonoid(a, m, rows, n0)

def is_square(a: NDType[T]) -> bool:
    return a.ndim == 2 and a.shape[0] == a.shape[1]

def is_abelian(a: NDArray[T]) -> bool:
    return is_square(a) and (a.T == a).all()
commutes = is_abelian

def is_endo(a: NDArray[int]) -> bool:
    return np.issubdtype(a.dtype, np.integer) and (a < max(a.shape)).all() and (0 <= a).all()

def is_associative(a: NDArray[int]) -> bool:
    try: row_closure(a, verbose = False, raise_on_novel = True)
    except NovelRow: return False
    return True
associates = is_associative

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
    return pd.DataFrame(a).apply(is_bijective, axis = 1).prod()

def is_right_cancellative(a: NDArray[int]) -> bool:
    return is_left_cancellative(a.T)

def is_latin_square(a: NDArray[int]) -> bool:
    return is_left_cancellative(a) and is_right_cancellative(a)
is_cancellative = is_latin_square
has_inverses = is_latin_square

def is_magma(a: NDArray[int]) -> bool:
    return is_square(a) and is_endo(a)

def is_semigroup(a: NDArray[int]) -> bool:
    return is_magma(a) and is_associative(a)

def is_quasigroup(a: NDArray[int]) -> bool:
    return is_magma(a) and is_latin_square(a)

def is_monoid(a: NDArray[int]) -> bool:
    return is_unital(a) and is_semigroup(a)

def is_loop(a: NDArray[int]) -> bool:
    return is_quasigroup(a) and is_unital(a)

def is_group(a: NDArray[int]) -> bool:
    return is_loop(a) and is_associative(a)

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

def pos_pow_hlpr(i:int, i_to_nmk:int, a:NDArray[int], k:int
                 orbit:list[int] = []) -> int:
    if not k: return i_to_nmk
    return pos_pow_hlpr(i, a[i_to_nmk, i], a, k - 1, orbit + [i_to_nmk])

def left_magma_pow(i:int, pwr:int, a:NDArray[int], check:bool = False) -> int:
    # only well defined for power-assoc magmas:
    assert k > 0, "undef for generic magma"
    if check:
        assert left_power_assoc_hlpr(i, i, a, pwr)
    if pwr == 1: return i
    return pos_pow_hlpr(i, i, a, pwr - 1)[0]

def magma_pow(i:int, pwr:int, a:NDArray[int],
              left:bool = True, check:bool = False) -> int:
    return left_magma_pow(i, pwr, a if left else a.T, check)

def group_pow(i:int, pwr:int, a:NDArray[int], check:bool = False) -> int:
    if check:
        assert is_group(a)
    # also works for a power associative magma
    if pwr < 0:
        i = a[i].argmin() # 0 is always ident here; finds the inverse
        pwr = abs(pwr)
    if not pwr: return 0
    return left_magma_pow(i, pwr, a, check = False)

def orbit()

Representation = Sequence[Callable[[int,...], bool]]

def is_rep_valid(rep:Representation, a:NDArray[int]) -> bool:
    assert is_group(a)
    for f in rep:
        


if __name__ == '__main__':

    if '--octonions' in sys.argv:
        # https://en.wikipedia.org/wiki/Octonion#Definition # with -e_0 -> 8
        octos = adjoin_negatives(
            np.array([[0,1,2,3,4,5,6,7], \
                      [1,8,3,-2,5,-4,-7,6],\
                      [2,-3,8,1,6,7,-4,-5],\
                      [3,2,-1,8,7,-6,5,-4],\
                      [4,-5,-6,-7,8,1,2,3],\
                      [5,4,-7,6,-1,8,-3,2],\
                      [6,7,4,-5,-2,3,8,-1],\
                      [7,-6,5,4,-3,-2,1,8]]) )
        fil = 'oct_monoid.csv'
        print(f'row_monoid(magma) demo using octonion magma; saving to {fil}')
        data = row_monoid(octos)
    else:
        fil = 'rps_monoid.csv'
        print(f'row_monoid(magma) demo using RPS magma; saving to {fil}')
        rps_magma = np.array([[0,1,0], [1,1,2], [0,2,2]])
        data = row_monoid(rps_magma)
    DataFrame(data.monoid_table).to_csv(fil, index=False, header=False)
    print('\n\n\nresults!')
    print(f'\n\noriginal magma:\n{data.row_closure[:data.magma_order]}')
    print(f'\n\nrow monoid:\n{data.monoid_table}')
    print('\n\nmapping: \n')
    pprint(data.row_map)
          
'''

it seems like for a one-generator abelian group <a|..>, we should
be able to form a monoid just knowing information about a*a...


def ring_extension(a: NDArray[int]) -> NDArray[int]:
    if not is_abelian(a) or not is_group(a):
        return None
    m = np.zeros(dtype = int, shape = np.r_[a.shape]+1)

'''
