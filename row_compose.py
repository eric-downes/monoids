from dataclasses import dataclass
from functools import partial
from hashlib import blake2b
from typing import TypeVar, Callable, Iterator

from numpy.typing import NDArray
from pandas import DataFrame
from pprint import pprint
import numpy as np

T = TypeVar('T')
DoK = dict[tuple[T, T], T]
RowId = bytes|tuple[int,...]
Rows = dict[RowId, int]

@dataclass
class RowMonoid:
    row_closure: NDArray[int]
    monoid_table: NDArray[int]
    row_map: Rows
    magma_order: int # m.row_closure[:m.magma_order] is original magma

class Applicator:
    def __init__(self, f : Callable[[NDArray[T], int, int], NDArray[T]]):
        self.f = f
        self.lo = 0
    def square(self, a: NDArray[T], until: int) -> NDArray[T]:
        for i,j in iprod(range(self.lo, until), range(self.lo, until)):
            a = self.f(a, i, j)
        self.lo = until
        return a
    def extend(self, a: NDArray[T], until: int) -> NDArray[T]:
        for i,j in iprod(range(0, self.lo), range(self.lo, until)):
            a = self.f(a, i, j)
            a = self.f(a, j, i)
        return self.square(a, until)

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
                       verbose: bool = False) -> NDArray[int]:
    n = prog['n']
    ak = a[i][ a[j] ] #even on large a, a[i] takes ns; prob dont need to cache
    k = rows.setdefault(row_hash(ak), n)
    dok[(i,j)] = k
    if k == n:
        a[n] = ak
        n += 1
        if n == a.shape[0]:
            a = double_rows(a)
        prog['n'] = n 
    if verbose:
        print(a[i], ' o ', a[j], ' = ',  ak)
    return a

def row_closure(a:NDArray[int],
                verbose:bool = False) -> tuple[NDArray[int], DoK[int], Rows]:
    dok = {}
    rows = {row_hash(r):i for i,r in enumerate(a)}
    prog = {'n': (n := a.shape[0])}
    fcn = partial(compose_and_record, dok = dok, rows = rows, prog = prog, verbose = verbose)
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

if __name__ == '__main__':
    fil = 'rps_monoid.csv'
    print(f'row_monoid(magma) demo using RPS magma; saving to {fil}')
    rps_magma = np.array([[0,1,0], [1,1,2], [0,2,2]])
    data = row_monoid(rps_magma, verbose = True)
    DataFrame(data.monoid_table).to_csv(fil, index=False, header=False)
    print('\n\n\nresults!')
    print(f'\n\noriginal magma:\n{data.row_closure[:data.magma_order]}')
    print(f'\n\nrow monoid:\n{data.monoid_table}')
    print('\n\nmapping: \n')
    pprint(data.row_map)
          
