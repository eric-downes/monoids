from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Pool
from itertools import combinations
from functools import partial
from typing import Iterator

from pandas import DataFrame
from numpy import ndarray
from arrow import now

from magmas import submagma, NDArray, is_abelian, is_unital
from monoids import cosets, row_monoid
from demo import octos

OCTS = set(range(16))
def too_many_octs(r) -> bool:
    return len(OCTS.intersection(r)) > 1

def searcher(name:str,
             shape:tuple[int,int],
             dtype:type,
             combo:tuple[int,...]) -> None:
    with SharedMemoryManager() as smm:
        shm = smm.SharedMemory(name = name)
        G = ndarray(shape = shape, dtype = dtype, buffer = shm.buf)
        M, melem = submagma(G, combo, max_order = 8)
        if len(melem) != 8 or \
           too_many_octs(melem) or \
           is_abelian(M) and is_unital(M) or \
           not cosets(G, melem, bail_on = too_many_octs):
            return
    print(now(), 'SUCCESS Found a candidate divisor!!!', melem)
    sig = '_'.join(melem)
    DataFrame(M).to_csv(f'candidates/{sig}.csv', header=False, index=False)

def combos(a:NDArray, upper_bound:int) -> Iterator[tuple[int,...]]:
    for n in range(1, upper_bound):
        for combo in combinations(range(len(a)), n):
            if len(OCTS.intersection(combo)) <= 1:
                yield combo
        print(now(), f'queued all at rank={n}')

def main():
    xs128 = row_monoid(octos).monoid_table
    with SharedMemoryManager() as smm, Pool() as pool:
        shm = smm.SharedMemory(size = xs128.nbytes)
        shared = ndarray(shape = xs128.shape, dtype = xs128.dtype, buffer = shm.buf)
        shared[:] = xs128[:]
        fcn = partial(searcher, name = shm.name, dtype = xs128.dtype, shape = xs128.shape)
        pool.map(fcn, combos(xs128, 7))

if __name__ == '__main__':
    main()
