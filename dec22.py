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
G = row_monoid(octos).monoid_table

def too_many_octs(r) -> bool:
    return len(OCTS.intersection(r)) > 1

def searcher(combo:tuple[int,...]) -> None:
    M, melem = submagma(G, combo, max_order = 8)
    if len(melem) != 8 or \
       too_many_octs(melem) or \
       is_abelian(M) and is_unital(M) or \
       not cosets(G, melem, bail_on = too_many_octs):
            return
    print(now(), 'SUCCESS Found a candidate divisor!!!', melem)
    sig = '_'.join(melem)
    DataFrame(M).to_csv(f'candidates/{sig}.csv', header=False, index=False)

def combos(n:int, u:int) -> Iterator[tuple[int,...]]:
    for m in range(1, u):
        for combo in combinations(range(n), m):
            if len(OCTS.intersection(combo)) <= 1:
                yield combo
        print(now(), f'queued all at rank={n}')

if __name__ == '__main__':
    with Pool() as pool:
        pool.map(searcher, combos(len(G), 7))
