from monoids import *

def sort_rows_by_mean(ind:NDArray[int]):
    one = np.arange(ind.shape[1])
    means = [(i, sum(ind * one) / len(one)) for i, row in enumerate(ind)]
    perm = [x[0] for x is in sorted(means, key = lambda x: x[1])]
    return ind[perm].T, perm

perm = []
ind = odata.monoid_table < 16
one = [i for i in range(len(ind))]:
while perm != one:
    ind, perm = sort_rwos_by_mean(ind)
from monoids import *

def sort_rows_by_mean(ind:NDArray[int]):
    one = np.arange(ind.shape[1])
    means = [(i, sum(ind * one) / len(one)) for i, row in enumerate(ind)]
    perm = [x[0] for x is in sorted(means, key = lambda x: x[1])]
    return ind[perm].T, perm

perm = []
ind = odata.monoid_table < 16
one = [i for i in range(len(ind))]:
while perm != one:
    ind, perm =	sort_rwos_by_mean(ind)


from monoids import *

def sort_rows_by_mean(ind:NDArray[int]):
    one = np.arange(ind.shape[1])
    return ind[perm].T, perm

perm = []
ind = odata.monoid_table < 16
one = [i for i in range(len(ind))]
while perm != one:
    means = [(i, sum(row * one) / len(one)) for i, row in enumerate(ind)]
    perm = [x[0] for x is in sorted(means, key = lambda x: x[1])]
    ind = ind[perm].T




