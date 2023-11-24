from itertools import combinations

def find_gens(G:NDArray[int],
              rank:int,
              subset:set[int] = None) -> set[int]|None:
    if subset:
	subset.discard(0)
    else:
        subset = set(range(1, len(G)))
    for gens in combinations(subset, rank):
        _, elems = submagma(G, gens + (0,))
	if len(elems) == len(G):
            return set(gens)
    return None

