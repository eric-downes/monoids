from monoids import *

class InvalidPres(ValueError):
    pass

@dataclass
class GrpGen:
    family: GrpGenFambly
    g: int
    name: str
    excluded: ChainMap[set]
    adopted: bool
    def __mul__(self, other:GrpGen) -> GrpGen:
        assert (G := self.family.G) == other.family.G
        g = self.family.G[self.g, other.g],
        cmap = ChainMap(set(group_orbit(g, G)),
                        self.excluded.parents, other.excluded.parents)
        return GrpGen(self.family, g, f'{self.name}.{other.name}', cmap, False)
    def __repr__(self) -> str:
        grp = self.family.G
        return f'{self} in Group({id(grp}) of order {len(grp)}'
    def __str__(self) -> str:
        return f'{self.name}({self.g})'
    def __eq__(self, other:int|GrpGen) -> bool:
        if isinstance(other, GrpGen):
            return self.family.G == other.family.G and self.g == other.g
        elif isinstance(other, int):
            return self.g == other

class GrpGenFambly:
    def __init__(self, grp:NDArray[int], rank:int,
                 namespace:list[int] = None):
        assert is_group(grp)
        assert rank >= 0
        self.rank = rank
        self.G = grp
        if namespace is None:
            self.namespace = self._create_names(rank)
        self.stack = [GrpGen(self, 0, '1', ChainMap(set()), True)]
    @classmethod
    def _create_names(cls, rank:int, symbols:str = None) -> list[str]:
        if symbols is None:
            symbols = string.lower_ascii
        names = ['1']
        for i, name in enumerate(combinations_with_replacement(
                symbols, 1 + math.log(rank) // math.log(len(symbols)))):
            names.append(name)
            if i == rank:
                return names
    def generators(self) -> Iterator[list[GrpGen]]:
        while True:
            if self._adoption_search():
                yield self.stack
            else: return
    def _adoption_search(self) -> bool:
        if len(self.stack) - 1 >= self.rank:
            return True
        for g in range(self._youngest().g + 1, len(self.G)):
            if g not in self._youngest().excluded:
                self._adopt_child(g)
            if self._adoption_search():
                return True
            else: self._disown_child()
        return False
    def _youngest(self) -> GrpGen:
        return self.stack[-1]
    def _adopt_child(self, g:int) -> None:
        assert len(self.stack) - 1 < self.rank
        name = self.namespace[len(self.stack)]
        cmap = ChainMap(set(group_orbit(g, self.G)), self._youngest().excluded)
        self.stack.append(GrpGen(self, g, name, cmap, True))
    def _disown_child(self) -> GrpGen:
        return self.stack.pop()


def is_group_pres_valid(a: NDArray[int], pres:Callable[NDArray[int],None]) -> bool:
    try:
        if not is_group(a):
            raise InvalidPres('not a group')
        pres(a)
        return True
    except InvalidPres as e:
        print(e)
        return False

def pres_Q128(a: NDArray[int]) -> None:
    # < x, y | x^64 = 1, y^2 = x^32, xyx = y >
    if (order := len(a)) != 128:
        raise InvalidPres(f'group order {order} != 128')
    n32 = 0
    for i in range(1, len(a)): # assumes ident at 0
        iorb = group_orbit(i, a)
        n32 += (order := len(iorb)) == 32
        if order > 64:
            raise InvalidPres(f'{i} has order {order} > 64')
        for j in set(range(len(a))) - set(iorb):
            if a[ a[i,j], i] != j:
                raise InvalidPres(f'xyx != y for x,y = {i},{j}')
            if a[j,j] != iorb[(32 - 1) % len(iorb)]:
                raise InvalidPres(f'y^2 != x^32 for x,y = {i},{j}')
    if n32 != 96:
        raise InvalidPres(f'should be 96 pts with order 32, there are {n32}')

def pres_xtraspec_128(G: NDArray[int]) -> None:
    # < a,b,c,d,e,f |
    #    a^2 = b^2 = (ab)^2 = c^2 = d^2 = (cd)^2 = (ef)^2
    #    e^2 = f^2 = 1
    #    ac = ca, ad = da, bc = cb, bd = db, ae = ea, be = eb
    #    ce = ec, de = ed, af = fa, bf = fb, cf = fc, df = fd >
    for ident, a, b, c, d, e, f in GrpGenFambly(G, 6).generators():
        asq = a * a
        for x in (b, c, d):
            sq_hlpr(asq, x)
        for x in (e, f):
            sq_hlpr(ident, x)
        for x, y in [(a,b), (c,d), (e,f)]:
            sq_hlpr(asq, x * y)
        for x, y in [(a,c), (a,d),
                     (b,c), (b,d),
                     (a,e), (b,e), (c,e), (d,e),
                     (a,f), (b,f), (c,f), (d,f)]:
            comm_hlpr(x, y)

def sq_hlpr(lhs:GrpGen, x:GrpGen) -> None:
    if lhs != x * x:
        InvalidPres(f'sq_hlpr: {lhs} != {xsq}')

def comm_hlpr(x:GrpGen, y:GrpGen) -> None:
    if (xy := x * y) != (yx := y * x):
        InvalidPres(f'comm_hlpr: {xy} != {yx}')


'''
class GroupGeneratorSeries:
    def __init__(self, grp:NDArray[int], rank:int, namespace: Iterator[str]):
        assert rank >= 0
        assert is_group(grp)
        self.G = grp
        self.rank = rank
        self.names = namespace
        self.stack = [GroupGenerator(series = self)] grp, 0, '1', set())]
    def full(self) -> bool:
        return len(self.stack) - 1 == self.rank
    def append(self, g:int) -> None:
        assert not self.full()
        assert g not in self.stack[-1].excluded
        n = next(namespace)
        x = ChainMap(set(group_orbit(g, self.G)), self.stack[-1].excluded)
        self.stack.append( GroupGenerator(self.G, g, n, x) )

class GroupGenerator:
    def __init__(self,
                 series: GroupGeneratorSeries = None
                 grp: NDArray[int],
                 g: int = 0,
                 name: str = '1',
                 excluded: set[int] = set(),
                 stack: GroupGenerator = None):
        self.G = grp
        self.g = g
        self.name = name
        self.excluded = excluded
        self.stack = stack or [GroupGenerator(grp)]
    def prev_excl(self) -> ChainMap|set:
        if self.prev is not None:
            return self.series.stack[
    def __mul__(self, other:NamedGen) -> NamedGen:
        assert self.G == other.G
        if self.series
        return NamedGen(self.G,
                        self.G[self.g, other.g],
                        f'{self.name}*{other.name}',
                        self.
    
    def new(self, newg:int) -> ConsGen:
        assert newg not in self.excluded
        exc = self.excluded.union(group_orbit(self.g, self.G))
        return cls(self.G, newg, self.name, exc)
'''
