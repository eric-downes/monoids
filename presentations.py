'''
context manager
 - accepts tests: list[Callable[[int,...],bool]]
   list and args are same len == rank
   
   
 - catches NextIter(i) exception


 
def fails_xsp128_o4_template(G, gs:list[int], i:int, j:int,
                             kk:int, ijij:int) -> tuple[bool,set[int]]:
    for k in (i,j):
        if G[gs[k], gs[k]] != kk:
            return {k}
    ij = G[gs[i], gs[j]]
    if G[ij, ij] != ijij:
        return {i,j}
    return set()

xsp128_fails = [lambda *args: 
    partial(fails_xsp128_o4_template, i = 1, j = 2
'''

DEFAULT_ALPHABET = '1abcdfghmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ'

orbits = {}
orders = {}
for g in range(len(G)):
    orb = orbit(g,G)
    orbits[g] = orb
    orders.setdefault(len(orb), set()).add(g)
young = {k:len(v) for k, v in orders.items()}

young_ok(pres, young)

Symbol = str|int

class AbstractGroup:
    def __call__(self, x:Symbol, Symbol): pass
    @lru_cache
    def orbit(self, x:Symbol) -> Symbol:
        orb = dict.fromkeys([y0 := self.identity])
        while y := self(x, y0) not in orb:
            orb[y] = None
            y0 = y
        return list(orb)
    def pow(self, x:Symbol, n:Symbol) -> Symbol:
        orb = self.orbit(x, self)
        return orb[n % len(orb]]
                   
class ConcreteGroup(AbstractGroup):
    def __init__(self, table:NDArray[int]):
        assert is_group(table)
        self.table = table
        self.identity = 0
        self.order = len(self.table)
    def __call__(self, x:int, y:int) -> int:
        return self.table[x,y]

Group = ConcreteGroup
Name = str
Assertion = Callable[[int,...], None] # raises
SafeAssertion = Assertion

GrpElem = str|int

@dataclass
class Symbol:
    context: TestingContext
    name: Name
    value: GrpElem
    def __mul__(self, other:GrpGen) -> GrpGen:
        return Symbol(self.context,
                      f'{self.name}.{other.name}',
                      G(self.value, other.value))
    def __repr__(self) -> str:
        grp = self.context.group
        return f'{self} in Group({grp.name}) of order {grp.order()}'
    def __str__(self) -> str:
        return f'{self.name}({self.value})'
    def __eq__(self, other:int|Symbol) -> bool:
        if isinstance(other, Symbol):
            return self.context == other.context and self.value == other.value


        
class Presentation:
    def __init__(self,
                 group_order:int,
                 orders:tuple[int,...]
                 ranked_assertions:list[Assertion],
                 alphabet:str = DEFAULT_ALPHABET):
        rank = len(orders)
        assert 0 < rank == len(ranked_assertions) <= len(alphabet)
        assert all([r > 1 for r in orders])
        self.namespace = alphabet[:rank]
        self.orders = orders
        self.tests : dict[str, list[SafeAssertion]] = {}
        for n, a, fs in enumerate(zip(alphabet, ranked_assertions)):
            self.tests[a] = [self.wrap(partial(assert_pt_order_n, n = n))]
            for f in fs:
                self.tests[a].append(f)
    @property
    def order(self, name:Name) -> int:
        return self.orders[name]
    def test_series(self, series:GroupGeneratorSeries, group:Group, start:int = 0) -> bool:
        context = TestingContext(self, series)
        for name, gen in zip(self.namespace[start:], series.stack[start:]):
            for test in self.tests[name]:
                try: test(gen, group)
                except (AssertionError, InvalidPres) as exc:
                    print(f'FAIL Test {test} with {symbol} raised: {exc}')
                    return False

def assert_pt_n_eq(x:Symbol, n:int, pwreq:Symbol, g:Group) -> None:
    assert g.pow(x, n) == pwreq, f'pt {x}^{n} != {pwreq} in {g}'

def assert_pt_order_n(x:Symbol, n:int, g:Group) -> None:
    assert g.order(x) == n, f'{x} not of order {n} in {g}'

def assert_pts_conj_eq(x:Symbol, y:Symbol, c:Symbol, g:Group) -> None:
    assert g(x, y) == g(y, c), f'{x}{y} != {y}{c} in {g}'
        
    data: pd.DataFrame # index: names; cols: order(int), tests(callable)
        

@dataclass
class TestingContext:
    series: GeneratorSeries
    presentation: Presentation
    
class GroupGenerator:
    def __init__(self, group: NDArray[int], value:int,
                 name:str = '', context:TestingContext = None):
        assert is_group(group)
        assert self.value < len(group)
        self.group = group
        self.value = value
        if name and context:
            assert context.presentation.data.order[name] == self.order
            assert self.value not in context.series.excluded[name]
            assert self.group == context.series.group
        self.name = name
        self.context = context
    @property
    def orbit(self) -> list[int]:
        return self.group.orbit(self.value, self.group)
    @property
    def order(self) -> int:
        return len(self.orbit)
    
class GroupGeneratorSeries:
    def __init_(self):
        self.pres: GroupPresentation
        self.group: GroupLaw
        self.stack = [GroupGenerator(self)]
    def items(self) -> Iterator[tuple[str,int]]:
        for gen in self.stack:
            yield gen.name, gen.value
    def evaluate(self, name:str, value:int) -> bool:
        d = {nam:val for nam, val in self.items()}
        assert name not in d
        d[name] = val
        return self.pres.test_series(d, self.group)
    def recurse(self,
                gen: GroupGenerator|None,
                group: Binop[[int,int],int],
                pres: GroupPresentation):
        gen = self.child(pres):
        if gen is not None:
            return True
        for g in group.young[gen.order]:
            if self.excludes(g) or pres.fail_tests(g, group):
            continue
        series.adopt(g, group)
        if self.recurse(gen.next(), group, pres):
            return True
        gen.disown()
    return False




        for g in range(self.stack[-1].g + 1, len(self.G)):
            if self._excluded(g): continue
            self._adopt_child(g)
            if self._adoption_search():
                return True
            self._disown_child()
        return False

for generator in sorted(pres.generators,
                        key = lambda x:x.order,
                        reverse = True):

    generator.adopt() # assign value consistent with prev info
    if not generator.evaluate(): # run tests with that subrank
        generator.disown() # 
    


    data.append([len(orb), orb, 
    orbits.setdefault(order := len(orb), {}).update({g: orb})
young = {order:len(elements) for order, elements 
orbits = [
young = pd.DataFrame()
young['count'] = pd.Series(map(len, orbits)).value_counts().sort_index()
young.index.name = 'order'

if pres_young and pres_young != young:
    raise InvalidPres()
if (pres_orders.value_counts().loc[young.index] < young).any():
    raise InvalidPres()
for order, pts
pt_classes = pd.DataFrame(young)





Gen = int
Gens = tuple[int,...]
Index = int
Test = Callable[[Gens], tuple[Index,...]]
def wrap_test(f:Test, x:Exception):
    res = f(gens)
    if res: raise x(res)
    else: return True
    

rank = len(G)
for gens in permutations(range(1, rank + 1), rank):
    for passing_test in tests:
        if not passing_test(G, gens): break
    else:
        if G == submagma(G, gens)[0]:
            return True
return False

def test_orbits(G, gens, orders) -> bool:
    excluded = set()
    for g in gens:
        if g in excluded: return False
        orb = orbit(g, G)
        if len(orb) != o: return False
        excluded.update(
            subgroup(G, excluded)[1])
        
        
    orbits = [orbit(g, G) for g in gens]
    
    if list(map(len, orbits)) != orders:
        return False
    
def basic_test(G, gens, orders) -> bool:

    subgens = []
    for i,g in enumerate(gens):
        subgens = append(
        subgroup(G, gens)
        
        orbits.append(orbit(i, G))


from __future__ import annotations
from itertools import combinations
import string
import math

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


class GrpGenFambly:
    def __init__(self, grp:NDArray[int], rank:int,
                 namespace:list[str] = None):
        assert is_group(grp)
        assert rank >= 0
        self.rank = rank
        self.G = grp
        if namespace is None:
            self.namespace = self._create_names(rank)
        self.stack = [GrpGen(self, 0, '1', ChainMap(set()), True)]
        self.passed = set()
    @classmethod
    def _create_names(cls, rank:int, symbols:str = None) -> list[str]:
        if symbols is None:
            symbols = string.ascii_lowercase
        names = ['1']
        for i, name in enumerate(combinations(
                symbols, 1 + math.floor(math.log(rank) // math.log(len(symbols))) )):
            names.append(name[0])
            if i == rank:
                return names
    def generators(self) -> Iterator[list[GrpGen]]:
        saved = tuple([-1] * (self.rank + 1))
        n = 0
        while True:
            if self._adoption_search():
                yield self.stack
                self.passed.add(self.sig())
                self.stack.pop()
            elif saved == self.sig():
                return
            saved = self.sig()
            self.passed.add(self.sig())
            self.stack.pop()
    def sig(self, g:int = None, k:int = None):
        if k is None: k = len(self.stack)
        s = tuple(x.g for x in self.stack[:k])
        if g is not None: s = s + (g,)
        return s
    def _excluded(self, g:int, k:int = None) -> bool:
        if k is None: k = len(self.stack) - 1
        return g in self.stack[k].excluded or \
            self.sig(g, k + 1) in self.passed
    def _adoption_search(self) -> bool:
        val = self.rank + 1 - len(self.stack)
        if not val:
            return True
        if val < 0:
            return False        
        for g in range(self.stack[-1].g + 1, len(self.G)):
            if self._excluded(g): continue
            self._adopt_child(g)
            if self._adoption_search():
                return True
            self._disown_child()
        return False
    def _adopt_child(self, g:int) -> None:
        assert len(self.stack) - 1 < self.rank
        name = self.namespace[len(self.stack)]
        cmap = ChainMap(group_orbit(g, self.G, set), self.stack[-1].excluded)
        self.stack.append(GrpGen(self, g, name, cmap, True))
    def _disown_child(self) -> GrpGen:
        return self.stack.pop()

def is_group_pres_valid(a: NDArray[int],
                        pres:Callable[NDArray[int],None],
                        verbose:bool = False) -> bool:
    try:
        if not is_group(a):
            raise InvalidPres('not a group')
        pres(a, verbose)
        print(f'{pres.__name__} IS VALID!')
        return True
    except InvalidPres as e:
        print(pres.__name__, 'INVALID:', e)
        return False

def pres_Q128(a: NDArray[int], verbose:bool = False) -> None:
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
        if verbose:
            print(f'Q128 local rels passed with i={i}')
    if n32 != 96:
        raise InvalidPres(f'should be 96 pts with order 32, there are {n32}')

def subgrp_o4_hlpr(G:NDArray[int],
                   ord4_pts:set[int],
                   helems:set[int] = None,
                   sqeq:int = None) -> set[int]:
    if helems is None: helems = set()
    else: ord4_pts -= helems
    while ord4_pts:
        a = ord4_pts.pop()
        asq = G[a,a]
        if sqeq is not None and sqeq != asq:
            continue 
        for b in ord4_pts:
            if G[b,b] == asq and G[ab := G[a,b], ab] == asq:
                break
        else: continue
        ord4_pts.drop(b)
        helems |= {0, a, b, G[a].argmin(), G[b].argmin()}
        helems.update(
            submagma(G, sorted(helems), elements_only = True)[1])
        ord4_pts -= helems
        return helems
    raise InvalidRep('exhausted pts of order 4')

with gens in GrpGenFambly(G, 6).generators():
    i,a,b,c,d,e,f = gens
    
helems = subgrp_o4_hlpr(G, ord4_pts) #<a,b>
subgrp_o4_hlpr(G, ord4_pts, helems) #<a,b,c,d>


    
    
def pres_xtraspec_128(G: NDArray[int], verbose:bool = False) -> None:
    # < a,b,c,d,e,f |
    #     a^2 = b^2 = (ab)^2 = c^2 = d^2 = (cd)^2 = (ef)^2
    #     e^2 = f^2 = 1
    #     ac = ca, ad = da, bc = cb, bd = db, ae = ea, be = eb
    #     ce = ec, de = ed, af = fa, bf = fb, cf = fc, df = fd >
    for g in GrpGenFambly(G, 6).generators():
        asq = g[1].pow(2)
        try: 
            for sqeq, i, j in [(asq,1,2), (asq,3,4), (0,5,6)]:
                if g[i].pow(2) != sqeq or g.mul([i, j, i, j]) != sqeq: 
                    IncrementFromHere()
        except IncrementFromHere as e:
            g.next_iter(i)
            
        g[2].excluded.update(
            dict.fromkeys(submagma(G, {0,1,2}, no_alias = True)[1]))        

        
        if c in b.excluded:
            c, 
        
        
        if 
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
        if set(gens) == set(submagma(G, gens, no_alias=True)):
            s = ', '.join([str(g) for g in gens])
            print(f'XtraSpec128 passed with {s}')
            return True
        else:
            print('WARNING')
            
             == submagma(G, (ident,a,b,c,d,e,f))

def sq_hlpr(lhs:GrpGen, x:GrpGen) -> None:
    if lhs != (xsq := x * x):
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
