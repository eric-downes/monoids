
def fingerprint(x):
    return hash(tuple(x))

NAMES = {(1,0,0):'Unital Magma',
         (1,1,0):'Monoid',
         (1,0,1):'Loop',
         (1,1,1):'Group'}

class UnitalMagma:
    # checks if the endomorphisms can be interpreted as a magma
    def __init__(self, table:List[Endo]):

        # necessary test for inverses
        hashes = {}
        self.inverses = True
        l = len(table)
        for i in range(l):
            hashes.add( fingerprint(table[i]) )
            if len(table[i]) != l:
                raise TypeError(f'argument must be square; else not a binary operator')
            if self.inverses and len(set(table[i])) != l:
                self.inverses = False

        # associativity test & unit-disqualification
        self.assoc = True
        units = set(range(l))
        for i, j in product(range(l), repeat = 2):
            if table[i][j] != i: units.discard(j)
            if self.assoc and fingerprint( table[i] * table[j] ) not in hashes:
                self.assoc = False
                
        # set or adjoin the unit
        if units:
            self.unit = min(units)
        else:
            self.unit = l
            table[l] = Endo(range(l+1))
            for i in range(l):
                table[i].append(i)
                
        # inverses sufficiency test
        if self.inverses:
            for i, row in pd.DataFrame(table).T.iterrows():
                if len(set(row)) != len(table):
                    self.inverses = False
                    break
        self.what = NAMES[(1, self.assoc, self.inverses)]

