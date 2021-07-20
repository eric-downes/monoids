


class Endo(tuple):
    
    def __init__(self, it:Iterable[int]):
        n = len(it)
        super().__init__(i % n for i in it)
        self.is_id = self == tuple(range(len(self)))
        
    def __call__(self, other:Union[int, Iterable[int]], strict:bool = True) -> Endo:
        if isinstance(other, int):
            return self[other % len(self)]
        if strict:
            assert len(self) == len(other)
        if self.is_id:
            return other
        if isinstance(other, Endo) and other.is_id:
            return self
        return Endo(tuple(self[i % len(self)] for i in other))
        
    def __mul__(self, other:Union[int, Iterable[int]]) -> Endo:
        return self(other)
        
    def __rmul__(self, other:Iterable[int]) -> Endo:
        return Endo(other)(self)

    
def closure(endos:List[Endo]) -> List[Endo]:
    trmag = {e:str(i) for i, e in enumerate(endos)}
    for i, u in enumerate(endos):
        for j, v in enumerate(endos):
            if (x := u(v)) not in trmag:
                trmag.update({x:f'{i}.{j}'})
    return len(trmag) == len(endos), trmag
    
class Coset(set):
    def __init__(self, *args, op:Binop = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = op
        
    def __mul__(self, x:Any):
        return {self.op(x,y) for y in self}
        

    
