


class Endo(list):
    def __init__(self, l:List[int]):
        super().__init__([i for i in l if isinstance(i, int)])
        self.is_id = self == list(range(len(self)))
        
    def __mul__(self, other:Union[int, List[int]]) -> Endo:
        return self(other)
        
    def __call__(self, other:Union[int, List[int]], strict:bool = True) -> Endo:
        if isinstance(other, int):
            return self[other % len(self)]
        if strict:
            assert len(self) == len(other)
        if self.is_id:
            return other
        if isinstance(other, Endo) and other.is_id:
            return self
        return [self[i % len(self)] for i in other]
        
    def __rmul__(self, other:List[int]) -> Endo:
        return Endo(other)(self)

class Coset(set):
    def __init__(self, *args, op:Binop = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = op
        
    def __mul__(self, x:Any):
        return {self.op(x,y) for y in self}
        

    
