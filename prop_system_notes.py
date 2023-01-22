PropFcn = Callable[[NDArray[int]], bool]

class Proposition:
    def __init__(self, things: PropFcn|list[PropFcn], strings:str|list[str]):
        if not isinstance(things, list):
            things = [things]
        if not isinstance(strings, list):
            strings = [strings]
        assert len(things) == len(strings)
        for string in strings:
            assert isinstance(string, str)
        for thing in things:
            assert callable(thing)
            ann = thing.__annotations__.copy()
            assert ann.pop('return') is bool
            assert len(ann) == 1 and ann.popitem()[1] == NDArray[int]
        self.assertion_zipper = zip(things, strings)
    def __call__(self, a:NDArray[int]) -> bool:
        try:
            for f, s in self.assertion_zipper:
                assert f(a), s
        except AssertionError as ss:
            print(ss)
            return False
        return True        
    def __mul__(self, other:Proposition) -> Proposition:
        thing = lambda a: self.thing(a) and other.thing(a)
        string =
        
