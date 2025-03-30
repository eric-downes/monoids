from functools import lru_cache
import numpy as np
from typing import Callable, List, Tuple, Dict
from numpy.typing import NDArray

def is_central_element(G: NDArray[int], elem: int) -> bool:
    """Check if an element is central in the group."""
    return all(G[elem, i] == G[i, elem] for i in range(len(G)))

def extract_2cocycle(G: NDArray[int], 
                    generators: List[int], 
                    central_elem: int = None,
                    compute_all: bool = False,
                    verify: bool = True) -> Callable:
    """
    Extract the 2-cocycle from a central extension by Z₂.
    
    Args:
        G: Group multiplication table
        generators: List of generators mapping to a basis of the quotient group
        central_elem: The central element of order 2 (if None, will try to identify it)
        compute_all: Whether to precompute all cocycle values
        verify: Whether to verify the central extension properties
    
    Returns:
        Callable computing the 2-cocycle value for any pair of quotient elements
    """
    # Assuming 0 is the identity
    identity = 0
    
    # Identify the central element if not provided
    if central_elem is None:
        # Look for a central element of order 2 different from identity
        for i in range(1, len(G)):
            if is_central_element(G, i) and G[i, i] == identity:
                central_elem = i
                break
        
        if central_elem is None:
            raise ValueError("Could not identify central element of order 2")
    
    # Verify the central extension properties if requested
    if verify:
        # Check that central_elem is central
        if not is_central_element(G, central_elem):
            raise ValueError(f"Element {central_elem} is not central")
        
        # Check that central_elem has order 2
        if G[central_elem, central_elem] != identity:
            raise ValueError(f"Element {central_elem} does not have order 2")
        
        # Check that the group size is consistent
        dim = len(generators)
        if len(G) != 2**(dim + 1):
            raise ValueError(f"Group size {len(G)} inconsistent with {dim} generators")
    
    # Create function to compute a group element from a quotient vector
    @lru_cache(maxsize=None)
    def vector_to_element(vec: Tuple[int, ...]) -> int:
        elem = identity
        for i, bit in enumerate(vec):
            if bit == 1:
                elem = G[elem, generators[i]]
        return elem
    
    # Precompute cocycle values if requested
    cocycle_values = {}
    if compute_all:
        dim = len(generators)
        for bits1 in range(2**dim):
            v1 = tuple(int(bit) for bit in format(bits1, f'0{dim}b'))
            for bits2 in range(2**dim):
                v2 = tuple(int(bit) for bit in format(bits2, f'0{dim}b'))
                
                g1 = vector_to_element(v1)
                g2 = vector_to_element(v2)
                g_prod = G[g1, g2]
                
                v_sum = tuple((a + b) % 2 for a, b in zip(v1, v2))
                g_sum = vector_to_element(v_sum)
                
                if g_prod == g_sum:
                    cocycle_values[(v1, v2)] = 0
                elif g_prod == G[g_sum, central_elem]:
                    cocycle_values[(v1, v2)] = 1
                else:
                    raise ValueError(f"Inconsistent group structure detected")
    
    @lru_cache(maxsize=None)
    def cocycle(v1: Tuple[int, ...], v2: Tuple[int, ...]) -> int:
        """
        Compute the 2-cocycle value for a pair of vectors in the quotient group.
        
        Args:
            v1, v2: Vectors in (Z₂)^n representing quotient elements
            
        Returns:
            0 or 1 (the cocycle value in Z₂)
        """
        if compute_all and (v1, v2) in cocycle_values:
            return cocycle_values[(v1, v2)]
        
        g1 = vector_to_element(v1)
        g2 = vector_to_element(v2)
        g_prod = G[g1, g2]
        
        v_sum = tuple((a + b) % 2 for a, b in zip(v1, v2))
        g_sum = vector_to_element(v_sum)
        
        if g_prod == g_sum:
            return 0
        elif g_prod == G[g_sum, central_elem]:
            return 1
        else:
            raise ValueError(f"Inconsistent group structure detected")
    
    return cocycle

def verify_cocycle_properties(cocycle: Callable, dim: int) -> bool:
    """
    Verify that a 2-cocycle satisfies the required properties.
    
    Args:
        cocycle: The 2-cocycle function
        dim: Dimension of the quotient group (Z₂)^dim
        
    Returns:
        True if the cocycle satisfies all required properties
    """
    # Create all vectors in (Z₂)^dim
    vectors = []
    for bits in range(2**dim):
        vec = tuple(int(bit) for bit in format(bits, f'0{dim}b'))
        vectors.append(vec)
    
    # Check normalization: α(0,v) = α(v,0) = 0
    zero_vec = tuple(0 for _ in range(dim))
    for v in vectors:
        if cocycle(zero_vec, v) != 0 or cocycle(v, zero_vec) != 0:
            return False
    
    # Check cocycle condition: α(u,v) + α(u+v,w) = α(v,w) + α(u,v+w)
    for u in vectors:
        for v in vectors:
            for w in vectors:
                u_plus_v = tuple((a + b) % 2 for a, b in zip(u, v))
                v_plus_w = tuple((a + b) % 2 for a, b in zip(v, w))
                
                lhs = (cocycle(u, v) + cocycle(u_plus_v, w)) % 2
                rhs = (cocycle(v, w) + cocycle(u, v_plus_w)) % 2
                
                if lhs != rhs:
                    return False
    
    return True

def pullback_cocycle(cocycle: Callable, 
                    eta: Callable[[int], Tuple[int, ...]]) -> Callable:
    """
    Compute the pullback of a 2-cocycle along a homomorphism.
    
    Args:
        cocycle: The 2-cocycle on the quotient group
        eta: Homomorphism from another group to the quotient group
        
    Returns:
        The pullback cocycle
    """
    @lru_cache(maxsize=None)
    def pulled_cocycle(x: int, y: int) -> int:
        return cocycle(eta(x), eta(y))
    return pulled_cocycle
