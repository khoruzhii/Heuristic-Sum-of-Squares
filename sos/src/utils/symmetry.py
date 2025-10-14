"""
Deterministic canonicalisation of a polynomial up to variable permutations
using python-igraph (BLISS backend). 

> pip install python-igraph 
"""

from typing import List, Tuple
import itertools
import igraph as ig
import time 
from utils.polynomial import permute_polynomial_object



# ----------------------------------------------------------------------
# helper – custom Polynomial  ➜  graph 
# ----------------------------------------------------------------------
def _polynomial_object_to_igraph(poly_obj, num_vars: int):
    """
    Build a 3-colour igraph

        • colour 0 : variable vertices            (ids 0 … nV-1)
        • colour 1 : monomial vertices            (ids nV … nV+nM-1)
        • colour 2 : one stub vertex PER exponent (remaining ids)

    Args
    ----
        poly_obj : instance of your Polynomial class
                   (assumed to have .terms : {Monomial: coeff}
                    and each Monomial has .exponents tuple)
        num_vars : total number of variables in the ring

    Returns
    -------
        g        : igraph.Graph
        colours  : list[int] parallel to vertex ids
        var_ids  : list[int] == [0,1,…,nV-1]
    """
    terms = list(poly_obj.terms.keys())            # monomials only
    nV    = num_vars
    nM    = len(terms)
    nS    = sum(sum(m.exponents[:num_vars]) for m in terms)   # one stub per power

    g = ig.Graph()
    g.add_vertices(nV + nM + nS)                   # vertices are 0 … N-1

    colours = [0]*nV + [1]*nM + [2]*nS
    var_ids = list(range(nV))

    m_ptr, s_ptr = nV, nV + nM
    for mono in terms:
        m = m_ptr; m_ptr += 1                      # monomial vertex id
        # pad / trim exponent tuple to length num_vars
        exp_vec = mono.exponents[:num_vars] + (0,)*(max(0, num_vars-len(mono.exponents)))
        for v_idx, k in enumerate(exp_vec):
            for _ in range(k):                     # one stub per exponent
                s = s_ptr; s_ptr += 1
                g.add_edges([(v_idx, s), (s, m)])

    return g, colours, var_ids
    


# ----------------------------------------------------------------------
# helper 2 – canonical permutation from igraph
# ----------------------------------------------------------------------
def canonical_variable_permutation2(polynomial, num_vars: int) -> Tuple[int, ...]:
    """
    Compute the canonical variable permutation for a Polynomial object
    (from the monomials module) using igraph's canonical_permutation.

    This function constructs a colored graph representation of the polynomial,
    then uses igraph's canonical_permutation to find a canonical ordering
    of the variables up to permutation symmetry.

    Args:
        polynomial: Polynomial object (from data_generation.monomials.monomials)
        num_vars: Number of variables in the polynomial

    Returns:
        A tuple representing the canonical permutation of variable indices.
        For example, (2, 0, 1) means variable 0 maps to position 2, etc.
    """
    g, colours, var_ids = _polynomial_object_to_igraph(polynomial, num_vars)

    # keyword name & return signature differ across igraph versions
    try:
        result = g.canonical_permutation(color=colours)   # igraph ≥0.9
    except TypeError:
        result = g.canonical_permutation(colors=colours)  # igraph 0.8
    perm = result[0] if isinstance(result, tuple) else result  # 0.8 / 0.9.9+

    ordered = sorted(var_ids, key=lambda v: perm[v])      # variable order
    return tuple(ordered.index(i) for i in range(num_vars))


def canonicalise2(polynomial, num_vars: int):
    """
    Return the canonical variable permutation for a Polynomial object.

    Args:
        polynomial: Polynomial object (from data_generation.monomials.monomials)
        num_vars: Number of variables in the polynomial

    Returns:
        The canonical permutation tuple as returned by canonical_variable_permutation2.
    """
    π = canonical_variable_permutation2(polynomial, num_vars)
    return π








# ----------------------------------------------------------------------
# demo – exhaustive S_3 test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Test with Polynomial objects
    print("\n" + "="*60)
    print("Testing with Polynomial objects from monomials module")
    print("="*60)
    
    # Import the monomials module
    from data_generation.monomials.monomials import Polynomial, Monomial
    
    # Create a non-symmetric polynomial: x0^3 + x1^2 + x2
    m1 = Monomial((3, 0, 0))  # x0^3
    m2 = Monomial((1, 1, 0))  # x1^2  
    m3 = Monomial((0, 2, 0))  # x2
    m4 = Monomial((0, 0, 1))  # x3
    m5 = Monomial((0, 0, 4))  # x4
    
    terms = {m1: 7.0, m2: 3.0, m3: 4.0, m4: 1.0, m5: 2.0}
    poly_obj = Polynomial(terms, rational=False)




    print(f"Original polynomial: {poly_obj}")
    p = canonicalise2(poly_obj, 3)
    can_poly = permute_polynomial_object(poly_obj, p, 3)
    #print(f"Canonical permutation: {p}")
    print(f"CANONICALISED: {can_poly}")
    for perm in itertools.permutations(range(3)):
        poly_obj_perm = permute_polynomial_object(poly_obj, perm, 3)

        p = list(canonicalise2(poly_obj_perm, 3))
        print(f"Permutation: {perm}")
        print(f"Permuted polynomial: {poly_obj_perm}")
        print("Canonicalised polynomial", permute_polynomial_object(poly_obj_perm, p, 3))