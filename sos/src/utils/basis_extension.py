import numpy as np
import random
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from data_generation.monomials.monomials import Monomial, Polynomial
from utils.polynomial import load_polynomials_and_bases_from_jsonl, remove_k_random_elements

def basis_extension(initial_basis: List[Monomial], poly: Polynomial, max_iter: int = 10) -> List[Monomial]:
    """
    Iteratively extend a monomial basis to cover the support of a given polynomial using a greedy divisor-based heuristic.

    Args:
        initial_basis: List of Monomial objects (the initial basis B0)
        poly: Polynomial object whose support we want to cover
        max_iter: Maximum number of extension iterations

    Returns:
        List of Monomial objects forming the extended basis
    """
    basis = set(initial_basis)
    support = set(poly.terms.keys())

    for _ in range(max_iter):
        # 1. Compute all products B * B
        product_monomials = set()
        basis_list = list(basis)
        for i in range(len(basis_list)):
            for j in range(i, len(basis_list)):
                product = basis_list[i] * basis_list[j]
                product_monomials.add(product)

        # 2. Find missing terms in the support
        missing = support - product_monomials
        if not missing:
            break  # All support covered

        # 3. For each missing term, try to divide by all elements in basis
        divisor_counter = Counter()
        for m in missing:
            for b in basis:
                # Try to divide m by b (componentwise)
                if all(e1 >= e2 for e1, e2 in zip(m.exponents, b.exponents)):
                    # Compute the divisor
                    divisor_exp = tuple(e1 - e2 for e1, e2 in zip(m.exponents, b.exponents))
                    divisor = Monomial(divisor_exp)
                    divisor_counter[divisor] += 1

        if not divisor_counter:
            break  # No divisors found, cannot extend further

        # 4. Add the most common divisor to the basis
        most_common_divisor, _ = divisor_counter.most_common(1)[0]
        if most_common_divisor in basis:
            break  # No new basis element found
        basis.add(most_common_divisor)

    return list(basis)


def basis_extension_comprehensive(initial_basis: List[Monomial], poly: Polynomial, min_score: int = 7, max_iter: int = 10) -> List[Monomial]:
    """
    Comprehensive basis extension using score-based selection of divisors.
    
    This function finds all monomials that can serve as divisors for multiple terms in the 
    polynomial's support, keeping those that meet a minimum score threshold. The score 
    of a monomial t is the number of terms in the support that are divisible by t with 
    the remainder being in the initial basis.
    
    Args:
        initial_basis: List of Monomial objects (the initial basis B0)
        poly: Polynomial object whose support we want to cover
        min_score: Minimum score threshold - keep monomials that divide at least this many support terms
    
    Returns:
        List of Monomial objects forming the extended basis (includes initial_basis + new monomials)
    """
    basis = set(initial_basis)
    support = set(poly.terms.keys())

    print("Starting basis extension")
    
    # Dictionary to store scores for potential divisors
    divisor_scores = defaultdict(int)
    
    # For each term in the support, find all possible divisors
    for support_term in support:
        # Try all possible divisors by considering all combinations of exponents
        # We'll generate candidates by looking at all possible ways to split the exponents
        max_exponents = support_term.exponents
        
        # Generate all possible divisor candidates up to the exponents of this support term
        def generate_divisor_candidates(max_exp):
            candidates = []
            # Use recursive generation to create all combinations
            def generate_recursive(current_exp, remaining_dims):
                if remaining_dims == 0:
                    candidates.append(Monomial(tuple(current_exp)))
                    return
                
                dim_idx = len(current_exp)
                max_val = max_exp[dim_idx]
                
                for exp_val in range(max_val + 1):
                    generate_recursive(current_exp + [exp_val], remaining_dims - 1)
            
            generate_recursive([], len(max_exp))
            return candidates
        
        candidates = generate_divisor_candidates(max_exponents)
        
        # For each candidate divisor, check if the remainder is in the initial basis
        for candidate in candidates:
            # Check if candidate divides support_term
            if all(e1 >= e2 for e1, e2 in zip(support_term.exponents, candidate.exponents)):
                # Compute remainder
                remainder_exp = tuple(e1 - e2 for e1, e2 in zip(support_term.exponents, candidate.exponents))
                remainder = Monomial(remainder_exp)
                
                # Check if remainder is in initial basis
                if remainder in basis:
                    divisor_scores[candidate] += 1
    
    # Add all divisors that meet the minimum score threshold
    extended_basis = set(initial_basis)
    for divisor, score in divisor_scores.items():
        if score >= min_score:
            extended_basis.add(divisor)

    print("Basis extension complete", len(extended_basis))
    
    return list(extended_basis)



if __name__ == "__main__":
    # Example: polynomial p(x, y) = x^2 + y^2 + 1 + 2xy + 2x + 2y
    from sos_transformer.data_generation.monomials.monomials import Monomial, Polynomial
    from fractions import Fraction

    # Monomials for x, y
    x = Monomial((1, 0))
    y = Monomial((0, 1))
    one = Monomial((0, 0))
    x2 = Monomial((2, 0))
    y2 = Monomial((0, 2))
    xy = Monomial((1, 1))

    # Polynomial: x^2 + y^2 + 1 + 2xy + 2x + 2y
    terms = {
        x2: 1,
        y2: 1,
        one: 1,
        xy: 2,
        x: 2,
        y: 2
    }
    poly = Polynomial(terms)

    # Initial basis: [1, x]
    initial_basis = [one, x]

    # Run basis extension
    extended_basis = basis_extension(initial_basis, poly)

    # Print result
    print("Initial basis:", initial_basis)
    print("Extended basis:", sorted(extended_basis, key=lambda m: m.exponents))
    
    # Test the comprehensive basis extension
    print("\n--- Testing Comprehensive Basis Extension ---")
    comprehensive_basis = basis_extension_comprehensive(initial_basis, poly, min_score=2)
    print("Comprehensive basis (min_score=2):", sorted(comprehensive_basis, key=lambda m: m.exponents))
    
    comprehensive_basis_3 = basis_extension_comprehensive(initial_basis, poly, min_score=3)
    print("Comprehensive basis (min_score=3):", sorted(comprehensive_basis_3, key=lambda m: m.exponents)) 

    # Load polynomials and bases from jsonl file
    data_path = "/scratch/htc/npelleriti/data/sos-transformer/sdp_experiments/n6_clique/train/examples.jsonl"
    polynomials_and_bases = load_polynomials_and_bases_from_jsonl(data_path)

    equal = 0
    superset = 0
    insufficient = 0
    #print(polynomials_and_bases)

    for poly, basis in polynomials_and_bases:
        # drop one random element from basis
        #print(f"Original basis: {sorted(basis, key=lambda m: m.exponents)}")
        original_basis = basis.copy()
        basis, removed = remove_k_random_elements(basis, 10)
        #print(f"Removed elements: {removed}")
        #print(f"Reduced basis: {sorted(basis, key=lambda m: m.exponents)}")
        extended_basis = basis_extension(basis, poly)
        #print("Extended basis:", sorted(extended_basis, key=lambda m: m.exponents)) 

        # check if extended basis is equal to original basis
        if set(extended_basis) == set(original_basis):
            equal += 1
        # check if extended basis is a superset of original basis
        elif set(extended_basis) > set(original_basis):
            superset += 1
        else:
            insufficient += 1

    print(f"Greedy method - Equal: {equal}")
    print(f"Greedy method - Superset: {superset}")
    print(f"Greedy method - Insufficient: {insufficient}")
    
    # Test comprehensive method on the same data
    print("\n--- Testing Comprehensive Method ---")
    equal_comp = 0
    superset_comp = 0
    insufficient_comp = 0
    
    for poly, basis in polynomials_and_bases[:10]:  # Test on first 10 for comparison
        original_basis = basis.copy()
        basis_reduced, removed = remove_k_random_elements(basis, 10)
        comprehensive_basis = basis_extension_comprehensive(basis_reduced, poly, min_score=3)
        
        # check if comprehensive basis is equal to original basis
        if set(comprehensive_basis) == set(original_basis):
            equal_comp += 1
        # check if comprehensive basis is a superset of original basis
        elif set(comprehensive_basis) >= set(original_basis):
            superset_comp += 1
        else:
            insufficient_comp += 1
    
    print(f"Comprehensive method - Equal: {equal_comp}")
    print(f"Comprehensive method - Superset: {superset_comp}")
    print(f"Comprehensive method - Insufficient: {insufficient_comp}")
