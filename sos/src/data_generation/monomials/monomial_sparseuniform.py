import numpy as np
from typing import List
from math import comb
from data_generation.monomials.monomial_base import BasisSampler
from data_generation.monomials.monomials import Monomial, MonomialBasis

class SparseUniformBasisSampler(BasisSampler):
    """Fast uniform sampling of monomial bases up to given degree."""
    
    def __init__(self, 
                 constant_term_prob: float = 0.5,
                 min_degree: int = 0,
                 num_monomials: int = 10,
                 min_sparsity: float = 0.3,
                 max_sparsity: float = 0.7,
                 num_vars: int = 4,
                 max_degree: int = 3):
        """Initialize the sampler.
        
        Args:
            constant_term_prob: Probability of including constant term
            min_degree: Minimum degree (0 or 1)
            num_monomials: Target number of monomials
        """
        self.constant_term_prob = constant_term_prob
        self.min_degree = min_degree
        self.num_monomials = num_monomials
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.num_vars = num_vars

    def _count_monomials_at_degree(self, num_vars: int, degree: int) -> int:
        """Count monomials of exactly given degree: C(n+d-1, d)."""
        if degree == 0:
            return 1
        return comb(num_vars + degree - 1, degree)
    
    def _sample_random_exponents(self, num_vars: int, degree: int) -> tuple:
        """Generate random exponent vector of given degree using multinomial sampling."""
        if degree == 0:
            return tuple([0] * num_vars)
        
        # Use numpy's multinomial for fast sampling
        # This is equivalent to throwing 'degree' balls into 'num_vars' bins
        exponents = np.random.multinomial(degree, [1/num_vars] * num_vars)
        return tuple(exponents)

    def sample(self, num_vars: int, max_degree: int, num_monomials: int = None) -> MonomialBasis:
        """Fast sampling of monomials by directly generating random exponent vectors.
        
        Args:
            num_vars: Number of variables
            max_degree: Maximum degree of monomials
            num_monomials: Number of monomials to sample (overrides default)
            
        Returns:
            List of sampled Monomial objects
        """
        base_size = num_monomials if num_monomials is not None else self.num_monomials
        variance = 2*np.sqrt(base_size)
        min_size = max(1, int(base_size)-variance)
        max_size = int(base_size + variance)
        target_size = np.random.randint(min_size, max_size + 1)
        
        basis = []
        seen_exponents = set()

        if self.num_vars:
            num_vars = self.num_vars
        if self.max_degree:
            max_degree = self.max_degree
        
        # Add constant term if requested
        if (self.min_degree == 0 and 
            np.random.random() < self.constant_term_prob and
            target_size > 0):
            constant_exp = tuple([0] * num_vars)
            basis.append(Monomial(constant_exp))
            seen_exponents.add(constant_exp)
            target_size -= 1
        
        # Sample remaining monomials by randomly choosing degrees and exponents
        degrees = list(range(max(1, self.min_degree), max_degree + 1))
        
        # Compute degree weights for uniform distribution over all monomials
        degree_weights = [self._count_monomials_at_degree(num_vars, d) for d in degrees]
        degree_weights = np.array(degree_weights, dtype=float)
        degree_weights /= degree_weights.sum()
        
        while len(basis) < target_size + (1 if seen_exponents and tuple([0] * num_vars) in seen_exponents else 0):
            # Sample degree with probability proportional to number of monomials at that degree
            degree = np.random.choice(degrees, p=degree_weights)
            
            # Generate random exponent vector for this degree
            exponents = self._sample_random_exponents(num_vars, degree)
            
            # Add if not duplicate
            if exponents not in seen_exponents:
                basis.append(Monomial(exponents))
                seen_exponents.add(exponents)
        
        return basis

    def __repr__(self) -> str:
        return (f"SparseUniformBasisSampler(min_degree={self.min_degree}, "
                f"num_monomials={self.num_monomials}, "
                f"constant_term_prob={self.constant_term_prob})")


if __name__ == "__main__":
    import sys
    import os
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    # Create a simple sampler
    sampler = SparseUniformBasisSampler(
        constant_term_prob=1,
        num_monomials=30,
        num_vars=16,
        max_degree=8
    )

    import time

    print("Sampling 1000 bases for 16 variables, max degree 8")
    
    # Time 1000 runs of sampling
    start_time = time.time()
    for i in range(1000):
        basis = sampler.sample(16, 8)
    end_time = time.time()
    
    print(f"Time for 1000 samples: {end_time - start_time:.4f} seconds")
    print(f"Average time per sample: {(end_time - start_time) / 1000:.6f} seconds")
    
    # Show one example basis
    basis = sampler.sample(num_vars=16, max_degree=8)
    print("Generated basis:")
    for monomial in basis:
        print(f"  {monomial} (degree: {monomial.degree})")
    print(f"Total monomials: {len(basis)}")