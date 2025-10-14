import time
import numpy as np
from typing import List, Any, Dict
from math import comb, sqrt
from .oracle_base import OracleBase
from utils.polynomial import analyze_basis_coverage
from utils.basis_extension import basis_extension
from data_generation.monomials.monomials import Monomial

class RandomOracle(OracleBase):
    def __init__(self, use_basis_extension: bool = True, basis_extension_params: Dict = None):
        super().__init__(use_basis_extension, basis_extension_params)
        # Default basis extension params for greedy repair
        if self.basis_extension_params is None:
            self.basis_extension_params = {'max_iter': 10}

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
        exponents = np.random.multinomial(degree, [1/num_vars] * num_vars)
        return tuple(exponents)

    def _sample_random_monomials(self, num_vars: int, max_degree: int, num_monomials: int) -> List[Monomial]:
        """Sample random monomials uniformly from the space up to max_degree."""
        basis = []
        seen_exponents = set()
        
        # Always include constant term
        constant_exp = tuple([0] * num_vars)
        basis.append(Monomial(constant_exp))
        seen_exponents.add(constant_exp)
        
        # Sample remaining monomials
        degrees = list(range(0, max_degree + 1))
        
        # Compute degree weights for uniform distribution over all monomials
        degree_weights = [self._count_monomials_at_degree(num_vars, d) for d in degrees]
        degree_weights = np.array(degree_weights, dtype=float)
        degree_weights /= degree_weights.sum()
        
        while len(basis) < num_monomials:
            # Sample degree with probability proportional to number of monomials at that degree
            degree = np.random.choice(degrees, p=degree_weights)
            
            # Generate random exponent vector for this degree
            exponents = self._sample_random_exponents(num_vars, degree)
            
            # Add if not duplicate
            if exponents not in seen_exponents:
                basis.append(Monomial(exponents))
                seen_exponents.add(exponents)
        
        return basis

    def predict_basis(self, **kwargs) -> Dict:
        # User must provide the Polynomial object as 'poly' in kwargs
        poly = kwargs.get('poly')
        if poly is None:
            raise ValueError("RandomOracle requires the 'poly' argument (Polynomial object)")
        
        start_time = time.time()
        
        # Find maximal degree and number of terms
        monomials = list(poly.terms.keys())
        max_degree = max(sum(m.exponents) for m in monomials) if monomials else 0
        num_terms = len(monomials)
        num_vars = max(len(m.exponents) for m in monomials) if monomials else 1
        
        # Predict sqrt of number of terms as initial basis size
        initial_basis_size = max(1, int(sqrt(num_terms)))
        
        # Sample random monomials
        predicted_basis = self._sample_random_monomials(num_vars, max_degree, initial_basis_size)
        
        oracle_time = time.time() - start_time
        
        # Perform basis extension (greedy repair) if requested
        if self.use_basis_extension:
            ext_start = time.time()
            predicted_basis = basis_extension(predicted_basis, poly, **self.basis_extension_params)
            ext_time = time.time() - ext_start
        else:
            ext_time = 0
        
        # Analyze the basis coverage
        basis_coverage = analyze_basis_coverage(predicted_basis, poly)
        
        result = {
            'basis': predicted_basis,
            'time': oracle_time,
            'vertex_bound': basis_coverage['num_necessary'],
            'combinatorial_bound': basis_coverage['combinatorial_bound'],
            'coverage_ratio': basis_coverage['coverage_ratio'],
            'initial_basis_size': initial_basis_size,
            'max_degree': max_degree,
            'num_terms': num_terms
        }
        
        if self.use_basis_extension:
            result['basis_extension_time'] = ext_time

        return result
