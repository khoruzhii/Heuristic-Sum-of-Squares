import time
from typing import List, Any, Dict
from .oracle_base import OracleBase
from utils.polynomial import get_even_monomials_sqrt, analyze_basis_coverage
from utils.basis_extension import basis_extension
from data_generation.monomials.monomials import Monomial

class EvenSqrtOracle(OracleBase):
    def __init__(self, use_basis_extension: bool = False, basis_extension_params: Dict = None):
        super().__init__(use_basis_extension, basis_extension_params)

    def predict_basis(self, **kwargs) -> Dict:
        # User must provide the Polynomial object as 'poly' in kwargs
        poly = kwargs.get('poly')
        if poly is None:
            raise ValueError("EvenSqrtOracle requires the 'poly' argument (Polynomial object)")
        
        start_time = time.time()
        
        # Get monomials with even exponents and take their square roots
        predicted_basis = get_even_monomials_sqrt(poly)
        
        oracle_time = time.time() - start_time
        
        # Perform basis extension if requested
        if self.use_basis_extension and predicted_basis:
            ext_start = time.time()
            predicted_basis = basis_extension(predicted_basis, poly, **self.basis_extension_params)
            ext_time = time.time() - ext_start
        else:
            ext_time = 0
        
        # Analyze the basis coverage
        if predicted_basis:
            basis_coverage = analyze_basis_coverage(predicted_basis, poly)
        else:
            # Handle empty basis case
            basis_coverage = {
                'num_necessary': 0,
                'combinatorial_bound': 0,
                'coverage_ratio': 0.0
            }
        
        result = {
            'basis': predicted_basis,
            'time': oracle_time,
            'vertex_bound': basis_coverage['num_necessary'],
            'combinatorial_bound': basis_coverage['combinatorial_bound'],
            'coverage_ratio': basis_coverage['coverage_ratio']
        }
        
        if self.use_basis_extension:
            result['basis_extension_time'] = ext_time

        return result
