import time
from typing import List, Any, Dict
from .oracle_base import OracleBase
from utils.polynomial import get_newton_polytope_basis
from utils.polynomial import analyze_basis_coverage
from data_generation.monomials.monomials import Monomial
import numpy as np

class NewtonOracle(OracleBase):
    def __init__(self, use_basis_extension: bool = False, basis_extension_params: Dict = None):
        super().__init__(use_basis_extension, basis_extension_params)

    def predict_basis(self, **kwargs) -> Dict:
        # User must provide the Polynomial object as 'poly' in kwargs
        poly = kwargs.get('poly')
        if poly is None:
            raise ValueError("NewtonOracle requires the 'poly' argument (Polynomial object)")
        start_time = time.time()
        exponents = [m.exponents for m in poly.terms.keys()]
        exponents = np.array(exponents)
        newton_basis_points = get_newton_polytope_basis(exponents)
        if newton_basis_points is not None:
            predicted_basis = [Monomial(tuple(map(int, point))) for point in newton_basis_points]
        else:
            predicted_basis = []
        oracle_time = time.time() - start_time

        
        # basis extension is not supported for the newton oracle
        if self.use_basis_extension:
            print("Warning: Basis extension is not supported for the newton oracle")

        # analyze the basis coverage
        basis_coverage = analyze_basis_coverage(predicted_basis, poly)
        
        result = {
            'basis': predicted_basis,
            'time': oracle_time,
            'vertex_bound': basis_coverage['num_necessary'],
            'combinatorial_bound': basis_coverage['combinatorial_bound']
        }

        return result 