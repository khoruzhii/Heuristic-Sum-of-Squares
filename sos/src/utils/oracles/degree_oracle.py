"""
DegreeOracle: An oracle that returns all monomials up to degree d for n variables.

This oracle provides a complete enumeration of all possible monomials up to a specified
degree, making it useful as a baseline or upper bound for SOS decomposition problems.

Example usage:
    oracle = DegreeOracle(max_degree=2, num_vars=3)
    result = oracle.predict_basis()
    # Returns all monomials like: 1, x1, x2, x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2
"""

import time
from typing import List, Any, Dict
from .oracle_base import OracleBase
from data_generation.monomials.monomials import Monomial

class DegreeOracle(OracleBase):
    """
    Oracle that returns all monomials up to degree d for n variables.
    """
    
    def __init__(self, max_degree: int, num_variables: int, use_basis_extension: bool = False, basis_extension_params: Dict = None):
        """
        Initialize the degree oracle.
        
        Args:
            max_degree: Maximum degree of monomials to include
            num_vars: Number of variables
            use_basis_extension: Whether to use basis extension
            basis_extension_params: Parameters for basis extension
        """
        super().__init__(use_basis_extension, basis_extension_params)
        self.max_degree = max_degree
        self.num_vars = num_variables

    def _generate_all_monomials_up_to_degree(self, num_vars: int, max_degree: int) -> List[Monomial]:
        """
        Generate all monomials up to degree max_degree for num_vars variables.
        
        This function enumerates all possible exponent tuples (e1, e2, ..., en) such that
        e1 + e2 + ... + en <= max_degree, where each ei >= 0.
        
        Args:
            num_vars: Number of variables
            max_degree: Maximum total degree
            
        Returns:
            List of Monomial objects in graded lexicographic order
        """
        monomials = []
        
        def generate_exponents_recursive(current_exponents: List[int], remaining_vars: int, remaining_degree: int):
            """
            Recursively generate all valid exponent combinations.
            
            Args:
                current_exponents: Exponents assigned so far
                remaining_vars: Number of variables left to assign exponents to
                remaining_degree: Maximum degree left to distribute
            """
            if remaining_vars == 0:
                # Base case: all variables have been assigned exponents
                monomials.append(Monomial(tuple(current_exponents)))
                return
            
            if remaining_vars == 1:
                # Last variable: assign all remaining degree to it (can be 0 to remaining_degree)
                for exp in range(remaining_degree + 1):
                    generate_exponents_recursive(current_exponents + [exp], remaining_vars - 1, remaining_degree - exp)
                return
            
            # General case: try all possible exponents for current variable
            for exp in range(remaining_degree + 1):
                generate_exponents_recursive(
                    current_exponents + [exp], 
                    remaining_vars - 1, 
                    remaining_degree - exp
                )
        
        # Start the recursive generation
        generate_exponents_recursive([], num_vars, max_degree)
        
        # Sort by total degree first, then lexicographically (graded lex order)
        # This gives the standard ordering: constant, x1, x2, x1^2, x1*x2, x2^2, etc.
        monomials.sort(key=lambda m: (m.degree, m.exponents))
        
        return monomials

    def predict_basis(self, **kwargs) -> Dict:
        """
        Predict basis using all monomials up to the specified degree.
        
        Args:
            **kwargs: Additional arguments (poly may be needed for basis extension)
            
        Returns:
            Dictionary containing:
                - 'basis': List of Monomial objects
                - 'time': Time taken for oracle computation
                - 'basis_extension_time': Time for basis extension (if used)
        """
        start_time = time.time()
        
        # Generate all monomials up to degree
        predicted_basis = self._generate_all_monomials_up_to_degree(self.num_vars, self.max_degree)
        
        oracle_time = time.time() - start_time
        
        result = {
            'basis': predicted_basis,
            'time': oracle_time
        }
        
        # Apply basis extension if requested
        if self.use_basis_extension:
            from utils.basis_extension import basis_extension
            poly = kwargs.get('poly')
            if poly is None:
                raise ValueError("DegreeOracle with basis extension requires the 'poly' argument (Polynomial object)")
            
            ext_start = time.time()
            extended_basis = basis_extension(predicted_basis, poly, **self.basis_extension_params)
            ext_time = time.time() - ext_start
            
            result['basis'] = extended_basis
            result['basis_extension_time'] = ext_time
        
        return result
