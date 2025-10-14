from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
import numpy as np
from sos_transformer.data_generation.monomials.monomials import Polynomial, Monomial, MonomialBasis

class SDPSolver(ABC):
    """Abstract base class for SDP solvers."""
    
    @abstractmethod
    def solve_sos_feasibility(self, 
                            poly: Polynomial, 
                            basis: Optional[MonomialBasis] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if a polynomial is SOS by solving a feasibility SDP.
        
        Args:
            poly: The polynomial to check
            basis: Optional monomial basis to use. If None, generates a complete basis.
                  If provided, attempts to find an SOS decomposition using only these basis elements.
            
        Returns:
            Tuple of (is_sos, gram_matrix) where:
                is_sos: True if polynomial is SOS using the given/generated basis
                gram_matrix: If is_sos is True, a PSD matrix Q such that p(x) = z^T Q z
                           where z is the monomial basis. None if is_sos is False.
        """
        pass
    
    @abstractmethod
    def get_sos_decomposition(self, 
                            poly: Polynomial, 
                            Q: np.ndarray,
                            basis: Optional[MonomialBasis] = None) -> str:
        """
        Given a polynomial and its Gram matrix, return the SOS decomposition.
        
        Args:
            poly: The polynomial
            Q: The Gram matrix from solve_sos_feasibility
            basis: The monomial basis used in the decomposition. If None,
                  uses the same basis generation as solve_sos_feasibility.
            
        Returns:
            String representation of the SOS decomposition p(x) = sum_i s_i(x)^2
        """
        pass 