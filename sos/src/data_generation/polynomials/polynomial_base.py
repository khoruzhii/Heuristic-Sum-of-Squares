from abc import ABC, abstractmethod
from typing import Optional, List
from sos_transformer.data_generation.monomials.monomials import Polynomial, MonomialBasis

class PolynomialSampler(ABC):
    """Abstract base class for sampling polynomials."""
    
    @abstractmethod
    def sample(self, num_vars: int, max_degree: int) -> Polynomial:
        """
        Sample a polynomial.
        
        Args:
            num_vars: Number of variables in the polynomial
            max_degree: Maximum degree of the polynomial
            
        Returns:
            A Polynomial object representing the sampled polynomial
        """
        pass
    
    