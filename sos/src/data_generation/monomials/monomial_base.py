import numpy as np
from abc import ABC, abstractmethod
from typing import List
from .monomials import Monomial, MonomialBasis

class BasisSampler(ABC):
    """Abstract base class for sampling monomial bases."""
    @abstractmethod
    def sample(self, num_vars: int, max_degree: int, num_monomials: int) -> MonomialBasis:
        pass


