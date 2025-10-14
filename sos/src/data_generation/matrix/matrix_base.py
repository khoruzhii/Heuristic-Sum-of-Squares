import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

class PSDMatrixSampler(ABC):
    """Abstract base class for sampling positive semidefinite matrices."""
    
    def __init__(self, min_eigenval: float = 0.0, rational: bool = False):
        """
        Initialize the PSD matrix sampler.
        
        Args:
            min_eigenval: Minimum eigenvalue for the generated matrices.
                         Default is 0.0 for positive semidefinite.
                         Set > 0 for positive definite matrices.
            rational: Whether to sample rational matrices.
        """
        self.min_eigenval = min_eigenval
        self.rational = rational
    
    @abstractmethod
    def sample(self, size: int, rank: Optional[int] = None) -> np.ndarray:
        """
        Sample a positive semidefinite matrix.
        
        Args:
            size: Size of the matrix (n x n)
            rank: Optional target rank of the matrix. If None, full rank is used.
                 Must be <= size.
        
        Returns:
            A positive semidefinite matrix of shape (size, size)
        """
        pass
    
    def verify_psd(self, matrix: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Verify that a matrix is positive semidefinite.
        
        Args:
            matrix: Matrix to verify
            tol: Tolerance for eigenvalue comparison
            
        Returns:
            True if the matrix is PSD (all eigenvalues >= -tol)
        """
        if not np.allclose(matrix, matrix.T):
            return False
        eigvals = np.linalg.eigvalsh(matrix)
        return np.all(eigvals >= -tol) 