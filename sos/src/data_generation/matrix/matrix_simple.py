import numpy as np
from fractions import Fraction
from typing import Optional, Union, Tuple
from sos_transformer.data_generation.matrix.matrix_base import PSDMatrixSampler

class SimpleRandomPSDSampler(PSDMatrixSampler):
    """
    A simple PSD matrix sampler that generates matrices by multiplying
    a random matrix by its transpose: A = Q Q^T.
    Supports both floating point and rational number generation.
    """
    
    def __init__(self, 
                 min_eigenval: float = 0.0,
                 scale: float = 1.0,
                 random_state: Optional[int] = None,
                 rational: bool = False,
                 max_numerator: int = 10,
                 max_denominator: int = 10):
        """
        Initialize the sampler.
        
        Args:
            min_eigenval: Minimum eigenvalue (adds to diagonal if needed)
            scale: Scale factor for the random entries
            random_state: Optional random seed for reproducibility
            rational: If True, generate rational-valued PSD matrices
            max_numerator: Maximum absolute value for numerators in rational mode
            max_denominator: Maximum value for denominators in rational mode
        """
        super().__init__(min_eigenval)
        self.scale = scale
        self.rng = np.random.RandomState(random_state)
        self.rational = rational
        self.max_numerator = max_numerator
        self.max_denominator = max_denominator

        #print("rational matrix:", self.rational)

    def _generate_rational_matrix(self, rows: int, cols: int) -> np.ndarray:
        """
        Generate a matrix of rational numbers with controlled numerators and denominators.
        The matrix is structured to ensure PSD property after multiplication.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            numpy array of Fraction objects
        """
        # Generate random integers for numerators
        numerators = self.rng.randint(-self.max_numerator, self.max_numerator + 1, (rows, cols))
        
        # Generate random integers for denominators (excluding 0)
        denominators = self.rng.randint(1, self.max_denominator + 1, (rows, cols))
        
        # Create matrix of Fraction objects
        Q = np.zeros((rows, cols), dtype=object)
        for i in range(rows):
            for j in range(cols):
                Q[i,j] = Fraction(numerators[i,j], denominators[i,j])
        
        return Q

    def _rational_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Multiply two matrices of Fraction objects.
        """
        rows, inner = A.shape
        cols = B.shape[1]
        result = np.zeros((rows, cols), dtype=object)
        
        for i in range(rows):
            for j in range(cols):
                sum_frac = Fraction(0)
                for k in range(inner):
                    sum_frac += A[i,k] * B[k,j]
                result[i,j] = sum_frac
        
        return result

    def sample(self, size: int, rank: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Sample a PSD matrix using random matrix multiplication.
        
        Args:
            size: Size of the output matrix (n x n)
            rank: Optional target rank. If None, uses full rank (size).
                 If specified, must be <= size.
        
        Returns:
            If rational=False:
                A positive semidefinite matrix of shape (size, size)
            If rational=True:
                Tuple of (numerator_matrix, denominator_matrix) where both are integer numpy arrays
        """
        if rank is None:
            rank = size
        elif rank > size:
            raise ValueError(f"Rank ({rank}) cannot exceed matrix size ({size})")

        if not self.rational:
            # Original floating-point implementation
            Q = self.rng.randn(size, rank) * self.scale
            A = Q @ Q.T
            
            if self.min_eigenval > 0:
                U, s, Vt = np.linalg.svd(A)
                s = np.zeros_like(s)
                s[:rank] = np.maximum(s[:rank], self.min_eigenval)
                A = U @ np.diag(s) @ Vt
            
            return A
        else:
            # Rational implementation
            Q = self._generate_rational_matrix(size, rank)
            Q_T = Q.T
            A = self._rational_matrix_multiply(Q, Q_T)
            
            # Convert to separate numerator and denominator matrices
            num_matrix = np.zeros((size, size), dtype=int)
            den_matrix = np.zeros((size, size), dtype=int)
            
            for i in range(size):
                for j in range(size):
                    num_matrix[i,j] = A[i,j].numerator
                    den_matrix[i,j] = A[i,j].denominator
            
            return num_matrix, den_matrix

    def verify_rational_psd(self, num_matrix: np.ndarray, den_matrix: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Verify that a rational matrix given by numerator and denominator matrices is PSD.
        
        Args:
            num_matrix: Matrix of numerators
            den_matrix: Matrix of denominators
            tol: Tolerance for eigenvalue checking
            
        Returns:
            True if the matrix is PSD
        """
        # Convert to floating point for eigenvalue computation
        float_matrix = num_matrix.astype(float) / den_matrix.astype(float)
        return self.verify_psd(float_matrix, tol)

    def get_rank(self, matrix: np.ndarray, tol: float = 1e-10) -> int:
        """
        Compute the numerical rank of a matrix.
        
        Args:
            matrix: Input matrix
            tol: Tolerance for singular values
            
        Returns:
            Numerical rank of the matrix
        """
        s = np.linalg.svd(matrix, compute_uv=False)
        return np.sum(s > tol)


if __name__ == "__main__":
    # Example usage with rational matrices
    sampler = SimpleRandomPSDSampler(
        min_eigenval=0.1, 
        scale=1.0, 
        random_state=42,
        rational=True,
        max_numerator=5,
        max_denominator=5
    )
    
    # Generate a 3x3 rational PSD matrix
    num_matrix, den_matrix = sampler.sample(size=3)
    
    print("Generated rational PSD matrix:")
    print("Numerators:")
    print(num_matrix)
    print("\nDenominators:")
    print(den_matrix)
    print("\nVerifying PSD property:", sampler.verify_rational_psd(num_matrix, den_matrix))
    
    # Print the fractions
    print("\nFraction representation:")
    for i in range(3):
        for j in range(3):
            print(f"{num_matrix[i,j]}/{den_matrix[i,j]}", end=" ")
        print()
    
    # Example usage
    sampler = SimpleRandomPSDSampler(min_eigenval=0.1, scale=1.0, random_state=42)
    
    # Generate a 4x4 PSD matrix
    A = sampler.sample(size=4)
    
    print("\nGenerated PSD matrix:")
    print(A)
    print("\nVerifying PSD property:", sampler.verify_psd(A))
    print("\nEigenvalues:", np.linalg.eigvalsh(A))
    print("Rank:", sampler.get_rank(A))
    
    # Generate a rank-2 PSD matrix
    A_lowrank = sampler.sample(size=4, rank=2)
    print("\nGenerated rank-2 PSD matrix:")
    print(A_lowrank)
    print("\nVerifying PSD property:", sampler.verify_psd(A_lowrank))
    print("\nEigenvalues:", np.linalg.eigvalsh(A_lowrank))
    print("Rank:", sampler.get_rank(A_lowrank)) 

    sampler = SimpleRandomPSDSampler(
    rational=True,
    max_numerator=2,
    max_denominator=3
    )
    num_matrix, den_matrix = sampler.sample(size=25)
    print("Generated rational PSD matrix:")
    print("Numerators:")
    print(num_matrix)
    print("\nDenominators:")
    print(den_matrix)
    print("\nVerifying PSD property:", sampler.verify_rational_psd(num_matrix, den_matrix))