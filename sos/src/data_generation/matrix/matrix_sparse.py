import numpy as np
from fractions import Fraction
from typing import Optional, Union, Tuple
from sos_transformer.data_generation.matrix.matrix_base import PSDMatrixSampler

class SparsePSDSampler(PSDMatrixSampler):
    """
    A PSD matrix sampler that generates matrices by multiplying
    a sparse random matrix by its transpose: A = Q Q^T.
    The sparsity of Q can be controlled via the sparsity parameter.
    Supports both floating point and rational number generation.
    """
    
    def __init__(self, 
                 min_eigenval: float = 0.0,
                 scale: float = 1.0,
                 sparsity: float = 0.5,
                 random_state: Optional[int] = None,
                 rational: bool = False,
                 max_numerator: int = 10,
                 max_denominator: int = 10,
                 min_sparsity: float = 0.05,
                 max_sparsity: float = 0.5):
        """
        Initialize the sampler.
        
        Args:
            min_eigenval: Minimum eigenvalue (adds to diagonal if needed)
            scale: Scale factor for the random entries
            sparsity: Fraction of entries that should be non-zero (between 0 and 1)
            random_state: Optional random seed for reproducibility
            rational: If True, generate rational-valued PSD matrices
            max_numerator: Maximum absolute value for numerators in rational mode
            max_denominator: Maximum value for denominators in rational mode
        """
        super().__init__(min_eigenval)
        if not 0 <= sparsity <= 1:
            raise ValueError("Sparsity must be between 0 and 1")
        
        self.scale = scale
        self.sparsity = sparsity
        self.rng = np.random.RandomState(random_state)
        self.rational = rational
        self.max_numerator = max_numerator
        self.max_denominator = max_denominator
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity

    def _generate_rational_matrix(self, rows: int, cols: int) -> np.ndarray:
        """
        Generate a sparse matrix of rational numbers with controlled numerators and denominators.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            numpy array of Fraction objects with controlled sparsity
        """
        # Generate random integers for numerators
        numerators = self.rng.randint(-self.max_numerator, self.max_numerator + 1, (rows, cols))
        
        # Generate random integers for denominators (excluding 0)
        denominators = self.rng.randint(1, self.max_denominator + 1, (rows, cols))
        
        # Generate sparsity mask
        mask = self.rng.random((rows, cols)) > self.sparsity
        
        # Create matrix of Fraction objects with sparsity
        Q = np.zeros((rows, cols), dtype=object)
        for i in range(rows):
            for j in range(cols):
                if not mask[i,j]:  # Only set non-zero values based on sparsity
                    Q[i,j] = Fraction(numerators[i,j], denominators[i,j])
                else:
                    Q[i,j] = Fraction(0)
        
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
        Sample a sparse PSD matrix using random matrix multiplication.
        
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

            # randomly determine sparsity
            sparsity = self.rng.uniform(self.min_sparsity, self.max_sparsity)
            
            # Generate mask for which entries should be zero
            mask = self.rng.random(Q.shape) > sparsity
            Q[mask] = 0
            
            A = Q @ Q.T
            
            if self.min_eigenval > 0:
                U, s, Vt = np.linalg.svd(A)
                s = np.maximum(s, self.min_eigenval)
                A = U @ np.diag(s) @ Vt
            
            #print(f"A: {A}")
            
            return A
        else:
            # Rational implementation with sparsity
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

    def get_sparsity(self, matrix: np.ndarray, tol: float = 1e-10) -> float:
        """
        Compute the sparsity (fraction of zero entries) of a matrix.
        
        Args:
            matrix: Input matrix
            tol: Tolerance for considering an entry as zero
            
        Returns:
            Fraction of entries that are zero
        """
        return np.sum(np.abs(matrix) < tol) / matrix.size


if __name__ == "__main__":
    # Example usage with floating-point sparse matrices
    sampler = SparsePSDSampler(
        min_eigenval=0.1,
        scale=1.0,
        sparsity=0.8,  # 80% of entries in Q will be non-zero
        random_state=42
    )
    
    # Generate a 4x4 sparse PSD matrix
    A = sampler.sample(size=4)
    print("Generated sparse PSD matrix (float):")
    print(A)
    print("\nVerifying PSD property:", sampler.verify_psd(A))
    print("Matrix sparsity:", sampler.get_sparsity(A))
    
    # Example with rational sparse matrices
    rational_sampler = SparsePSDSampler(
        min_eigenval=0.1,
        scale=1.0,
        sparsity=0.7,  # 70% of entries in Q will be non-zero
        random_state=42,
        rational=True,
        max_numerator=5,
        max_denominator=5
    )
    
    # Generate a 3x3 rational sparse PSD matrix
    num_matrix, den_matrix = rational_sampler.sample(size=3)
    print("\nGenerated rational sparse PSD matrix:")
    print("Numerators:")
    print(num_matrix)
    print("\nDenominators:")
    print(den_matrix)
    print("\nVerifying PSD property:", rational_sampler.verify_rational_psd(num_matrix, den_matrix))
    print("Matrix sparsity:", rational_sampler.get_sparsity(num_matrix))
    
    # Print the fractions
    print("\nFraction representation:")
    for i in range(3):
        for j in range(3):
            print(f"{num_matrix[i,j]}/{den_matrix[i,j]}", end=" ")
        print()
    
    # Example with very sparse rational matrices
    very_sparse_rational = SparsePSDSampler(
        min_eigenval=0.1,
        scale=1.0,
        sparsity=0.25,  # only 20% of entries in Q will be non-zero
        random_state=42,
        rational=True,
        max_numerator=3,
        max_denominator=3
    )
    
    # Generate a larger sparse rational matrix
    num_sparse, den_sparse = very_sparse_rational.sample(size=5)
    print("\nGenerated very sparse 5x5 rational PSD matrix:")
    print("Numerators:")
    print(num_sparse)
    print("\nDenominators:")
    print(den_sparse)
    print("\nVerifying PSD property:", very_sparse_rational.verify_rational_psd(num_sparse, den_sparse))
    print("Matrix sparsity:", very_sparse_rational.get_sparsity(num_sparse))