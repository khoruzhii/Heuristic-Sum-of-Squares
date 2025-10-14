import numpy as np
from fractions import Fraction
from typing import Optional, Union, Tuple
from sos_transformer.data_generation.matrix.matrix_base import PSDMatrixSampler

class LowRankPSDSampler(PSDMatrixSampler):
    """
    A PSD matrix sampler that generates low-rank dense matrices using SVD decomposition.
    Constructs matrices as A = U @ diag(eigenvals) @ U.T where U has orthonormal columns.
    All entries are typically non-zero (dense), with controllable rank.
    Supports both floating point and rational number generation.
    """
    
    def __init__(self, 
                 min_eigenval: float = 0.0,
                 max_eigenval: float = 1.0,
                 random_state: Optional[int] = None,
                 rational: bool = False,
                 max_numerator: int = 10,
                 max_denominator: int = 10,
                 max_rank: int = 3,
                 min_rank: int = 1):
        """
        Initialize the sampler.
        
        Args:
            min_eigenval: Minimum eigenvalue for non-zero eigenvalues
            max_eigenval: Maximum eigenvalue for the generated matrices
            random_state: Optional random seed for reproducibility
            rational: If True, generate rational-valued PSD matrices
            max_numerator: Maximum absolute value for numerators in rational mode
            max_denominator: Maximum value for denominators in rational mode
            max_rank: Maximum rank for generated matrices (default 3)
            min_rank: Minimum rank for generated matrices (default 1)
        """
        super().__init__(min_eigenval)
        if max_eigenval < min_eigenval:
            raise ValueError("max_eigenval must be >= min_eigenval")
        
        self.max_eigenval = max_eigenval
        self.rng = np.random.RandomState(random_state)
        self.rational = rational
        self.max_numerator = max_numerator
        self.max_denominator = max_denominator
        self.max_rank = max_rank
        self.min_rank = min_rank    

    def _generate_orthonormal_matrix(self, size: int, rank: int) -> np.ndarray:
        """Generate a matrix with orthonormal columns using QR decomposition."""
        # Generate random matrix and orthogonalize
        A = self.rng.randn(size, rank)
        Q, _ = np.linalg.qr(A)
        return Q

    def _generate_rational_orthonormal(self, size: int, rank: int) -> np.ndarray:
        """Generate rational orthonormal matrix by starting with integer matrix."""
        # Generate integer matrix with small values for better rational representation
        A = self.rng.randint(-self.max_numerator//2, self.max_numerator//2 + 1, (size, rank))
        
        # Convert to fractions and apply Gram-Schmidt
        Q = np.zeros((size, rank), dtype=object)
        
        for j in range(rank):
            # Start with j-th column as fractions
            v = np.array([Fraction(A[i, j], self.max_denominator) for i in range(size)])
            
            # Subtract projections onto previous columns
            for k in range(j):
                # Compute dot product
                dot_prod = Fraction(0)
                for i in range(size):
                    dot_prod += v[i] * Q[i, k]
                
                # Subtract projection
                for i in range(size):
                    v[i] -= dot_prod * Q[i, k]
            
            # Normalize (approximate for rational numbers)
            norm_sq = Fraction(0)
            for i in range(size):
                norm_sq += v[i] * v[i]
            
            # Simple normalization for rational case
            if norm_sq > 0:
                for i in range(size):
                    Q[i, j] = v[i] / Fraction(int(np.sqrt(float(norm_sq)) * self.max_denominator), self.max_denominator)
            else:
                # Fallback to unit vector
                Q[j % size, j] = Fraction(1)
        
        return Q

    def _rational_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Multiply two matrices of Fraction objects."""
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
        Sample a low-rank dense PSD matrix using SVD construction.
        
        Args:
            size: Size of the output matrix (n x n)
            rank: Target rank. If None, samples a rank between min_rank and max_rank.
                 If specified, must be <= size.
        
        Returns:
            If rational=False:
                A positive semidefinite matrix of shape (size, size)
            If rational=True:
                Tuple of (numerator_matrix, denominator_matrix) where both are integer numpy arrays
        """
        # Determine rank
        if rank is None:
            # Ensure rank doesn't exceed matrix size
            effective_max_rank = min(self.max_rank, size)
            effective_min_rank = min(self.min_rank, effective_max_rank)
            rank = self.rng.randint(effective_min_rank, effective_max_rank + 1)
        else:
            if rank > size:
                raise ValueError(f"rank ({rank}) cannot exceed matrix size ({size})")
            if rank < 1:
                raise ValueError(f"rank ({rank}) must be at least 1")

        if not self.rational:
            # Generate orthonormal matrix
            U = self._generate_orthonormal_matrix(size, rank)
            
            # Generate eigenvalues
            eigenvals = self.rng.uniform(self.min_eigenval, self.max_eigenval, rank)
            eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
            
            # Construct PSD matrix: A = U @ diag(eigenvals) @ U.T
            A = U @ np.diag(eigenvals) @ U.T
            
            return A
        else:
            # Rational implementation
            U = self._generate_rational_orthonormal(size, rank)
            
            # Generate rational eigenvalues
            eigenvals = np.zeros(rank, dtype=object)
            for i in range(rank):
                num = self.rng.randint(max(1, int(self.min_eigenval * self.max_denominator)), 
                                     self.max_numerator + 1)
                den = self.rng.randint(1, self.max_denominator + 1)
                eigenvals[i] = Fraction(num, den)
            
            # Create diagonal matrix
            D = np.zeros((rank, rank), dtype=object)
            for i in range(rank):
                for j in range(rank):
                    D[i,j] = eigenvals[i] if i == j else Fraction(0)
            
            # Construct A = U @ D @ U.T
            UD = self._rational_matrix_multiply(U, D)
            A = self._rational_matrix_multiply(UD, U.T)
            
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
    # Example usage with floating-point matrices
    sampler = LowRankPSDSampler(
        min_eigenval=0.1,
        max_eigenval=2.0,
        random_state=42
    )
    
    # Generate a 5x5 low-rank dense PSD matrix
    A = sampler.sample(size=5, rank=3)
    print("Generated low-rank dense PSD matrix (float):")
    print(A)
    print("\nVerifying PSD property:", sampler.verify_psd(A))
    print("Matrix rank:", sampler.get_rank(A))
    print("Eigenvalues:", np.sort(np.linalg.eigvalsh(A))[::-1])
    print("Matrix density (fraction non-zero):", np.mean(np.abs(A) > 1e-10))
    
    # Example with rational matrices
    rational_sampler = LowRankPSDSampler(
        min_eigenval=0.1,
        max_eigenval=2.0,
        random_state=42,
        rational=True,
        max_numerator=3,
        max_denominator=4
    )
    
    # Generate a 3x3 rational low-rank dense PSD matrix
    num_matrix, den_matrix = rational_sampler.sample(size=3, rank=2)
    print("\nGenerated rational low-rank dense PSD matrix:")
    print("Numerators:")
    print(num_matrix)
    print("\nDenominators:")
    print(den_matrix)
    print("\nVerifying PSD property:", rational_sampler.verify_rational_psd(num_matrix, den_matrix))
    print("Matrix rank:", rational_sampler.get_rank(num_matrix.astype(float) / den_matrix.astype(float)))
