import numpy as np
from fractions import Fraction
from typing import Optional, Union, Tuple
from sos_transformer.data_generation.matrix.matrix_base import PSDMatrixSampler

class BlockDiagonalPSDSampler(PSDMatrixSampler):
    """
    A PSD matrix sampler that generates block diagonal matrices.
    Each block is independently generated as PSD using Q @ Q.T construction.
    Block sizes are randomly sampled up to a maximum size.
    """
    
    def __init__(self, 
                 min_eigenval: float = 0.0,
                 max_block_size: int = 3,
                 scale: float = 1.0,
                 random_state: Optional[int] = None,
                 rational: bool = False,
                 max_numerator: int = 10,
                 max_denominator: int = 10):
        """
        Initialize the block diagonal sampler.
        
        Args:
            min_eigenval: Minimum eigenvalue for blocks
            max_block_size: Maximum size of individual blocks
            scale: Scale factor for random entries in blocks
            random_state: Optional random seed for reproducibility
            rational: If True, generate rational-valued PSD matrices
            max_numerator: Maximum absolute value for numerators in rational mode
            max_denominator: Maximum value for denominators in rational mode
        """
        super().__init__(min_eigenval)
        self.max_block_size = max_block_size
        self.scale = scale
        self.rng = np.random.RandomState(random_state)
        self.rational = rational
        self.max_numerator = max_numerator
        self.max_denominator = max_denominator

    def _generate_block_sizes(self, total_size: int) -> list[int]:
        """Generate a list of block sizes that sum to total_size."""
        block_sizes = []
        remaining = total_size
        
        while remaining > 0:
            # Sample block size up to min(max_block_size, remaining)
            max_allowed = min(self.max_block_size, remaining)
            block_size = self.rng.randint(1, max_allowed + 1)
            block_sizes.append(block_size)
            remaining -= block_size
            
        return block_sizes

    def _generate_psd_block(self, size: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate a single PSD block using Q @ Q.T construction."""
        if not self.rational:
            # Float implementation
            Q = self.rng.randn(size, size) * self.scale
            block = Q @ Q.T
            
            # Ensure minimum eigenvalue if needed
            if self.min_eigenval > 0:
                eigenvals, eigenvecs = np.linalg.eigh(block)
                eigenvals = np.maximum(eigenvals, self.min_eigenval)
                block = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                
            return block
        else:
            # Rational implementation
            Q = self._generate_rational_matrix(size, size)
            Q_T = Q.T
            block = self._rational_matrix_multiply(Q, Q_T)
            
            # Convert to numerator and denominator matrices
            num_matrix = np.zeros((size, size), dtype=int)
            den_matrix = np.zeros((size, size), dtype=int)
            
            for i in range(size):
                for j in range(size):
                    num_matrix[i,j] = block[i,j].numerator
                    den_matrix[i,j] = block[i,j].denominator
                    
            return num_matrix, den_matrix

    def _generate_rational_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Generate a matrix of rational numbers."""
        numerators = self.rng.randint(-self.max_numerator, self.max_numerator + 1, (rows, cols))
        denominators = self.rng.randint(1, self.max_denominator + 1, (rows, cols))
        
        Q = np.zeros((rows, cols), dtype=object)
        for i in range(rows):
            for j in range(cols):
                Q[i,j] = Fraction(numerators[i,j], denominators[i,j])
        
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
        Sample a block diagonal PSD matrix.
        
        Args:
            size: Size of the output matrix (n x n)
            rank: Not used for block diagonal (determined by block structure)
        
        Returns:
            If rational=False:
                A block diagonal positive semidefinite matrix of shape (size, size)
            If rational=True:
                Tuple of (numerator_matrix, denominator_matrix) where both are integer numpy arrays
        """
        block_sizes = self._generate_block_sizes(size)
        
        if not self.rational:
            # Float implementation
            A = np.zeros((size, size))
            offset = 0
            
            for block_size in block_sizes:
                block = self._generate_psd_block(block_size)
                A[offset:offset+block_size, offset:offset+block_size] = block
                offset += block_size
                
            return A
        else:
            # Rational implementation
            num_matrix = np.zeros((size, size), dtype=int)
            den_matrix = np.ones((size, size), dtype=int)  # Initialize with 1s for proper fractions
            offset = 0
            
            for block_size in block_sizes:
                block_num, block_den = self._generate_psd_block(block_size)
                num_matrix[offset:offset+block_size, offset:offset+block_size] = block_num
                den_matrix[offset:offset+block_size, offset:offset+block_size] = block_den
                offset += block_size
                
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
    sampler = BlockDiagonalPSDSampler(
        min_eigenval=0.1,
        max_block_size=3,
        scale=1.0,
        random_state=42
    )
    
    # Generate a 10x10 block diagonal PSD matrix
    A = sampler.sample(size=10)
    print("Generated block diagonal PSD matrix (float):")
    print(A)
    print("\nVerifying PSD property:", sampler.verify_psd(A))
    print("Matrix rank:", sampler.get_rank(A))
    print("Eigenvalues:", np.sort(np.linalg.eigvalsh(A))[::-1])
    
    # Example with rational matrices
    rational_sampler = BlockDiagonalPSDSampler(
        min_eigenval=0.0,
        max_block_size=2,
        random_state=42,
        rational=True,
        max_numerator=3,
        max_denominator=4
    )
    
    # Generate a 5x5 rational block diagonal PSD matrix
    num_matrix, den_matrix = rational_sampler.sample(size=5)
    print("\nGenerated rational block diagonal PSD matrix:")
    print("Numerators:")
    print(num_matrix)
    print("\nDenominators:")
    print(den_matrix)
    print("\nVerifying PSD property:", rational_sampler.verify_rational_psd(num_matrix, den_matrix))
    print("Matrix rank:", rational_sampler.get_rank(num_matrix.astype(float) / den_matrix.astype(float)))
