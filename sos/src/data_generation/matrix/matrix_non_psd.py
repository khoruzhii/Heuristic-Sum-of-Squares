import numpy as np
from fractions import Fraction
from typing import Optional, Union, Tuple
from sos_transformer.data_generation.matrix.matrix_base import PSDMatrixSampler

class SimpleRandomNonPSDSampler(PSDMatrixSampler):
    """
    A simple non-PSD matrix sampler that generates matrices by starting with 
    a PSD structure and then introducing subtle negative eigenvalues.
    Uses the approach: A = Q Q^T - small_perturbation to make it non-PSD.
    Supports both floating point and rational number generation.
    """
    
    def __init__(self, 
                 min_eigenval: float = -0.8,
                 scale: float = 1.0,
                 random_state: Optional[int] = None,
                 rational: bool = False,
                 max_numerator: int = 10,
                 max_denominator: int = 10,
                 perturbation_strength: float = 0.25):
        """
        Initialize the sampler.
        
        Args:
            min_eigenval: Most negative eigenvalue allowed (should be < 0 for non-PSD)
            scale: Scale factor for the random entries
            random_state: Optional random seed for reproducibility
            rational: If True, generate rational-valued non-PSD matrices
            max_numerator: Maximum absolute value for numerators in rational mode
            max_denominator: Maximum value for denominators in rational mode
            perturbation_strength: Controls how negative the eigenvalues can be
        """
        super().__init__(min_eigenval)
        self.scale = scale
        self.rng = np.random.RandomState(random_state)
        self.rational = rational
        self.max_numerator = max_numerator
        self.max_denominator = max_denominator
        self.perturbation_strength = perturbation_strength

    def _generate_rational_matrix(self, rows: int, cols: int) -> np.ndarray:
        """
        Generate a matrix of rational numbers with controlled numerators and denominators.
        
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

    def _make_non_psd(self, matrix: np.ndarray) -> np.ndarray:
        """
        Convert a PSD matrix to non-PSD by introducing negative eigenvalues.
        Uses eigenvalue decomposition and modifies some eigenvalues to be negative.
        """
        # Get eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(matrix)
        
        # Make some eigenvalues negative (but not too many to keep it subtle)
        n_negative = max(1, int(len(eigvals) * 0.2))  # Make ~20% of eigenvalues negative
        negative_indices = self.rng.choice(len(eigvals), size=n_negative, replace=False)
        
        for idx in negative_indices:
            # Make eigenvalue negative, but not too extreme
            eigvals[idx] = -abs(eigvals[idx]) * self.perturbation_strength
        
        # Reconstruct matrix
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _make_rational_non_psd(self, A: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert rational PSD matrix to non-PSD and return as numerator/denominator matrices.
        """
        # Convert to float for eigenvalue manipulation
        float_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                float_matrix[i,j] = float(A[i,j])
        
        # Make non-PSD
        non_psd_float = self._make_non_psd(float_matrix)
        
        # Convert back to rational representation with controlled precision
        num_matrix = np.zeros((size, size), dtype=int)
        den_matrix = np.zeros((size, size), dtype=int)
        
        for i in range(size):
            for j in range(size):
                # Use limited precision to get reasonable fractions
                frac = Fraction(non_psd_float[i,j]).limit_denominator(self.max_denominator)
                num_matrix[i,j] = frac.numerator
                den_matrix[i,j] = frac.denominator
        
        return num_matrix, den_matrix

    def sample(self, size: int, rank: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Sample a non-PSD matrix by starting with PSD structure and introducing negative eigenvalues.
        
        Args:
            size: Size of the output matrix (n x n)
            rank: Optional target rank. If None, uses full rank (size).
                 If specified, must be <= size.
        
        Returns:
            If rational=False:
                A non-positive semidefinite matrix of shape (size, size)
            If rational=True:
                Tuple of (numerator_matrix, denominator_matrix) where both are integer numpy arrays
        """
        if rank is None:
            rank = size
        elif rank > size:
            raise ValueError(f"Rank ({rank}) cannot exceed matrix size ({size})")

        if not self.rational:
            # Generate initial PSD matrix
            Q = self.rng.randn(size, rank) * self.scale
            A = Q @ Q.T
            
            # Make it non-PSD
            A = self._make_non_psd(A)
            
            return A
        else:
            # Rational implementation
            Q = self._generate_rational_matrix(size, rank)
            Q_T = Q.T
            A = self._rational_matrix_multiply(Q, Q_T)
            
            # Convert to non-PSD rational representation
            num_matrix, den_matrix = self._make_rational_non_psd(A, size)
            
            return num_matrix, den_matrix

    def verify_rational_psd(self, num_matrix: np.ndarray, den_matrix: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Verify that a rational matrix given by numerator and denominator matrices is PSD.
        (This should return False for matrices generated by this sampler)
        
        Args:
            num_matrix: Matrix of numerators
            den_matrix: Matrix of denominators
            tol: Tolerance for eigenvalue checking
            
        Returns:
            True if the matrix is PSD (should be False for this sampler)
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
    sampler = SimpleRandomNonPSDSampler(
        min_eigenval=-0.1, 
        scale=1.0, 
        random_state=42,
        rational=True,
        max_numerator=5,
        max_denominator=5,
        perturbation_strength=0.3
    )
    
    # Generate a 3x3 rational non-PSD matrix
    num_matrix, den_matrix = sampler.sample(size=3)
    
    print("Generated rational non-PSD matrix:")
    print("Numerators:")
    print(num_matrix)
    print("\nDenominators:")
    print(den_matrix)
    print("\nVerifying PSD property (should be False):", sampler.verify_rational_psd(num_matrix, den_matrix))
    
    # Print the fractions
    print("\nFraction representation:")
    for i in range(3):
        for j in range(3):
            print(f"{num_matrix[i,j]}/{den_matrix[i,j]}", end=" ")
        print()
    
    # Example usage with floating point
    sampler = SimpleRandomNonPSDSampler(min_eigenval=-0.1, scale=1.0, random_state=42)
    
    # Generate a 4x4 non-PSD matrix
    A = sampler.sample(size=4)
    
    print("\nGenerated non-PSD matrix:")
    print(A)
    print("\nVerifying PSD property (should be False):", sampler.verify_psd(A))
    print("\nEigenvalues (some should be negative):", np.linalg.eigvalsh(A))
    print("Rank:", sampler.get_rank(A))
    
    # Generate a rank-2 non-PSD matrix
    A_lowrank = sampler.sample(size=4, rank=2)
    print("\nGenerated rank-2 non-PSD matrix:")
    print(A_lowrank)
    print("\nVerifying PSD property (should be False):", sampler.verify_psd(A_lowrank))
    print("\nEigenvalues (some should be negative):", np.linalg.eigvalsh(A_lowrank))
    print("Rank:", sampler.get_rank(A_lowrank))

    # Test with small matrix to show non-PSD property clearly
    sampler = SimpleRandomNonPSDSampler(
        rational=True,
        max_numerator=2,
        max_denominator=3,
        perturbation_strength=0.5
    )
    num_matrix, den_matrix = sampler.sample(size=3)
    print("\nSmall rational non-PSD matrix:")
    print("Numerators:")
    print(num_matrix)
    print("\nDenominators:")
    print(den_matrix)
    print("\nVerifying PSD property (should be False):", sampler.verify_rational_psd(num_matrix, den_matrix))
    
    # Show eigenvalues of the float version
    float_matrix = num_matrix.astype(float) / den_matrix.astype(float)
    print("\nEigenvalues of float version:", np.linalg.eigvalsh(float_matrix))
