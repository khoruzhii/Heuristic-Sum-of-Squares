import numpy as np
from fractions import Fraction
from typing import Dict, Optional, Union, Tuple
from data_generation.polynomials.polynomial_base import PolynomialSampler
from data_generation.matrix.matrix_base import PSDMatrixSampler
from data_generation.monomials.monomial_base import BasisSampler
from data_generation.monomials.monomials import Monomial, MonomialBasis, Polynomial

class SOSPolynomialSampler(PolynomialSampler):
    """
    Samples sum-of-squares polynomials by combining a monomial basis sampler
    and a PSD matrix sampler to generate polynomials of the form z^T Q z,
    where z is a vector of monomials and Q is a PSD matrix.
    
    Supports both floating-point and rational number generation.
    """
    
    def __init__(self,
                 basis_sampler: BasisSampler,
                 matrix_sampler: PSDMatrixSampler,
                 rational: bool = False):
        """
        Initialize the SOS polynomial sampler.
        
        Args:
            basis_sampler: Sampler for the monomial basis
            matrix_sampler: Sampler for PSD matrices
            rational: If True, generate rational-valued SOS polynomials
        """
        self.basis_sampler = basis_sampler
        self.matrix_sampler = matrix_sampler
        self.rational = rational
    
    def _multiply_monomials(self, m1: Monomial, m2: Monomial) -> Monomial:
        """Multiply two monomials by adding their exponents."""
        return m1 * m2
    
    def _compute_sos_terms_rational(self, 
                                  basis: MonomialBasis,
                                  Q_num: np.ndarray,
                                  Q_den: np.ndarray) -> Dict[Monomial, Fraction]:
        """
        Compute the terms of z^T Q z given basis z and rational matrix Q.
        
        Args:
            basis: List of monomials forming the basis vector z
            Q_num: Numerator matrix
            Q_den: Denominator matrix
            
        Returns:
            Dictionary mapping resulting monomials to their rational coefficients
        """
        terms = {}
        n = len(basis)
        # Compute z^T Q z with rational arithmetic
        for i in range(n):
            for j in range(n):
                # Create Fraction from numerator and denominator
                coeff = Fraction(int(Q_num[i, j]), int(Q_den[i, j]))
                if coeff != 0:  # Skip zero coefficients
                    # Multiply monomials from basis[i] and basis[j]
                    result_monomial = self._multiply_monomials(basis[i], basis[j])
                    # Add to terms (combining like terms)
                    terms[result_monomial] = terms.get(result_monomial, Fraction(0)) + coeff
        
        return terms
    
    def _compute_sos_terms_float(self, 
                               basis: MonomialBasis,
                               Q: np.ndarray) -> Dict[Monomial, float]:
        """
        Compute the terms of z^T Q z given basis z and floating-point matrix Q.
        
        Args:
            basis: List of monomials forming the basis vector z
            Q: PSD matrix
            
        Returns:
            Dictionary mapping resulting monomials to their coefficients
        """
        terms = {}
        n = len(basis)
        
        # Compute z^T Q z
        for i in range(n):
            for j in range(n):
                coeff = Q[i, j]
                if abs(coeff) > 1e-10:  # Skip near-zero coefficients
                    # Multiply monomials from basis[i] and basis[j]
                    result_monomial = self._multiply_monomials(basis[i], basis[j])
                    # Add to terms (combining like terms)
                    terms[result_monomial] = terms.get(result_monomial, 0.0) + coeff
        
        return terms
    
    def _filter_zero_diagonal_elements(self, basis: MonomialBasis, Q: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], tol: float = 1e-12) -> Tuple[MonomialBasis, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Remove basis elements corresponding to zero diagonal elements in Q.
        
        Args:
            basis: Original monomial basis
            Q: PSD matrix (either float matrix or tuple of (numerator, denominator) matrices)
            tol: Tolerance for considering diagonal elements as zero
            
        Returns:
            Tuple of (filtered_basis, filtered_Q)
        """
        if not self.rational:
            # For floating-point matrices
            diagonal = np.diag(Q)
            non_zero_indices = np.where(np.abs(diagonal) > tol)[0]
            
            if len(non_zero_indices) == len(basis):
                # No zero diagonal elements, return as is
                return basis, Q
            
            # Filter basis and matrix
            filtered_basis = [basis[i] for i in non_zero_indices]
            filtered_Q = Q[np.ix_(non_zero_indices, non_zero_indices)]
            
            return filtered_basis, filtered_Q
        else:
            # For rational matrices (tuple of numerator, denominator)
            Q_num, Q_den = Q
            diagonal_num = np.diag(Q_num)
            diagonal_den = np.diag(Q_den)
            
            # A diagonal element is zero if numerator is zero (denominator should never be zero)
            non_zero_indices = np.where(diagonal_num != 0)[0]
            
            if len(non_zero_indices) == len(basis):
                # No zero diagonal elements, return as is
                return basis, Q
            
            # Filter basis and matrices
            filtered_basis = [basis[i] for i in non_zero_indices]
            filtered_Q_num = Q_num[np.ix_(non_zero_indices, non_zero_indices)]
            filtered_Q_den = Q_den[np.ix_(non_zero_indices, non_zero_indices)]
            
            return filtered_basis, (filtered_Q_num, filtered_Q_den)

    def sample(self, num_vars: int, max_degree: int) -> Tuple[Polynomial, MonomialBasis, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Sample a SOS polynomial.
        
        Args:
            num_vars: Number of variables
            max_degree: Maximum degree (the result will have degree 2*max_degree)
            
        Returns:
            A tuple containing:
            - The sampled polynomial (either with float or rational coefficients)
            - The monomial basis used (filtered to remove elements with zero diagonal)
            - The Q matrix (either a float matrix or a tuple of (numerator, denominator) matrices, filtered)
        """
        # Sample the basis (each monomial should have degree <= max_degree)
        basis = self.basis_sampler.sample(num_vars=num_vars,
                                        max_degree=max_degree)
        
        # Sample a PSD matrix
        Q = self.matrix_sampler.sample(size=len(basis))
        
        # Filter out basis elements corresponding to zero diagonal elements
        filtered_basis, filtered_Q = self._filter_zero_diagonal_elements(basis, Q)
        
        if not self.rational:
            # Compute the resulting polynomial terms
            terms = self._compute_sos_terms_float(filtered_basis, filtered_Q)
            return Polynomial(terms, rational=False), filtered_basis, filtered_Q
        else:
            # Compute the resulting polynomial terms with rational arithmetic
            rational_terms = self._compute_sos_terms_rational(filtered_basis, filtered_Q[0], filtered_Q[1])
            # Create a rational polynomial directly from the terms
            return Polynomial(rational_terms, rational=True), filtered_basis, filtered_Q


if __name__ == "__main__":
    from data_generation.matrix.matrix_simple import SimpleRandomPSDSampler
    from data_generation.monomials.monomial_sparseuniform import SparseUniformBasisSampler
    from data_generation.monomials.monomial_clique import CliqueBasisSampler
    
    # Example with rational matrices
    basis_sampler = SparseUniformBasisSampler(
        min_sparsity=0.3,
        max_sparsity=0.5,
        min_degree=1,
        max_degree=2
    )
    
    matrix_sampler = SimpleRandomPSDSampler(
        rational=True,
        max_numerator=3,
        max_denominator=3
    )
    
    # Create the polynomial sampler with rational=True
    poly_sampler = SOSPolynomialSampler(
        basis_sampler=basis_sampler,
        matrix_sampler=matrix_sampler,
        rational=True
    )
    
    # Sample a rational polynomial
    poly, basis, Q = poly_sampler.sample(num_vars=3, max_degree=2)
    
    print("Generated rational SOS polynomial:")
    print(poly)  # Now uses the new rational polynomial string representation
    
    print(Q)

    # print as sequence of tokens
    print(poly.to_sequence(num_vars=3, include_coefficients=True, include_plus=False))
    
    # Example with floating-point matrices
    matrix_sampler_float = SimpleRandomPSDSampler(
        min_eigenval=0,
        scale=1.0,
        random_state=42
    )
    
    poly_sampler_float = SOSPolynomialSampler(
        basis_sampler=basis_sampler,
        matrix_sampler=matrix_sampler_float,
        rational=False
    )
    
    # Sample a floating-point polynomial
    poly, basis, Q = poly_sampler_float.sample(num_vars=3, max_degree=2)
    
    print("\nGenerated floating-point SOS polynomial:")
    print(poly)

    # print as sequence of tokens
    print(poly.to_sequence(num_vars=3, include_coefficients=True, include_plus=False))