"""
Data generation module for SOS Transformer.

This module provides functionality for generating training data for the SOS Transformer,
including monomial sampling, matrix generation, and polynomial SOS decomposition.
"""

from .monomials import (
    BasisSampler,
    CliqueBasisSampler,
    PolytopeBasisSampler,
    SparseUniformBasisSampler,
)

from .matrix import (
    PSDMatrixSampler,
    SimpleRandomPSDSampler
)

from .polynomials import (
    PolynomialSampler,
    SOSPolynomialSampler
)

__all__ = [
    # Monomials
    'BasisSampler',
    'CliqueBasisSampler',
    'PolytopeBasisSampler',
    'SparseUniformBasisSampler',
    
    # Matrix
    'PSDMatrixSampler',
    'SimpleRandomPSDSampler',
    
    # Polynomials
    'PolynomialSampler',
    'SOSPolynomialSampler'
]
