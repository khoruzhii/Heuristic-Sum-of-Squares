import numpy as np
from fractions import Fraction
from typing import Dict, Tuple, List, Union, Optional
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class Monomial:
    """Represents a monomial by its exponent tuple, e.g., x_1^2 * x_2 -> (2, 1)."""
    exponents: Tuple[int, ...]

    @property
    def degree(self) -> int:
        return sum(self.exponents)

    def to_sequence(self, 
                   num_vars: int, 
                   rational: bool = False,
                   include_coefficient: bool = True) -> List[str]:
        """
        Convert the monomial to a sequence of tokens using the format:
        - Coefficient (always 1): C1_1 (rational) or C1.0 (float)
        - Exponents: Individual E{value} tokens (e.g., E0 E1 E2)
        
        Args:
            num_vars: Number of variables (used for padding exponents)
            rational: If True, use rational coefficient format (C1_1), else float (C1.0)
            include_coefficient: Whether to include the dummy coefficient token
            
        Returns:
            List of tokens in the format ["C1_1", "E0", "E1", "E2"] or ["C1.0", "E0", "E1", "E2"]
        """
        tokens = []
        
        # Add dummy coefficient if requested
        if include_coefficient:
            tokens.append("C1_1" if rational else "C1.0")
        
        # Handle exponents as individual tokens
        exponents = list(self.exponents)
        if len(exponents) < num_vars:
            exponents.extend([0] * (num_vars - len(exponents)))
        elif len(exponents) > num_vars:
            exponents = exponents[:num_vars]
        
        # Add each exponent as a separate token
        for exp in exponents:
            tokens.append(f"E{exp}")
        
        return tokens

    @staticmethod
    def from_sequence(tokens: List[str]) -> Tuple['Monomial', Union[float, Fraction]]:
        """
        Create a Monomial from a sequence of tokens.
        
        Args:
            tokens: List of tokens in format ["C1_1", "E0", "E1", "E2"] or ["C1.0", "E0", "E1", "E2"]
            
        Returns:
            Tuple of (Monomial, coefficient) where coefficient is Fraction for C1_1 format
            or float for C1.0 format
        """
        # Extract coefficient if present
        coeff = None
        start_idx = 0
        if tokens[0].startswith('C'):
            if '_' in tokens[0]:  # Rational format C1_1
                num, den = map(int, tokens[0][1:].split('_'))
                coeff = Fraction(num, den)
            else:  # Float format C1.0
                coeff = float(tokens[0][1:])
            start_idx = 1
            
        # Parse exponents
        exponents = []
        for token in tokens[start_idx:]:
            if token.startswith('E'):
                exponents.append(int(token[1:]))
                
        return Monomial(tuple(exponents)), coeff if coeff is not None else Fraction(1, 1)

    def __mul__(self, other: 'Monomial') -> 'Monomial':
        """Multiply two monomials by adding their exponents."""
        # Get the maximum number of variables
        max_vars = max(len(self.exponents), len(other.exponents))
        
        # Pad both exponents to the same length
        self_exp = list(self.exponents) + [0] * (max_vars - len(self.exponents))
        other_exp = list(other.exponents) + [0] * (max_vars - len(other.exponents))
        
        # Add exponents
        new_exponents = tuple(a + b for a, b in zip(self_exp, other_exp))
        return Monomial(exponents=new_exponents)

    def __repr__(self) -> str:
        """String representation of the monomial."""
        terms = []
        for i, exp in enumerate(self.exponents):
            if exp == 0:
                continue
            elif exp == 1:
                terms.append(f"x{i+1}")
            else:
                terms.append(f"x{i+1}^{exp}")
        return "*".join(terms) if terms else "1"

class Polynomial:
    """
    Represents a sparse polynomial as a dictionary from Monomial to coefficient.
    Supports both floating-point and rational coefficients.
    """
    def __init__(self, terms: Dict[Monomial, Union[float, Fraction]], rational: bool = False):
        """
        Initialize a polynomial with either floating-point or rational coefficients.
        
        Args:
            terms: Dictionary mapping monomials to their coefficients
            rational: If True, store coefficients as Fractions, otherwise as floats
        """
        self.rational = rational
        if rational:
            # Convert all coefficients to Fractions if they aren't already
            self.terms = {
                m: (c if isinstance(c, Fraction) else Fraction(c).limit_denominator())
                for m, c in terms.items()
                if c != 0 and c != Fraction(0)
            }
        else:
            # Store as floats, filtering out near-zero terms
            self.terms = {
                m: float(c) for m, c in terms.items()
                if not np.isclose(float(c), 0)
            }

    @classmethod
    def from_rational_parts(cls, num_terms: Dict[Monomial, int], den_terms: Dict[Monomial, int]) -> 'Polynomial':
        """
        Create a rational polynomial from separate numerator and denominator dictionaries.
        
        Args:
            num_terms: Dictionary mapping monomials to numerators
            den_terms: Dictionary mapping monomials to denominators
            
        Returns:
            A Polynomial with rational coefficients
        """
        terms = {
            monomial: Fraction(num, den_terms[monomial])
            for monomial, num in num_terms.items()
            if monomial in den_terms and den_terms[monomial] != 0
        }
        return cls(terms, rational=True)

    def to_sequence(self, 
                   num_vars: int, 
                   include_coefficients: bool = True, 
                   include_plus: bool = True,
                   round_coeffs: bool = False,
                   digits: int = 1, 
                   sort_polynomials: bool = False) -> List[str]:
        """
        Convert the polynomial to a sequence of tokens using new format:
        - Rational coefficients: C{numerator}_{denominator} (e.g., C1_3)
        - Real coefficients: C{value:.1f} (e.g., C1.5)
        - Exponents: Individual E{value} tokens (e.g., E0 E1 E2)
        
        Args:
            num_vars: Number of variables (used for padding exponents)
            include_coefficients: Whether to include coefficient tokens
            include_plus: Whether to include plus tokens between terms
            round_coeffs: If True, round rational coefficients to floating point
            digits: Number of decimal places for floating point coefficients (default 1)
            
        Returns:
            List of tokens in the format ["C1_3", "E0", "E1", "E2", "+", ...] or
            ["C1.5", "E0", "E1", "E2", "+", ...]
        """
        if not self.terms:
            # Return tokens for zero polynomial (all exponents 0)
            return [f"E0"] * num_vars if not include_coefficients else ["C0.0"] + [f"E0"] * num_vars
        
        tokens = []
        terms = list(self.terms.items())

        # Sort terms lexicographically
        if sort_polynomials:
            terms.sort(key=lambda x: x[0].exponents)
        
        for i, (monomial, coeff) in enumerate(terms):
            # Handle coefficient
            if include_coefficients:
                if self.rational and not round_coeffs:
                    # Use fraction format: C{numerator}_{denominator}
                    tokens.append(f"C{coeff.numerator}_{coeff.denominator}")
                else:
                    # Format as float with specified digits
                    if self.rational:
                        coeff_val = float(coeff.numerator) / float(coeff.denominator)
                    else:
                        coeff_val = float(coeff)
                    tokens.append(f"C{coeff_val:.{digits}f}")
            
            # Handle monomial exponents as individual tokens
            exponents = list(monomial.exponents)
            if len(exponents) < num_vars:
                exponents.extend([0] * (num_vars - len(exponents)))
            elif len(exponents) > num_vars:
                exponents = exponents[:num_vars]
            
            # Add each exponent as a separate token
            for exp in exponents:
                tokens.append(f"E{exp}")
            
            # Add plus token between terms (but not after the last term)
            if include_plus and i < len(terms) - 1:
                tokens.append("+")

        
        return tokens

    @staticmethod
    def from_sequence(tokens: List[str]) -> 'Polynomial':
        """
        Create a Polynomial from a sequence of tokens.
        
        Args:
            tokens: List of tokens in format ["C1_1", "E0", "E1", "E2", "+", "C2_1", "E1", "E0", "E0", ...]
            or ["C1.5", "E0", "E1", "E2", "+", "C2.0", "E1", "E0", "E0", ...]
            
        Returns:
            Polynomial object with terms constructed from the sequence
        """
        terms = {}
        current_tokens = []
        rational = False
        
        # Determine if we're dealing with rational coefficients
        for token in tokens:
            if token.startswith('C'):
                rational = '_' in token
                break
        
        # Process tokens
        for token in tokens:
            if token == "+":
                if current_tokens:
                    monomial, coeff = Monomial.from_sequence(current_tokens)
                    terms[monomial] = coeff
                    current_tokens = []
            else:
                current_tokens.append(token)
                
        # Handle last term
        if current_tokens:
            monomial, coeff = Monomial.from_sequence(current_tokens)
            terms[monomial] = coeff
            
        return Polynomial(terms, rational=rational)

    def __repr__(self) -> str:
        """String representation of the polynomial."""
        if not self.terms:
            return "0"
        
        if self.rational:
            # For rational coefficients, show as fractions
            terms = []
            for m, c in self.terms.items():
                if c.denominator == 1:
                    coeff_str = str(c.numerator)
                else:
                    coeff_str = f"({c.numerator}/{c.denominator})"
                terms.append(f"{coeff_str}*{m}" if m.__repr__() != "1" else coeff_str)
        else:
            # For float coefficients, use standard format
            terms = [f"{c:.4f}*{m}" if m.__repr__() != "1" else f"{c:.4f}" 
                    for m, c in self.terms.items()]
        
        return " + ".join(terms)

    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        """Add two polynomials by combining their terms."""
        if self.rational != other.rational:
            raise ValueError("Cannot add rational and floating-point polynomials directly")
        
        result_terms = dict(self.terms)
        for monomial, coeff in other.terms.items():
            if monomial in result_terms:
                result_terms[monomial] += coeff
            else:
                result_terms[monomial] = coeff
        
        return Polynomial(result_terms, rational=self.rational)

    def __eq__(self, other: 'Polynomial') -> bool:
        """Check if two polynomials are equal by comparing their terms dictionaries."""
        if not isinstance(other, Polynomial):
            return False
        
        # Check if both have the same rational flag
        if self.rational != other.rational:
            return False
        
        # Check if terms dictionaries have the same keys and values
        if len(self.terms) != len(other.terms):
            return False
        
        for monomial, coeff in self.terms.items():

            if monomial not in other.terms.keys():
                return False
            if coeff != other.terms[monomial]:
                return False
        
        return True

    def to_float(self) -> 'Polynomial':
        """Convert a rational polynomial to a floating-point polynomial."""
        if not self.rational:
            return self
        
        float_terms = {m: float(c) for m, c in self.terms.items()}
        return Polynomial(float_terms, rational=False)

    def to_rational(self, max_denominator: Optional[int] = None) -> 'Polynomial':
        """
        Convert a floating-point polynomial to a rational polynomial.
        
        Args:
            max_denominator: Optional maximum denominator for the fractions
            
        Returns:
            A new Polynomial with rational coefficients
        """
        if self.rational:
            return self
        
        rational_terms = {}
        for m, c in self.terms.items():
            frac = Fraction(c).limit_denominator(max_denominator) if max_denominator else Fraction(c)
            rational_terms[m] = frac
        
        return Polynomial(rational_terms, rational=True)

    def min_sos_basis_size(self) -> int:
        """
        Calculate the minimum number of basis elements required for any SOS decomposition
        of this polynomial, based on the number of terms (support) in the polynomial.
        
        Returns:
            The minimum number of basis elements required, given by (sqrt(1 + 8*support) - 1)/2
            where support is the number of terms in the polynomial.
        """
        support = len(self.terms)
        return int((np.sqrt(1 + 8*support) - 1) / 2)


MonomialBasis = List[Monomial]


if __name__ == "__main__":
    # Example: create monomials for x^2, x*y, y^2 in two variables (x, y)
    m_x2 = Monomial((2, 0))   # x^2
    m_xy = Monomial((1, 1))   # x*y
    m_y2 = Monomial((0, 2))   # y^2

    # Show monomial sequences
    print("Monomial sequences:")
    print("x^2 (float):", m_x2.to_sequence(2, rational=False))
    print("x^2 (rational):", m_x2.to_sequence(2, rational=True))
    print("x*y (float):", m_xy.to_sequence(2, rational=False))
    print("y^2 (rational):", m_y2.to_sequence(2, rational=True))
    print()

    # Create a floating-point polynomial: 3.0*x^2 + 2.0*x*y - 1.5*y^2
    float_terms = {
        m_x2: 3.0,
        m_xy: 2.0,
        m_y2: -1.5
    }
    float_poly = Polynomial(float_terms)

    # Create a rational polynomial: (1/2)*x^2 + (2/3)*x*y - (3/4)*y^2
    rational_terms = {
        m_x2: Fraction(1, 2),
        m_xy: Fraction(2, 3),
        m_y2: Fraction(-3, 4)
    }
    rational_poly = Polynomial(rational_terms, rational=True)

    # Create from separate numerator and denominator terms
    num_terms = {m_x2: 1, m_xy: 2, m_y2: -3}
    den_terms = {m_x2: 2, m_xy: 3, m_y2: 4}
    rational_poly2 = Polynomial.from_rational_parts(num_terms, den_terms)

    print("Floating-point polynomial:")
    print(float_poly)
    print("Sequence:", float_poly.to_sequence(2))
    print("\nRational polynomial:")
    print(rational_poly)
    print("Sequence:", rational_poly.to_sequence(2))
    print("\nRational polynomial (from parts):")
    print(rational_poly2)
    print("Sequence:", rational_poly2.to_sequence(2))
    print("\nConverted to float:")
    print(rational_poly.to_float())
    print("\nFloat converted to rational:")
    print(float_poly.to_rational(max_denominator=100))

