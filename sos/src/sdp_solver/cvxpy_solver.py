import numpy as np
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Set
from itertools import combinations_with_replacement
from data_generation.monomials.monomials import Polynomial, Monomial, MonomialBasis
from sdp_solver.sdp_interface import SDPSolver
from utils.polynomial import get_newton_polytope_basis
from cvxpy.error import SolverError

class CVXPYSOSSolver(SDPSolver):
    """SDP solver for SOS problems using CVXPY."""
    
    def __init__(self, solver: str = 'SCS', verbose: bool = False):
        """
        Initialize the solver.
        
        Args:
            solver: CVXPY solver to use (e.g., 'MOSEK', 'SCS', 'CVXOPT')
            verbose: Whether to print solver output
        """
        self.solver = solver
        self.verbose = verbose
    
    def _generate_monomial_basis(self, poly: Polynomial) -> MonomialBasis:
        """Generate a monomial basis suitable for the SOS decomposition."""
        # Get the number of variables from any monomial
        num_vars = len(next(iter(poly.terms)).exponents)
        
        # Maximum degree of basis elements should be half the maximum degree of poly
        max_deg = max(sum(m.exponents) for m in poly.terms.keys())
        basis_max_deg = max_deg // 2
        
        # Generate all monomials up to basis_max_deg
        basis = []
        
        def generate_exponents(current: List[int], remaining_vars: int, remaining_deg: int):
            if remaining_vars == 0:
                if remaining_deg == 0:  # Only add when we've used exactly the degree
                    basis.append(Monomial(tuple(current)))
                return
            
            # Try all possible exponents for the current variable
            for d in range(remaining_deg + 1):
                generate_exponents(
                    current + [d],
                    remaining_vars - 1,
                    remaining_deg - d
                )
        
        # Generate all monomials up to basis_max_deg
        for d in range(basis_max_deg + 1):
            generate_exponents([], num_vars, d)
        
        if self.verbose:
            print(f"Generated basis of size {len(basis)}:")
            for m in basis:
                print(f"  {m}")
        
        return basis
    
    def _build_coefficient_map(self, 
                             basis: MonomialBasis,
                             Q: cp.Variable) -> Dict[Monomial, cp.Expression]:
        """Build map from monomials to their coefficients in z^T Q z."""
        coeff_map = {}
        n = len(basis)
        
        # Compute z^T Q z symbolically
        for i in range(n):
            for j in range(n):
                # Multiply basis elements
                result_monomial = basis[i] * basis[j]
                # Add Q[i,j] to coefficient of resulting monomial
                if result_monomial in coeff_map:
                    coeff_map[result_monomial] += Q[i,j]
                else:
                    coeff_map[result_monomial] = Q[i,j]
        
        if self.verbose:
            print("\nCoefficient map:")
            for m, coeff in coeff_map.items():
                print(f"  {m}: {coeff}")
        
        return coeff_map
    
    def solve_sos_feasibility(self, 
                            poly: Polynomial,
                            basis: Optional[MonomialBasis] = None,
                            solver_options: dict = None) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if a polynomial is SOS by solving a feasibility SDP.
        
        Args:
            poly: The polynomial to check
            basis: Optional monomial basis to use. If None, generates a complete basis.
                  If provided, attempts to find an SOS decomposition using only these basis elements.
            solver_options: Optional solver options
            
        Returns:
            Tuple of (is_sos, gram_matrix)
        """
        if self.verbose:
            print("\nSolving SOS feasibility for polynomial:")
            print(poly)
        
        # Use provided basis or generate complete basis
        if basis is None:
            basis = self._generate_monomial_basis(poly)
        
        if self.verbose:
            print(f"\nUsing basis of size {len(basis)}:")
            for m in basis:
                print(f"  {m}")
        
        n = len(basis)
        
        try:
            # Create PSD matrix variable with better numerical properties
            Q = cp.Variable((n, n), symmetric=True)
            
            # Add PSD constraint with small regularization for numerical stability
            epsilon = 1e-8
            constraints = [Q >> epsilon * np.eye(n)]
            
            # Build coefficient map for z^T Q z
            coeff_map = self._build_coefficient_map(basis, Q)
            
            # Scale the coefficients for better numerical conditioning
            scale = max(abs(coeff) for coeff in poly.terms.values())
            if scale > 0:
                scaled_poly = {m: c/scale for m, c in poly.terms.items()}
            else:
                scaled_poly = poly.terms
            
            # Add constraints matching coefficients
            for monomial, target_coeff in scaled_poly.items():
                if monomial in coeff_map:
                    constraints.append(coeff_map[monomial] == target_coeff)
                    if self.verbose:
                        print(f"\nAdding constraint for {monomial}: {coeff_map[monomial]} == {target_coeff}")
                else:
                    if self.verbose:
                        print(f"\nMonomial {monomial} not in basis products, cannot represent with this basis")
                    return False, None
            
            # Add constraints that all other terms must be zero
            for monomial, expr in coeff_map.items():
                if monomial not in scaled_poly:
                    constraints.append(expr == 0)
                    if self.verbose:
                        print(f"\nAdding zero constraint for {monomial}: {expr} == 0")
            
            # Add trace constraint for better numerical properties
            constraints.append(cp.trace(Q) <= 100 * n) 
            
            # Solve the feasibility SDP
            problem = cp.Problem(cp.Minimize(0), constraints)
            
            if self.verbose:
                print("\nSolving SDP...")
            
            # Handle solver options
            if solver_options is None:
                solver_options = {}
            
            # Add solver-specific default options
            if self.solver == 'MOSEK':
                pass
            elif self.solver == 'CLARABEL':
                pass
            else:
                # Add default options for SCS and other solvers
                default_options = {
                    'normalize': True,
                    'scale': 1.0,
                    'eps': 1e-8,
                    'max_iters': 10000,
                }
                # Merge with user options, preferring user values
                solver_options = {**default_options, **solver_options}
            
            result = problem.solve(solver=self.solver, verbose=self.verbose, **solver_options)
            
            is_feasible = problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
            Q_val = Q.value if is_feasible else None
            
            if Q_val is not None:
                # Scale back the solution
                Q_val *= float(scale)
                
                # Verify PSD property
                min_eig = np.min(np.linalg.eigvalsh(Q_val))
                is_feasible = is_feasible and min_eig > -1e-2
            
            if self.verbose:
                print(f"Solver status: {problem.status}")
                print(f"Optimal value: {result}")
                if Q_val is not None:
                    print("Q matrix eigenvalues:", np.linalg.eigvalsh(Q_val))
            
            return is_feasible, Q_val
            
        except (cp.error.SolverError, ValueError) as e:
            if self.verbose:
                print(f"Solver error: {e}")
            return False, None
    
    def get_sos_decomposition(self, 
                            poly: Polynomial, 
                            Q: np.ndarray,
                            basis: Optional[MonomialBasis] = None) -> str:
        """Get the SOS decomposition using eigendecomposition of Q."""
        # Use provided basis or generate complete basis
        if basis is None:
            basis = self._generate_monomial_basis(poly)
            
        if self.verbose:
            print("\nGenerating SOS decomposition using basis:")
            for m in basis:
                print(f"  {m}")
        
        # Compute eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(Q)
        
        # Keep only significant eigenvalues and vectors
        tol = 1e-10
        significant = eigvals > tol
        eigvals = eigvals[significant]
        eigvecs = eigvecs[:, significant]
        
        # Build the squares
        terms = []
        for i, (val, vec) in enumerate(zip(eigvals, eigvecs.T)):
            # Construct the term sqrt(λ)(v₁z₁ + v₂z₂ + ...)
            coeff_str = f"{np.sqrt(val):.6f}"
            term_parts = []
            for coeff, monomial in zip(vec, basis):
                if abs(coeff) > tol:
                    # Format the monomial term
                    term = self._format_monomial(monomial, coeff)
                    if term:
                        term_parts.append(term)
            
            if term_parts:
                terms.append(f"({coeff_str}*({' + '.join(term_parts)}))²")
        
        return " + ".join(terms)
    
    def _format_monomial(self, monomial: Monomial, coeff: float) -> str:
        """Format a monomial term with coefficient."""
        if abs(coeff) < 1e-10:
            return ""
            
        # Convert exponents to string representation
        var_terms = []
        for i, exp in enumerate(monomial.exponents):
            if exp == 0:
                continue
            elif exp == 1:
                var_terms.append(f"x{i+1}")
            else:
                var_terms.append(f"x{i+1}^{exp}")
        
        if not var_terms:
            return f"{coeff:.6f}"
        
        term = "*".join(var_terms)
        if abs(coeff - 1.0) < 1e-10:
            return term
        elif abs(coeff + 1.0) < 1e-10:
            return f"-{term}"
        else:
            return f"{coeff:.6f}*{term}"