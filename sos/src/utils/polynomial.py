"""
Utilities for polynomial manipulation and analysis.
"""

import numpy as np
import random
import re
from scipy.spatial import ConvexHull, Delaunay
from typing import List, Union, Optional, Tuple, Dict, Any
from utils.parse_polynomial import get_high_dim_example
from data_generation.monomials.monomials import Monomial, Polynomial
import itertools
from scipy.optimize import linprog
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, PULP_CBC_CMD
import json


try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _lazy_enum(mins: np.ndarray,
               maxs: np.ndarray,
               A: np.ndarray,
               b: np.ndarray,
            eps: float = 1e-6) -> np.ndarray:
    """Yield each lattice point satisfying Ax + b ≤ 0."""
    row_eps = eps * np.linalg.norm(A, axis=1)        # ‖aᵢ‖–scaled slack
    for x in itertools.product(*(range(lo, hi + 1) for lo, hi in zip(mins, maxs))):
        v = np.asarray(x, dtype=float)
        if np.all(A @ v + b <= row_eps):     # Slightly more lenient tolerance
            yield x


if _HAS_NUMBA:
    # 30-100 µs for the inequality test on medium arrays
    @njit(cache=True, fastmath=True)
    def _filter_points_numba(candidates: np.ndarray,
                             A: np.ndarray,
                             b: np.ndarray,
                            eps: float = 1e-6) -> np.ndarray:
        keep = []
        row_eps = eps * np.linalg.norm(A, axis=1)        # ‖aᵢ‖–scaled slack
        for i in range(candidates.shape[0]):
            v = candidates[i]
            if np.all(A @ v + b <= row_eps):  # Match tolerance with _lazy_enum
                keep.append(i)
        return np.array(keep, dtype=np.int64)


def get_newton_polytope_basis(
        exponent_vectors: Union[List[Tuple[int, ...]], np.ndarray],
        return_num_vertices: bool = False
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], int]]:
    """
    Integer points in N(p)/2, where N(p) is the Newton polytope.
    Only points that are exactly integers after division by 2 are included.
    
    For example, if N(p) contains (3,1), the point (3/2, 1/2) is not included
    since it's not an integer point.

    For lower dimensional spaces (e.g. points lying in a plane or line),
    we compute the Newton polytope in that subspace.

    Parameters
    ----------
    exponent_vectors
        Sequence of integer exponent tuples.
    return_num_vertices
        If ``True`` also return the number of hull vertices.

    Returns
    -------
    ndarray | None
        (m, d) array of lattice points ordered as discovered,
        or ``None`` if the polytope contains no integer points.
    (…, int)
        When *return_num_vertices* is *True* the tuple adds the
        vertex count of the convex hull.
    """
    pts = np.asarray(exponent_vectors, dtype=float)
    if pts.size == 0:
        return (None, 0) if return_num_vertices else None

    try:
        # First compute convex hull of original points
        hull = ConvexHull(pts, qhull_options="QJ")
    except Exception as e:
        # Degenerate: all points identical or lower-dimensional
        if len(pts) == 1 or np.allclose(pts, pts[0]):
            # For a single point p, check if p/2 is integer
            halved = pts[0] / 2
            if np.allclose(halved, np.round(halved)):
                result = np.round(halved).astype(int)[None, ...]
                return (result, 1) if return_num_vertices else result
            return (None, 1) if return_num_vertices else None
        
        # For collinear points or points in a lower dimensional space
        if len(pts) >= 2:
            # Check if points lie in a lower dimensional space by computing rank
            centered = pts - pts.mean(axis=0)
            rank = np.linalg.matrix_rank(centered, tol=1e-10)
            
            if rank < pts.shape[1]:  # Points lie in a lower dimensional space
                # Project points onto their principal directions
                u, s, vh = np.linalg.svd(centered)
                # Use only the significant directions (where s > tol)
                significant_dirs = vh[:rank]
                # Project all points onto these directions
                mean = pts.mean(axis=0)
                projs = (pts - mean) @ significant_dirs.T      # ① subtract translation
                
                # For rank 1 (line), generate points along the line
                if rank == 1:
                    # Find the integer points directly from the input points
                    # Since we know the input points are integers
                    start_point = pts[np.argmin(projs)]
                    end_point = pts[np.argmax(projs)]
                    direction = end_point - start_point
                    # Get the GCD of the direction vector to find the smallest integer step
                    gcd = np.gcd.reduce(np.abs(direction).astype(int))
                    if gcd > 0:
                        unit_step = direction / gcd
                        # Generate all points along the line with integer steps
                        steps = np.arange(gcd + 1)[:, None]  # +1 to include endpoint
                        line_points = start_point + steps * unit_step
                        # Now divide these points by 2 and keep only those that are integers
                        valid = []
                        for p in line_points:
                            halved = p / 2
                            if np.allclose(halved, np.round(halved)):
                                valid.append(np.round(halved).astype(int))
                        if valid:
                            valid = np.unique(np.array(valid), axis=0)
                            return (valid, len(pts)) if return_num_vertices else valid
                
                # For rank > 1, try to compute hull in the projected space
                try:
                    hull = ConvexHull(projs, qhull_options="QJ")
                    # Get half-space representation in projected space
                    A_proj, b_proj = hull.equations[:, :-1], hull.equations[:, -1]
                    # Map back to original space
                    A = A_proj @ significant_dirs                  # ② map normals back
                    b = b_proj - (A @ mean)
                except:
                    # If hull computation fails, return None
                    return (None, len(pts)) if return_num_vertices else None
            else:
                # Try computing hull with different options
                try:
                    hull = ConvexHull(pts, qhull_options="QJ Pp")
                    A, b = hull.equations[:, :-1], hull.equations[:, -1]
                except:
                    return (None, len(pts)) if return_num_vertices else None

    else:
        # Normal case - hull computation succeeded
        A, b = hull.equations[:, :-1], hull.equations[:, -1]

    # Scale the inequalities to represent N(p)/2
    A_half = A.copy()
    b_half = b/2

    # Compute bounds for integer points in N(p)/2
    # We only need to check points that could be integers after division by 2
    margin = 1e-7
    mins = np.ceil(np.min(pts, axis=0)/2 - margin).astype(int)
    maxs = np.floor(np.max(pts, axis=0)/2 + margin).astype(int)

    # --------- lattice enumeration ----------
    if _HAS_NUMBA and (maxs - mins).prod() > 10_000:
        # Big box – generate grid as int32, filter in Numba
        grids = [np.arange(lo, hi + 1, dtype=np.int32)
                 for lo, hi in zip(mins, maxs)]
        mesh = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, len(mins))
        keep = _filter_points_numba(mesh.astype(np.float64), A_half, b_half)
        valid = mesh[keep].astype(int)
    else:
        # Convert generator output to array of points
        points = list(_lazy_enum(mins, maxs, A_half, b_half))
        valid = np.array(points, dtype=int) if points else np.empty((0, pts.shape[1]), dtype=int)

    if len(valid) == 0:
        return (None, len(hull.vertices)) if return_num_vertices else None

    # Remove duplicates and ensure points are sorted
    valid = np.unique(valid, axis=0)
    
    return (valid, len(hull.vertices)) if return_num_vertices else valid



def parse_monomial_str(
    monomial_str: str, 
    no_coeff: bool = False, 
    num_variables: int = 5
) -> Tuple[Monomial, float]:
    """Parse a monomial string of the form 'coeff*x1^a*x2^b...' into a Monomial object and coefficient.

    Args:
        monomial_str (str): The monomial string to parse.
        no_coeff (bool): If True, the string is just the monomial part (no coefficient).
        num_variables (int): Number of variables (e.g., 5 for x1...x5).

    Returns:
        Tuple[Monomial, float]: The Monomial object and its coefficient.
    """
    # Handle signs
    monomial_str = monomial_str.strip()
    if monomial_str.startswith('+'):
        monomial_str = monomial_str[1:].strip()
    elif not monomial_str:
        return Monomial(tuple([0] * num_variables)), 1.0  # Return constant 1 for empty string
        
    # Handle constant monomial "1"
    if monomial_str == "1":
        return Monomial(tuple([0] * num_variables)), 1.0
    
    if no_coeff:
        # The string is just the monomial part, e.g. "x1^2*x2"
        coeff = 1.0
        var_part_str = monomial_str
    else:
        # Split coefficient and monomial parts
        parts = monomial_str.split('*', 1)
        try:
            coeff = float(parts[0])
        except ValueError:
            if parts[0] == "1":  # Handle case where the string starts with "1"
                coeff = 1.0
                var_part_str = parts[1] if len(parts) > 1 else ""
            else:
                return Monomial(tuple([0] * num_variables)), 0.0

        if len(parts) == 1:  # Constant term
            return Monomial(tuple([0] * num_variables)), coeff
            
        # Handle case where the variable part is just "1"
        if parts[1].strip() == "1":
            return Monomial(tuple([0] * num_variables)), coeff
            
        var_part_str = parts[1]
    
    # Parse variables and exponents
    var_parts = var_part_str.split('*')
    exponents = [0] * num_variables  # Now adjustable number of variables
    for var_part in var_parts:
        if not var_part or var_part == "1":  # Skip empty parts and constant terms
            continue
        if '^' in var_part:
            var, exp = var_part.split('^')
            if var.startswith('x'):
                idx = int(var[1:]) - 1  # Support x10, x11, etc.
                if 0 <= idx < num_variables:
                    exponents[idx] = int(exp)
        else:
            if var_part.startswith('x'):
                idx = int(var_part[1:]) - 1
                if 0 <= idx < num_variables:
                    exponents[idx] = 1
    
    return Monomial(tuple(exponents)), coeff


def parse_polynomial_str(poly_str: str) -> Polynomial:
    """Parse a polynomial string into a Polynomial object."""
    terms = {}
    # Split on + or - while keeping the signs
    parts = re.split(r'(?=[-+])', poly_str.strip())
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        monomial, coeff = parse_monomial_str(part)
        if coeff != 0:  # Only add non-zero terms
            terms[monomial] = coeff
        
    return Polynomial(terms)


def verify_basis_reconstruction(poly: Polynomial, basis: List[Monomial], Q: np.ndarray) -> bool:
    """Verify that basis^T * Q * basis approximately gives back the original polynomial."""
    n = len(basis)
    reconstructed_terms = {}
    
    # Compute basis^T * Q * basis symbolically
    for i in range(n):
        for j in range(n):
            result_monomial = basis[i] * basis[j]
            coeff = Q[i,j]
            
            if result_monomial in reconstructed_terms:
                reconstructed_terms[result_monomial] += coeff
            else:
                reconstructed_terms[result_monomial] = coeff
                
    # Use a more lenient relative tolerance for coefficient comparison
    rtol = 1e-2  # 1% relative tolerance
    
    # Compare coefficients
    for monomial, coeff in poly.terms.items():
        if monomial not in reconstructed_terms or not np.isclose(coeff, reconstructed_terms[monomial], rtol=rtol):
            return False
            
    # Check for significant extra terms
    for monomial, coeff in reconstructed_terms.items():
        if abs(coeff) > rtol * max(abs(c) for c in poly.terms.values()) and (monomial not in poly.terms):
            return False
            
    return True


def get_newton_polytope_basis_mip(exponent_vectors: Union[List[Tuple[int, ...]], np.ndarray], 
                           return_num_vertices: bool = False) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], int]]:
    """
    Calculates all integer points in N(p)/2, where N(p) is the Newton polytope of polynomial p.
    These points correspond to the possible monomials that can appear in an SOS decomposition.
    
    For a polynomial p(x) = Σ cₐxᵃ, if it has an SOS decomposition p(x) = Σfᵢ(x)², then the
    exponents of the monomials in fᵢ(x) must lie in N(p)/2.
    
    Args:
        exponent_vectors: List of exponent vectors from the original polynomial p(x)
        return_num_vertices: If True, also return the number of vertices in the original polytope
        
    Returns:
        Array of exponent vectors that could appear in the SOS decomposition,
        or None if no valid points found.
        If return_num_vertices is True, also returns number of vertices in original polytope.
    """
    # Convert input to numpy array if needed
    points = np.array(exponent_vectors, dtype=float)
    if len(points) == 0:
        return None if not return_num_vertices else (None, 0)
    
    # Scale the polytope by 1/2 to get the potential exponents for the SOS decomposition
    points_halved = points / 2.0
    
    # Get bounding box for the scaled polytope
    min_coords = np.floor(np.min(points_halved, axis=0)).astype(int)
    max_coords = np.ceil(np.max(points_halved, axis=0)).astype(int)
    
    # Generate all integer points within the bounding box
    ranges = [range(min_coord, max_coord + 1) for min_coord, max_coord in zip(min_coords, max_coords)]
    all_integer_points = np.array(list(itertools.product(*ranges)))
    
    if len(all_integer_points) == 0:
        return None if not return_num_vertices else (None, 0)
    
    # Function to check if a point lies in the convex hull
    def point_in_polytope(point):
        # Special case for zero point - check if it's exactly half of any original point
        if np.all(point == 0):
            # Check if any original point has all even coordinates
            for p in points:
                if np.all(p % 2 == 0):
                    return True
            return False
            
        # Create a new problem for each point to avoid constraint conflicts
        prob = LpProblem("PointInPolytope", LpMinimize)
        
        # Variables for convex combination coefficients
        lambdas = [LpVariable(f'lambda_{i}', 0, 1) for i in range(len(points_halved))]
        
        # Convex combination constraint
        prob += lpSum(lambdas) == 1, "sum_to_one"
        
        # Add constraints for each dimension with tolerance
        tol = 1e-10
        for j, coord in enumerate(point):
            expr = lpSum(points_halved[i,j] * lambdas[i] for i in range(len(points_halved)))
            prob += expr <= coord + tol, f"coord_{j}_upper"
            prob += expr >= coord - tol, f"coord_{j}_lower"
            
        # Solve
        status = prob.solve(PULP_CBC_CMD(msg=False))
        return status == 1
    
    # Filter points that lie in the convex hull
    valid_points = []
    for point in all_integer_points:
        if point_in_polytope(point):
            valid_points.append(point)
    
    if not valid_points:
        return None if not return_num_vertices else (None, 0)
    
    result = np.array(valid_points)
    
    if return_num_vertices:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)  # Use original points for vertex count
        return result, len(hull.vertices)
    
    return result


def check_diagonal_inconsistency(candidate_monomials: np.ndarray, polynomial_terms: np.ndarray) -> np.ndarray:
    """
    Checks for diagonal inconsistency in candidate monomials for SOS decomposition.
    
    A candidate monomial x^s_i is redundant if its square (2s_i) cannot be obtained as:
    1. A term in the original polynomial (p_k)
    2. A sum of two other candidate monomials (s_j + s_l)
    
    Args:
        candidate_monomials: Array where each row is an exponent vector of a candidate monomial
        polynomial_terms: Array where each row is an exponent vector of a term in the original polynomial
        
    Returns:
        np.ndarray: Boolean mask where True indicates the monomial should be kept
    """
    n_candidates = len(candidate_monomials)
    keep_monomial = np.ones(n_candidates, dtype=bool)
    
    # Keep iterating until no more reductions are possible
    while True:
        reduced = False
        for i in range(n_candidates):
            if not keep_monomial[i]:
                continue
                
            # Get the squared exponents of the candidate monomial
            squared = 2 * candidate_monomials[i]
            
            # Check if the squared term appears in the original polynomial
            in_polynomial = any(np.array_equal(squared, term) for term in polynomial_terms)
            if in_polynomial:
                continue
            
            # Check if the squared term can be obtained as a sum of two other candidates
            can_be_sum = False
            for j in range(n_candidates):
                if not keep_monomial[j] or j == i:
                    continue
                for l in range(j + 1, n_candidates):
                    if not keep_monomial[l] or l == i:
                        continue
                    if np.array_equal(squared, candidate_monomials[j] + candidate_monomials[l]):
                        can_be_sum = True
                        break
                if can_be_sum:
                    break
            
            # If neither condition is met, mark the monomial as redundant
            if not can_be_sum:
                keep_monomial[i] = False
                reduced = True
        
        # If no reductions were made in this iteration, we're done
        if not reduced:
            break
    
    return keep_monomial

def get_newton_polytope_basis_with_diagonal_check(
    exponent_vectors: Union[List[Tuple[int, ...]], np.ndarray],
    return_num_vertices: bool = False
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], int]]:
    """
    Calculates the monomial basis for an SOS decomposition using both
    Newton polytope method followed by diagonal inconsistency check.
    
    Args:
        exponent_vectors: A list of the exponent vectors of the terms in the polynomial
        return_num_vertices: If True, also return the number of vertices
        
    Returns:
        If return_num_vertices is False:
            Array of basis exponent vectors or None if computation fails
        If return_num_vertices is True:
            Tuple of (basis_array, num_vertices) or (None, 0) if computation fails
    """
    exponents = np.array(exponent_vectors, dtype=int)
    
    # First apply Newton polytope
    result = get_newton_polytope_basis(exponents, return_num_vertices=True)
    if result[0] is None:
        return (None, 0) if return_num_vertices else None
        
    # Then apply diagonal check
    keep_mask = check_diagonal_inconsistency(result[0], exponents)
    if return_num_vertices:
        return result[0][keep_mask], result[1]
    else:
        return result[0][keep_mask]


def get_M_token_variance_ordering(tokens: List[str]) -> Dict[str, Any]:
    """
    Compute the variance-based permutation for a list of tokenized monomials in M_i_j_k_l format.
    
    Args:
        tokens (List[str]): List of strings, each representing a monomial in M_i_j_k_l format
                      Example: ['M_0_1_2_0', 'M_1_0_0_1']
                      
    Returns:
        dict: Dictionary containing:
            - 'permutation': Array of indices ordering variables by variance (highest to lowest)
            - 'variances': Array of variances for each variable
            - 'n_variables': Number of variables
            - 'exponents': The computed exponent vectors
    """
    # Convert tokens to exponent vectors
    exponents = []
    for token in tokens:
        if not token.startswith('M_'):
            raise ValueError(f"Invalid token format: {token}. Expected format: M_i_j_k_l")
        # Skip the 'M_' prefix and split into exponents
        parts = token.split('_')
        if len(parts) <= 1:
            raise ValueError(f"Invalid token format: {token}. Expected format: M_i_j_k_l")
        exp_values = parts[1:]  # Skip the 'M' part
        try:
            exponents.append([int(v) for v in exp_values])
        except ValueError:
            raise ValueError(f"Invalid exponents in token: {token}. All values after M_ must be integers.")
    
    if not exponents:
        raise ValueError("No valid tokens provided")
        
    exponents = np.array(exponents)
    n_vars = exponents.shape[1]
    
    # Compute variance ordering
    variances = np.var(exponents, axis=0)
    variance_ordering = np.argsort(-variances)  # Sort in descending order
    
    return {
        'permutation': variance_ordering,
        'variances': variances,
        'n_variables': n_vars,
        'exponents': exponents
    }

def permute_monomial_token(token: str, permutation: List[int]) -> str:
    """
    Permute a monomial token in M_i_j_k_l format according to a given permutation.
    
    Args:
        token (str): Monomial token in M_i_j_k_l format (e.g., 'M_0_1_2_0')
        permutation (List[int]): List of indices indicating the new positions
                                e.g., [2,0,3,1] means var_2 goes to position 0,
                                var_0 goes to position 1, etc.
    
    Returns:
        str: Permuted monomial token
    """
    # Split token into parts
    parts = token.split('_')
    if len(parts) <= 1 or parts[0] != 'M':
        return token
    
    # Get exponents and verify length matches permutation
    exponents = parts[1:]
    if len(exponents) != len(permutation):
        raise ValueError(f"Permutation length {len(permutation)} does not match number of variables {len(exponents)}")
    
    # Create new exponent list according to permutation
    new_exponents = ['0'] * len(permutation)
    for old_pos, new_pos in enumerate(permutation):
        new_exponents[new_pos] = exponents[old_pos]
    
    # Reconstruct token
    return 'M_' + '_'.join(new_exponents)

def permute_monomial_tokens(tokens: List[str], permutation: List[int]) -> List[str]:
    """
    Permute a list of monomial tokens according to a given permutation.
    
    Args:
        tokens (List[str]): List of monomial tokens in M_i_j_k_l format
        permutation (List[int]): List of indices indicating the new positions
                                e.g., [2,0,3,1] means var_2 goes to position 0,
                                var_0 goes to position 1, etc.
    
    Returns:
        List[str]: List of permuted monomial tokens
    """
    return [permute_monomial_token(token, permutation) for token in tokens]

def permute_exponent_vector(exponent: List[int], permutation: List[int]) -> List[int]:
    """
    Permute an exponent vector according to a given permutation.
    
    Args:
        exponent (List[int]): List of exponents [i,j,k,l]
        permutation (List[int]): List of indices indicating the new positions
                                e.g., [2,0,3,1] means var_2 goes to position 0,
                                var_0 goes to position 1, etc.
    
    Returns:
        List[int]: Permuted exponent vector
    """
    if len(exponent) != len(permutation):
        raise ValueError(f"Permutation length {len(permutation)} does not match exponent length {len(exponent)}")
    
    result = [0] * len(permutation)
    for old_pos, new_pos in enumerate(permutation):
        result[new_pos] = exponent[old_pos]
    return result

def permute_exponent_vectors(exponents: Union[List[List[int]], np.ndarray], permutation: List[int]) -> np.ndarray:
    """
    Permute a list/array of exponent vectors according to a given permutation.
    
    Args:
        exponents (Union[List[List[int]], np.ndarray]): List or array of exponent vectors
        permutation (List[int]): List of indices indicating the new positions
                                e.g., [2,0,3,1] means var_2 goes to position 0,
                                var_0 goes to position 1, etc.
    
    Returns:
        np.ndarray: Array of permuted exponent vectors
    """
    exponents = np.array(exponents)
    if exponents.shape[1] != len(permutation):
        raise ValueError(f"Permutation length {len(permutation)} does not match number of variables {exponents.shape[1]}")
    
    # Use numpy's advanced indexing to permute columns
    return exponents[:, permutation]

def get_monomial_exponents(token: str) -> List[int]:
    """Extract exponents from a monomial token in M_i_j_k_l format."""
    if not token.startswith("M_"):
        raise ValueError(f"Invalid monomial token format: {token}")
    return [int(x) for x in token[2:].split("_")]

def load_polynomials_and_bases_from_jsonl(data_path: str, num_examples: int = 1000):
    """
    Load a jsonl file, extract 'polynomial_tokens' and 'basis_tokens',
    and construct Polynomial and basis (list of Monomials) objects.

    Args:
        data_path (str): Path to the jsonl file.
        num_examples (int): Number of examples to load.
    Returns:
        List of tuples: (Polynomial, List[Monomial])
    """
    results = []
    count = 0
    with open(data_path, 'r') as f:
        for line in f:
            if count >= num_examples:
                break
            entry = json.loads(line)
            poly_tokens = entry['polynomial_tokens']
            basis_tokens = entry['basis_tokens']
            poly = Polynomial.from_sequence(poly_tokens)
            # basis_tokens is a list of lists of tokens for each monomial
            basis = Polynomial.from_sequence(basis_tokens)
            # convert basis to list of Monomials
            basis = list(basis.terms.keys())
            results.append((poly, basis))
            count += 1
    return results

def remove_k_random_elements(basis: List, k: int, seed: int = None) -> Tuple[List, List]:
    """
    Remove k random elements from the basis.

    Args:
        basis (List): The original basis list.
        k (int): Number of elements to remove.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        Tuple[List, List]: (reduced_basis, removed_elements)
    """
    if seed is not None:
        random.seed(seed)
    if k > len(basis):
        k = len(basis) - 1
    removed = random.sample(basis, k)
    reduced_basis = [b for b in basis if b not in removed]
    return reduced_basis, removed

def random_monomial(n_vars: int, max_degree: int) -> Monomial:
    """
    Generate a random monomial with the given number of variables and maximum degree.
    """
    exponents = tuple(np.random.randint(0, max_degree + 1, size=n_vars))
    return Monomial(exponents)

def permute_polynomial_tokens(tokens: List[str], permutation: List[int]) -> List[str]:
    """
    Permute polynomial tokens in the format ['C1.0', 'E3', 'E0', 'E0', '+', 'C1.0', 'E1', 'E1', 'E0', ...].
    
    Args:
        tokens (List[str]): List of polynomial tokens in the format:
                           ['C1.0', 'E3', 'E0', 'E0', '+', 'C1.0', 'E1', 'E1', 'E0', ...]
        permutation (List[int]): List of indices indicating the new positions
                                e.g., [2,0,3,1] means var_2 goes to position 0,
                                var_0 goes to position 1, etc.
    
    Returns:
        List[str]: List of permuted polynomial tokens
    """
    if not tokens:
        return tokens
    
    # Determine number of variables from the first monomial
    num_vars = 0
    for i, token in enumerate(tokens):
        if token.startswith('E'):
            num_vars += 1
        elif token == '+':
            break
    
    if num_vars != len(permutation):
        raise ValueError(f"Permutation length {len(permutation)} does not match number of variables {num_vars}")
    
    result = []
    i = 0
    
    while i < len(tokens):
        # Handle coefficient token
        if tokens[i].startswith('C'):
            result.append(tokens[i])
            i += 1
            
            # Handle exponent tokens for this monomial
            exponents = []
            for j in range(num_vars):
                if i < len(tokens) and tokens[i].startswith('E'):
                    exponents.append(int(tokens[i][1:]))  # Remove 'E' prefix
                    i += 1
                else:
                    exponents.append(0)
            
            # Apply permutation to exponents
            permuted_exponents = [0] * num_vars
            for old_pos, new_pos in enumerate(permutation):
                permuted_exponents[new_pos] = exponents[old_pos]
            
            # Add permuted exponent tokens
            for exp in permuted_exponents:
                result.append(f'E{exp}')
        
        # Handle plus token
        elif tokens[i] == '+':
            result.append(tokens[i])
            i += 1
        
        # Handle any other tokens (shouldn't happen in this format)
        else:
            result.append(tokens[i])
            i += 1
    
    return result


def permute_polynomial_tokens_old_format(tokens: List[str], permutation: List[int]) -> List[str]:
    """
    Permute polynomial tokens in the old M_i_j_k_l format.
    This is kept for backward compatibility.
    
    Args:
        tokens (List[str]): List of polynomial tokens in M_i_j_k_l format
        permutation (List[int]): List of indices indicating the new positions
    
    Returns:
        List[str]: List of permuted polynomial tokens
    """
    return [permute_monomial_token(token, permutation) if token.startswith('M_') else token 
            for token in tokens]


def permute_polynomial_object(polynomial: Polynomial,
                              permutation: List[int],
                              num_vars: int) -> Polynomial:
    """
    Apply a variable permutation π to a Polynomial.

        π[i] == new position of variable i.
    """
    # --- infer variable count -----------------------------------------
    if num_vars is None:
        num_vars = max((len(m.exponents) for m in polynomial.terms), default=0)

    if num_vars != len(permutation):
        raise ValueError("Permutation length and num_vars disagree")

    new_terms: dict[Monomial, float | Fraction] = {}

    for monom, coeff in polynomial.terms.items():
        # pad/truncate exponents to num_vars
        old_exp = list(monom.exponents) + [0]*(num_vars - len(monom.exponents))
        old_exp = old_exp[:num_vars]

        # ---------- core fix: place each exponent in its new slot -------
        new_exp = [0]*num_vars
        for i in range(num_vars):
            new_pos          = permutation[i]     # where var i goes
            new_exp[new_pos] = old_exp[i]

        new_monom = Monomial(tuple(new_exp))
        new_terms[new_monom] = coeff

    return Polynomial(new_terms, rational=polynomial.rational)


def get_even_monomials_sqrt(poly: Polynomial) -> List[Monomial]:
    """
    Extract monomials with all even exponents and return their square roots.
    
    For each monomial in the polynomial with all even exponents, 
    returns the monomial obtained by dividing all exponents by 2.
    
    Args:
        poly: Polynomial object
        
    Returns:
        List of Monomial objects with exponents divided by 2
    """
    sqrt_monomials = []
    
    for monomial in poly.terms.keys():
        # Check if all exponents are even
        if all(exp % 2 == 0 for exp in monomial.exponents):
            # Divide all exponents by 2
            sqrt_exponents = tuple(exp // 2 for exp in monomial.exponents)
            sqrt_monomials.append(Monomial(sqrt_exponents))
    
    return sqrt_monomials


def count_necessary_basis_elements(basis: List[Monomial], poly: Polynomial) -> int:
    """
    Count the number of necessary elements in a basis for covering a polynomial's support.
    
    A basis element is necessary if removing it would result in some element from the 
    polynomial's support no longer being obtainable as a product of two basis elements.
    
    Args:
        basis: List of Monomial objects forming the basis B
        poly: Polynomial object whose support we want to cover
        
    Returns:
        int: Number of necessary basis elements
    """
    from collections import defaultdict
    
    support = set(poly.terms.keys())
    basis_set = set(basis)
    
    # Step 1: Compute B*B with decomposition information
    # products_decomposition[product] = list of (m1, m2) pairs that produce this product
    products_decomposition = defaultdict(list)
    
    for i, m1 in enumerate(basis):
        for j, m2 in enumerate(basis):
            product = m1 * m2
            products_decomposition[product].append((m1, m2))
    
    # Step 2: Find necessary elements
    necessary_elements = set()
    
    for basis_element in basis:
        # Check if removing this basis element would make any support element uncoverable
        is_necessary = False
        
        for support_element in support:
            if support_element in products_decomposition:
                # Get all ways to produce this support element
                decompositions = products_decomposition[support_element]
                
                # Check if all decompositions involve the current basis element
                all_involve_element = True
                for m1, m2 in decompositions:
                    if m1 != basis_element and m2 != basis_element:
                        all_involve_element = False
                        break
                
                # If all decompositions involve this element, it's necessary
                if all_involve_element and len(decompositions) > 0:
                    is_necessary = True
                    break
        
        if is_necessary:
            necessary_elements.add(basis_element)
    
    return len(necessary_elements)


def analyze_basis_coverage(basis: List[Monomial], poly: Polynomial) -> Dict[str, Any]:
    """
    Analyze how a basis covers a polynomial's support, providing detailed information
    about the coverage and necessary elements.
    
    Args:
        basis: List of Monomial objects forming the basis B
        poly: Polynomial object whose support we want to cover
        
    Returns:
        Dict containing:
            - 'products_decomposition': Dict mapping each B*B element to its decompositions
            - 'covered_support': Set of support elements that can be obtained from B*B
            - 'uncovered_support': Set of support elements that cannot be obtained from B*B
            - 'necessary_elements': Set of basis elements that are necessary
            - 'redundant_elements': Set of basis elements that are redundant
            - 'num_necessary': Number of necessary elements
            - 'coverage_ratio': Ratio of covered support elements
    """
    from collections import defaultdict
    
    support = set(poly.terms.keys())
    basis_set = set(basis)
    
    # Compute B*B with decomposition information
    products_decomposition = defaultdict(list)
    
    for i, m1 in enumerate(basis):
        for j, m2 in enumerate(basis):
            product = m1 * m2
            products_decomposition[product].append((m1, m2))
    
    # Find covered and uncovered support elements
    all_products = set(products_decomposition.keys())
    covered_support = support.intersection(all_products)
    uncovered_support = support - all_products
    
    # Find necessary and redundant elements
    necessary_elements = set()
    
    for basis_element in basis:
        is_necessary = False
        
        for support_element in covered_support:
            decompositions = products_decomposition[support_element]
            
            # Check if all decompositions involve the current basis element
            all_involve_element = True
            for m1, m2 in decompositions:
                if m1 != basis_element and m2 != basis_element:
                    all_involve_element = False
                    break
            
            if all_involve_element and len(decompositions) > 0:
                is_necessary = True
                break
        
        if is_necessary:
            necessary_elements.add(basis_element)
    
    redundant_elements = basis_set - necessary_elements
    
    # Compute combinatorial bound
    num_support_elements = len(support)
    combinatorial_bound = (np.sqrt(1 + 8 * num_support_elements) - 1) / 2
    
    return {
        'products_decomposition': dict(products_decomposition),
        'covered_support': covered_support,
        'uncovered_support': uncovered_support,
        'necessary_elements': necessary_elements,
        'redundant_elements': redundant_elements,
        'num_necessary': len(necessary_elements),
        'coverage_ratio': len(covered_support) / len(support) if support else 1.0,
        'num_support_elements': num_support_elements,
        'combinatorial_bound': combinatorial_bound
    }
