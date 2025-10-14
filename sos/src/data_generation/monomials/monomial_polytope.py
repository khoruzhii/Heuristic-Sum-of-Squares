import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import ConvexHull
from data_generation.monomials.monomial_base import BasisSampler
from data_generation.monomials.monomials import Monomial, MonomialBasis

class PolytopeBasisSampler(BasisSampler):
    """Samples monomial bases by sampling points from a polytope in the exponent space."""
    
    def __init__(self,
                 vertices: Optional[np.ndarray] = None,
                 num_monomials: int = 10,
                 constant_term_prob: float = 1.0):
        """
        Initialize the polytope-based sampler.
        
        Args:
            vertices: Array of shape (n_vertices, n_vars) defining the vertices of the polytope
                     If None, a default polytope will be created based on degree bounds
            num_monomials: Number of monomials to sample
            constant_term_prob: Probability of including a constant term
        """
        self.vertices = vertices
        self.num_monomials = num_monomials
        self.constant_term_prob = constant_term_prob
        self.hull = None
        
    def _create_default_polytope(self, num_vars: int, max_degree: int):
        """Create a default polytope based on degree bounds."""
        # Create vertices for a simplex-like polytope
        # Include origin and points along each axis up to max_degree
        vertices = [[0] * num_vars]  # Origin
        for i in range(num_vars):
            vertex = [0] * num_vars
            vertex[i] = max_degree
            vertices.append(vertex)
        
        self.vertices = np.array(vertices)
        
    def _sample_from_polytope(self, num_points: int) -> np.ndarray:
        """Sample points from inside the polytope using hit-and-run sampling."""
        if self.hull is None:
            self.hull = ConvexHull(self.vertices)
            
        points = []
        n_vars = self.vertices.shape[1]
        
        # Get bounding box
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        
        while len(points) < num_points:
            # Sample random point in bounding box
            point = np.random.uniform(min_coords, max_coords)
            
            # Check if point is in polytope using ConvexHull
            if self._point_in_hull(point):
                # Round to nearest integer (since exponents must be integers)
                point = np.round(point).astype(int)
                points.append(point)
                
        return np.array(points)
    
    def _point_in_hull(self, point: np.ndarray) -> bool:
        """Check if a point lies inside the convex hull."""
        if self.hull is None:
            return False
            
        # Get equations defining the hull
        equations = self.hull.equations
        
        # Point is inside if all equations evaluate to <= 0
        return all(np.dot(eq[:-1], point) + eq[-1] <= 1e-10 for eq in equations)
    
    def sample(self, num_vars: int, max_degree: int) -> MonomialBasis:
        """
        Sample a monomial basis using points from the polytope.
        
        Args:
            num_vars: Number of variables
            max_degree: Maximum degree of monomials
            
        Returns:
            A list of Monomial objects forming the basis
        """
        # Create default polytope if none provided
        if self.vertices is None:
            self._create_default_polytope(num_vars, max_degree)
        
        basis = []
        
        # Add constant term with specified probability
        if np.random.random() < self.constant_term_prob:
            basis.append(Monomial(tuple([0] * num_vars)))
        
        # Sample remaining points from polytope
        remaining_terms = self.num_monomials - len(basis)
        if remaining_terms > 0:
            points = self._sample_from_polytope(remaining_terms)
            
            # Convert points to monomials
            for point in points:
                monomial = Monomial(tuple(point))
                if monomial not in basis:  # Avoid duplicates
                    basis.append(monomial)
        
        return basis
    
    def __repr__(self) -> str:
        return (f"PolytopeBasisSampler(vertices={self.vertices}, "
                f"num_monomials={self.num_monomials}, "
                f"constant_term_prob={self.constant_term_prob})")

if __name__ == "__main__":
    # Example 1: Default polytope (simplex-like)
    sampler1 = PolytopeBasisSampler(
        num_monomials=5,
        constant_term_prob=1.0
    )
    print("\nExample 1: Default polytope (simplex-like)")
    basis1 = sampler1.sample(num_vars=3, max_degree=3)
    print("Generated basis:")
    for monomial in basis1:
        print(f"  {monomial} (degree: {monomial.degree})")

    # Example 2: Custom polytope (box constraint)
    # Define vertices of a box: 0 ≤ x_i ≤ 2 for all i
    box_vertices = np.array([
        [0, 0, 0],  # Origin
        [2, 0, 0],  # Points along axes
        [0, 2, 0],
        [0, 0, 2],
        [2, 2, 0],  # Points on faces
        [2, 0, 2],
        [0, 2, 2],
        [2, 2, 2]   # Maximum point
    ])
    sampler2 = PolytopeBasisSampler(
        vertices=box_vertices,
        num_monomials=6,
        constant_term_prob=0.5  # 50% chance of including constant term
    )
    print("\nExample 2: Box constraint polytope")
    print("Vertices shape:", box_vertices.shape)
    basis2 = sampler2.sample(num_vars=3, max_degree=2)
    print("Generated basis:")
    for monomial in basis2:
        print(f"  {monomial} (degree: {monomial.degree})")

    # Example 3: Custom polytope (pyramid-like)
    # Define vertices of a pyramid in 3D
    pyramid_vertices = np.array([
        [0, 0, 0],  # Base
        [2, 0, 0],
        [0, 2, 0],
        [2, 2, 0],
        [1, 1, 3]   # Apex
    ])
    sampler3 = PolytopeBasisSampler(
        vertices=pyramid_vertices,
        num_monomials=7
    )
    print("\nExample 3: Pyramid-like polytope")
    print("Vertices shape:", pyramid_vertices.shape)
    basis3 = sampler3.sample(num_vars=3, max_degree=3)
    print("Generated basis:")
    for monomial in basis3:
        print(f"  {monomial} (degree: {monomial.degree})") 