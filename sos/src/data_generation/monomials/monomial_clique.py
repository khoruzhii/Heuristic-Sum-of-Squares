import numpy as np
from typing import Tuple, List, Set
import itertools

from data_generation.monomials.monomial_base import BasisSampler
from data_generation.monomials.monomials import Monomial, MonomialBasis



class CliqueBasisSampler(BasisSampler):
    """Samples monomial bases based on variable cliques to create correlative sparsity."""

    def __init__(self,
                 min_cliques: int = 2,
                 max_cliques: int = 5,
                 min_clique_size: int = 2,
                 max_clique_size: int = 4,
                 max_degree_per_clique: int = 4,
                 num_monomials_per_clique: int = 5,
                 constant_term_prob: float = 0.8):
        """
        Initialize the sampler with parameters for clique-based generation.

        Args:
            min_cliques: Minimum number of variable cliques to generate.
            max_cliques: Maximum number of variable cliques to generate.
            min_clique_size: Minimum number of variables in a clique.
            max_clique_size: Maximum number of variables in a clique.
            max_degree_per_clique: Max degree of monomials generated within a clique.
            num_monomials_per_clique: Number of monomials to sample from each clique.
            constant_term_prob: Probability of including a constant term in the final basis.
        """
        self.min_cliques = min_cliques
        self.max_cliques = max_cliques
        self.min_clique_size = min_clique_size
        self.max_clique_size = max_clique_size
        self.max_degree_per_clique = max_degree_per_clique
        self.num_monomials_per_clique = num_monomials_per_clique
        self.constant_term_prob = constant_term_prob

    def _generate_monomials_for_clique(self, clique_indices: List[int]) -> List[Tuple[int, ...]]:
        """Generates all possible monomials for a given clique of variables."""
        clique_size = len(clique_indices)
        monomial_exponents_local = []

        # Recursive function to find combinations of exponents
        def find_exponent_combinations(remaining_degree, current_exponents):
            if len(current_exponents) == clique_size:
                monomial_exponents_local.append(tuple(current_exponents))
                return
            
            for d in range(remaining_degree + 1):
                find_exponent_combinations(remaining_degree - d, current_exponents + [d])

        find_exponent_combinations(self.max_degree_per_clique, [])
        # Filter out the zero-degree monomial as it's handled separately
        return [exp for exp in monomial_exponents_local if sum(exp) > 0]

    def sample(self, num_vars: int, max_degree: int = -1) -> MonomialBasis:
        """
        Sample a sparse monomial basis using the clique method.

        Args:
            num_vars: Total number of variables in the polynomial.
            max_degree: This argument is ignored for this sampler, as degree is
                        controlled per clique by `max_degree_per_clique`.

        Returns:
            A list of Monomial objects forming the basis.
        """
        final_basis_set: Set[Monomial] = set()

        # 1. Generate random cliques
        num_cliques = np.random.randint(self.min_cliques, self.max_cliques + 1)
        cliques = []
        for _ in range(num_cliques):
            clique_size = np.random.randint(self.min_clique_size, self.max_clique_size + 1)
            clique_size = min(clique_size, num_vars) # Ensure clique is not larger than num_vars
            clique_indices = np.random.choice(num_vars, size=clique_size, replace=False).tolist()
            cliques.append(clique_indices)

        # 2. For each clique, sample monomials
        for clique in cliques:
            # Generate all possible local monomials for the clique
            local_monomials = self._generate_monomials_for_clique(clique)
            
            if not local_monomials:
                continue

            # Sample a subset of these monomials
            num_to_sample = min(self.num_monomials_per_clique, len(local_monomials))
            selected_indices = np.random.choice(len(local_monomials), size=num_to_sample, replace=False)

            # Map local monomials to the full variable space
            for idx in selected_indices:
                full_exponent_vec = [0] * num_vars
                local_exponents = local_monomials[idx]
                for i, exponent in enumerate(local_exponents):
                    variable_index_in_full_space = clique[i]
                    full_exponent_vec[variable_index_in_full_space] = exponent
                
                final_basis_set.add(Monomial(tuple(full_exponent_vec)))
        
        # 3. Handle constant term
        if np.random.random() < self.constant_term_prob:
            final_basis_set.add(Monomial(tuple([0] * num_vars)))

        # Ensure basis is not empty
        if not final_basis_set:
            return [Monomial(tuple([0] * num_vars))]

        return list(final_basis_set)

    def __repr__(self) -> str:
        return (f"CliqueBasisSampler(cliques=({self.min_cliques},{self.max_cliques}), "
                f"size=({self.min_clique_size},{self.max_clique_size}), "
                f"degree={self.max_degree_per_clique})")


if __name__ == "__main__":
    # Create a sampler that generates 2-3 cliques, each with 2-3 variables.
    clique_sampler = CliqueBasisSampler(
        min_cliques=2,
        max_cliques=3,
        min_clique_size=2,
        max_clique_size=3,
        max_degree_per_clique=3,
        num_monomials_per_clique=4,
        constant_term_prob=1.0
    )

    # Sample a basis for a 6-variable problem
    basis = clique_sampler.sample(num_vars=5)
    
    # Sort for consistent display
    sorted_basis = sorted(basis, key=lambda m: m.exponents)

    print(f"Generated basis from sampler: {clique_sampler}")
    print(f"Total monomials in basis: {len(sorted_basis)}")
    for monomial in sorted_basis:
        print(f"  {monomial} (degree: {monomial.degree})")