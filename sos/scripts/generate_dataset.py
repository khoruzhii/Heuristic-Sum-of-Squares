"""
Script to generate a large dataset of polynomial-basis pairs.
Each example consists of a polynomial and its corresponding monomial basis,
separated by a [BIGSEP] token.
"""

import os
import json
import logging
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from tqdm import tqdm
import wandb
from multiprocessing import Pool, Manager
from functools import partial

# Add the sos/src directory to Python path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# print the current working directory
print(os.getcwd())
print("Python path:", sys.path[0])

from data_generation.polynomials.polynomial_base import PolynomialSampler
from data_generation.polynomials.polynomial_sos import SOSPolynomialSampler
from data_generation.matrix.matrix_simple import SimpleRandomPSDSampler
from data_generation.matrix.matrix_non_psd import SimpleRandomNonPSDSampler
from data_generation.matrix.matrix_sparse import SparsePSDSampler
from data_generation.matrix.matrix_lowrank import LowRankPSDSampler
from data_generation.matrix.matrix_blockdiag import BlockDiagonalPSDSampler
from data_generation.monomials.monomial_sparseuniform import SparseUniformBasisSampler
from data_generation.monomials.monomial_clique import CliqueBasisSampler
from data_generation.monomials.monomials import (
    Monomial,
    Polynomial,
    MonomialBasis
)
from utils.polynomial import get_newton_polytope_basis_mip, get_newton_polytope_basis, permute_polynomial_object
from utils.symmetry import canonicalise2

# Constants
BATCH_SIZE = 20000  # Process 20000 examples at a time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monomial_to_point(monomial: Monomial) -> np.ndarray:
    """Convert a Monomial object to a point (exponent vector)"""
    return np.array(monomial.exponents)

def point_to_monomial(point: np.ndarray, num_vars: int) -> Monomial:
    """Convert a point (exponent vector) to a Monomial object"""
    # Ensure point has the right length
    if len(point) < num_vars:
        point = np.pad(point, (0, num_vars - len(point)))
    return Monomial(exponents=tuple(map(int, point)))

def get_newton_polytope_monomials(poly: Polynomial, num_vars: int) -> List[Monomial]:
    """
    Compute the Newton polytope basis monomials for a polynomial.
    Returns a list of Monomial objects.
    """
    # Convert polynomial terms to points
    points = []
    for monomial in poly.terms.keys():
        points.append(monomial_to_point(monomial))
    
    """
    # Get Newton polytope basis points
    basis_points = get_newton_polytope_basis(np.array(points))
    if basis_points is None:
        return []
    
    
    # Convert points back to monomials
    basis_monomials = []
    for point in basis_points:
        basis_monomials.append(point_to_monomial(point, num_vars))
    """
    basis_monomials = []
    
    return basis_monomials

@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    # Dataset size and split
    num_examples: int = 100
    train_ratio: float = 0.99
    val_ratio: float = 0.005
    test_ratio: float = 0.005
    
    # Polynomial parameters
    num_variables: int = 5

    # basis sampler 
    basis_sampler: str = "sparse_uniform" # options: "sparse_uniform", "clique"
    max_degree: int = 5
    min_sparsity: float = 0.3
    max_sparsity: float = 0.5
    min_degree: int = 1
    num_monomials: int = 30

    min_cliques: int = 2
    max_cliques: int = 4
    min_clique_size: int = 2
    max_clique_size: int = 3
    max_degree_per_clique: int = 4
    
    # Matrix parameters
    matrix_sampler: str = "simple_random" # options: "simple_random", "sparse", "lowrank", "blockdiag"   
    min_eigenval: float = 0
    matrix_scale: float = 1.0
    sparsity: float = 0.1
    max_denominator: int = 5
    max_numerator: int = 5
    min_sparsity_matrix: float = 0.01
    max_sparsity_matrix: float = 0.2

    max_rank: int = 1
    min_rank: int = 1
    max_block_size: int = 3

    rational: bool = False
    digits: int = 1
    
    # Canonicalization method
    canonicalization: str = "none"  # options: "none", "graph"
    
    # Monomial ordering
    monomial_order: str = "lex"  # options: "lex" for lexicographic, anything else for no sorting
    
    # Tokenizer parameters
    field: str = "QQ"
    max_coeff: int = 100
    
    # Output parameters
    output_dir: str = "/scratch/htc/npelleriti/data/sos-transformer/toy"
    random_seed: int = 42
    
    # Wandb parameters
    use_wandb: bool = False
    wandb_project: str = "sos-transformer-dataset"
    wandb_run_name: Optional[str] = None
    
    @classmethod
    def from_args_and_wandb(cls, args: argparse.Namespace) -> 'DatasetConfig':
        """Create config from command line args and wandb config (if using wandb)"""
        # Start with default config instance
        default_config = cls()
        config_dict = asdict(default_config)
        
        # Override with command line arguments (only if not None)
        for key, value in vars(args).items():
            if value is not None and hasattr(cls, key):
                config_dict[key] = value
        
        # Override with wandb config if using wandb
        if getattr(args, 'use_wandb', False):
            wandb_config = wandb.config
            for key in config_dict.keys():
                if key in wandb_config:
                    config_dict[key] = wandb_config[key]
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging"""
        return asdict(self)

def setup_samplers(config: DatasetConfig) -> SOSPolynomialSampler:
    """Set up the polynomial sampler with its components"""
    # Create the component samplers
    if config.basis_sampler == "sparse_uniform":
        basis_sampler = SparseUniformBasisSampler(
            min_sparsity=config.min_sparsity,
            max_sparsity=config.max_sparsity,
            min_degree=config.min_degree,
            max_degree=config.max_degree,
            num_monomials=config.num_monomials,
            num_vars=config.num_variables
        )
    elif config.basis_sampler == "clique":
        basis_sampler = CliqueBasisSampler(
            min_cliques=config.min_cliques,
            max_cliques=config.max_cliques,
            min_clique_size=config.min_clique_size,
            max_clique_size=config.max_clique_size,
            max_degree_per_clique=config.max_degree_per_clique
        )
    else:
        raise ValueError(f"Invalid basis sampler: {config.basis_sampler}")
    
    if config.matrix_sampler == "simple_random":
        matrix_sampler = SimpleRandomPSDSampler(
            min_eigenval=config.min_eigenval,
            scale=config.matrix_scale,
            random_state=config.random_seed,
            rational=config.rational
        )
    elif config.matrix_sampler == "sparse":
        matrix_sampler = SparsePSDSampler(
            min_eigenval=config.min_eigenval,
            scale=config.matrix_scale,
            sparsity=config.sparsity,
            random_state=config.random_seed,
            max_numerator=config.max_numerator,
            max_denominator=config.max_denominator,
            rational=config.rational,
            min_sparsity=config.min_sparsity_matrix,
            max_sparsity=config.max_sparsity_matrix
        )
    elif config.matrix_sampler == "lowrank":
        matrix_sampler = LowRankPSDSampler(
            min_eigenval=config.min_eigenval,
            random_state=config.random_seed,
            rational=config.rational,
            max_numerator=config.max_numerator,
            max_denominator=config.max_denominator,
            max_rank=config.max_rank,
            min_rank=config.min_rank
        )
    elif config.matrix_sampler == "blockdiag":
        matrix_sampler = BlockDiagonalPSDSampler(
            min_eigenval=config.min_eigenval,
            max_block_size=config.max_block_size,
            scale=config.matrix_scale,
            random_state=config.random_seed,
            rational=config.rational,
            max_numerator=config.max_numerator,
            max_denominator=config.max_denominator
        )
    elif config.matrix_sampler == "non_psd":
        matrix_sampler = SimpleRandomNonPSDSampler(
            min_eigenval=-1.0,
            scale=config.matrix_scale,
            random_state=config.random_seed,
            rational=config.rational
        )
    else:
        raise ValueError(f"Invalid matrix sampler: {config.matrix_sampler}")
    
    # Create the polynomial sampler
    return SOSPolynomialSampler(
        basis_sampler=basis_sampler,
        matrix_sampler=matrix_sampler, 
        rational=config.rational
    )
"""
def setup_tokenizer(config: DatasetConfig) -> MonomialTokenizer:
    tokenizer_config = MonomialTokenizerConfig(
        num_variables=config.num_variables,
        max_degree=config.max_degree * 2,  # Double max_degree as polynomials are squared
        field=config.field,
        max_coeff=config.max_coeff
    )
    return MonomialTokenizer(tokenizer_config)
"""
def monomial_to_str(monomial: Monomial) -> str:
    """Convert a Monomial object to our string format"""
    if not monomial.exponents:  # Handle empty monomial
        return "1"
        
    parts = []
    for idx, exp in enumerate(monomial.exponents):
        if exp > 0:
            if exp == 1:
                parts.append(f"x{idx+1}")
            else:
                parts.append(f"x{idx+1}^{exp}")
    return "*".join(parts) if parts else "1"

def polynomial_to_str(poly: Polynomial) -> str:
    """Convert a Polynomial object to our string format"""
    terms = []
    for monomial, coeff in poly.terms.items():
        # Skip zero coefficients
        if abs(coeff) < 1e-10:
            continue
            
        # Format the term
        monomial_str = monomial_to_str(monomial)
        if abs(coeff - 1.0) < 1e-10 and monomial_str != "1":
            terms.append(monomial_str)
        elif abs(coeff + 1.0) < 1e-10:
            terms.append(f"-{monomial_str}")
        else:
            terms.append(f"{coeff:.4f}*{monomial_str}")
    
    return " + ".join(terms) if terms else "0"

def generate_example(
    sampler: SOSPolynomialSampler,
    tokenizer: None,
    config: DatasetConfig
) -> Dict[str, str]:
    """Generate a single example with tokenization"""

    
    # Sample polynomial and basis
    if config.rational:
        poly, basis, Q = sampler.sample(
            num_vars=config.num_variables,
            max_degree=config.max_degree
        )
    else:
        poly, basis, Q = sampler.sample(
            num_vars=config.num_variables,
            max_degree=config.max_degree
        )
    
    # Apply canonicalization if requested
    if config.canonicalization == "graph":
        # Apply graph canonicalization: canonicalise2 + permute_polynomial_object
        can_perm = canonicalise2(poly, config.num_variables)
        poly = permute_polynomial_object(poly, can_perm, config.num_variables)
        
        # Also canonicalize the basis by applying the same permutation
        canonical_basis = []
        for monomial in basis:
            # Create a single-term polynomial for this monomial
            mono_poly = Polynomial({monomial: 1.0}, rational=config.rational)
            # Apply the same permutation
            canonical_mono_poly = permute_polynomial_object(mono_poly, can_perm, config.num_variables)
            # Extract the monomial (should be the only term)
            if canonical_mono_poly.terms:
                canonical_monomial = list(canonical_mono_poly.terms.keys())[0]
                canonical_basis.append(canonical_monomial)
        basis = canonical_basis
    
    # Get Newton polytope basis
    time_start = time.time()
    newton_basis = []#get_newton_polytope_monomials(poly, config.num_variables)
    time_end = time.time()
    
    # Convert polynomial to our string format and tokenize
    poly_str = polynomial_to_str(poly)
    poly_tokens = poly.to_sequence(config.num_variables, digits=config.digits, sort_polynomials=(config.monomial_order == "lex"))

    # sort basis lexicographically if requested
    if config.monomial_order == "lex":
        basis = sorted(basis, key=lambda x: x.exponents)
    
    # Convert basis to our string format and tokenize
    basis_tokens = []
    for i, m in enumerate(basis):
        basis_tokens.extend(m.to_sequence(config.num_variables, config.rational))
        if i < len(basis) - 1:  # Add plus token between monomials, but not after the last one
            basis_tokens.append("+")
    
    # Convert Newton polytope basis to string and tokenize
    
    newton_basis_tokens = []
    for i, m in enumerate(newton_basis):
        newton_basis_tokens.extend(m.to_sequence(config.num_variables, config.rational))
        if i < len(newton_basis) - 1:  # Add plus token between monomials, but not after the last one
            newton_basis_tokens.append("+")

    # check if basis is a subset of newton basis
    basis_subset = all(m in newton_basis for m in basis)
    if not basis_subset:
        #print(f"basis is not a subset of newton basis")
        #print(f"basis: {basis}")
        #print(f"newton_basis: {newton_basis}")
        pass
    
    # Combine with [BIGSEP] token
    combined_tokens = poly_tokens + ["[BIGSEP]"] + basis_tokens
    
    return {
        #"polynomial": poly_str,
        #tokens": combined_tokens,
        "polynomial_tokens": poly_tokens,
        "basis_tokens": basis_tokens,
        #"newton_basis_tokens": newton_basis_tokens,
        #"matrix": Q.tolist(),  # Include the Q matrix for verification
        "newton_basis_size": len(newton_basis),
        "actual_basis_size": len(basis),
        "canonicalization_method": config.canonicalization
    }

def save_dataset_batch(examples: List[Dict], split: str, output_dir: str, mode: str = "a", num_variables: int = 5, basis_sampler: str = "sparse_uniform", max_degree: int = 5, num_monomials: int = 30, matrix_sampler: str = "simple_random"):
    """Save a batch of examples to a jsonl file"""
    output_path = Path(output_dir) / f"n{num_variables}_{basis_sampler}_{matrix_sampler}_d{max_degree}_m{num_monomials}" / f"{split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode) as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for dataset generation"""
    
    class BooleanOptionalAction(argparse.Action):
        """Custom action to handle boolean arguments from both CLI and wandb"""
        def __init__(self, option_strings, dest, nargs=None, default=None, type=None, choices=None, required=False, help=None, metavar=None):
            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)
                if option_string.startswith('--'):
                    option_string = '--no-' + option_string[2:]
                    _option_strings.append(option_string)
            super().__init__(option_strings=_option_strings, dest=dest, nargs='?', default=default, type=self._str2bool, choices=None, required=required, help=help, metavar=metavar)

        def _str2bool(self, v):
            if isinstance(v, bool):
                return v
            if v is None:
                return True  # For flag-style usage
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string and option_string.startswith('--no-'):
                setattr(namespace, self.dest, False)
            else:
                setattr(namespace, self.dest, self._str2bool(values))

        def format_usage(self):
            return ' | '.join(self.option_strings)
    
    parser = argparse.ArgumentParser(description='Generate polynomial-basis dataset')
    
    # Dataset size and split
    parser.add_argument('--num_examples', type=int, help='Number of examples to generate')
    parser.add_argument('--train_ratio', type=float, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, help='Test set ratio')
    
    # Polynomial parameters - sparse uniform
    parser.add_argument('--num_variables', type=int, help='Number of variables')
    parser.add_argument('--basis_sampler', type=str, help='sparse_uniform or clique')
    parser.add_argument('--max_degree', type=int, help='Maximum degree')
    parser.add_argument('--min_sparsity', type=float, help='Minimum sparsity')
    parser.add_argument('--max_sparsity', type=float, help='Maximum sparsity')
    parser.add_argument('--min_degree', type=int, help='Minimum degree')
    parser.add_argument('--num_monomials', type=int, help='Number of monomials')
    # Polynomial parameters - clique
    parser.add_argument('--min_cliques', type=int, help='Minimum number of cliques')
    parser.add_argument('--max_cliques', type=int, help='Maximum number of cliques')
    parser.add_argument('--min_clique_size', type=int, help='Minimum size of cliques')
    parser.add_argument('--max_clique_size', type=int, help='Maximum size of cliques')
    parser.add_argument('--max_degree_per_clique', type=int, help='Maximum degree per clique')
    
    # Matrix parameters
    parser.add_argument('--matrix_sampler', type=str, help='simple_random, sparse, lowrank, or blockdiag')
    parser.add_argument('--min_eigenval', type=float, help='Minimum eigenvalue')
    parser.add_argument('--matrix_scale', type=float, help='Matrix scale')
    parser.add_argument('--sparsity', type=float, help='Sparsity')
    parser.add_argument('--max_denominator', type=int, help='Maximum denominator')
    parser.add_argument('--max_numerator', type=int, help='Maximum numerator')
    parser.add_argument('--min_sparsity_matrix', type=float, help='Minimum sparsity')
    parser.add_argument('--max_sparsity_matrix', type=float, help='Maximum sparsity')
    parser.add_argument('--max_rank', type=int, help='Maximum rank')
    parser.add_argument('--min_rank', type=int, help='Minimum rank')
    parser.add_argument('--max_block_size', type=int, help='Maximum block size for block diagonal matrices')

    # Coefficient parameters
    parser.add_argument('--rational', type=bool, help='Use rational numbers')
    parser.add_argument('--digits', type=int, help='Number of digits')
    
    # Canonicalization method
    parser.add_argument('--canonicalization', type=str, choices=['none', 'graph'], help='Canonicalization method: "none" for no canonicalization, "graph" for canonicalise2 + permute_polynomial_object')
    
    # Monomial ordering
    parser.add_argument('--monomial_order', type=str, help='Monomial ordering: "lex" for lexicographic, anything else for no sorting')
    
    # Tokenizer parameters
    parser.add_argument('--field', type=str, help='Field for tokenizer')
    parser.add_argument('--max_coeff', type=int, help='Maximum coefficient')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--random_seed', type=int, help='Random seed')
    
    # Wandb parameters - using custom action to handle both CLI and wandb formats
    parser.add_argument('--use_wandb', action=BooleanOptionalAction, help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, help='Wandb run name')
    
    return parser

def generate_example_wrapper(config: DatasetConfig, seed: int) -> Optional[Dict[str, str]]:
    """Wrapper for parallel example generation with proper seed management."""
    # Set different seed for each process
    np.random.seed(seed)
    sampler = setup_samplers(config)
    try:
        return generate_example(sampler, None, config)
    except Exception as e:
        logging.error(f"Failed to generate example with seed {seed}: {str(e)}")
        return None

def main():
    """Generate the dataset"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Initialize wandb if requested (without config for now)
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project or "sos-transformer-dataset",
            name=args.wandb_run_name
        )
    
    # Load configuration from args and wandb
    config = DatasetConfig.from_args_and_wandb(args)
    
    # Log configuration
    logger.info(f"Dataset generation configuration: {config.to_dict()}")
    if config.use_wandb:
        wandb.config.update(config.to_dict())
    
    # Calculate split sizes
    train_size = int(config.num_examples * config.train_ratio)
    val_size = int(config.num_examples * config.val_ratio)
    test_size = config.num_examples - train_size - val_size
    
    # Clear existing files
    for split in ["train", "val", "test"]:
        output_path = Path(config.output_dir) / split
        output_path.mkdir(parents=True, exist_ok=True)
        if (output_path / "examples.jsonl").exists():
            (output_path / "examples.jsonl").unlink()
    
    # Initialize statistics
    generation_stats = {
        'successful_examples': 0,
        'failed_examples': 0,
        'avg_newton_basis_size': 0,
        'avg_actual_basis_size': 0,
        'avg_polynomial_length': 0
    }
    
    # Create a partial function with fixed config
    gen_func = partial(generate_example_wrapper, config)
    
    # Generate different seeds for each example
    base_seed = config.random_seed
    seeds = [base_seed + i for i in range(config.num_examples)]
    
    # Use number of CPUs minus 1 to leave one for system
    num_processes = min(12, max(1, os.cpu_count() - 1))
    
    logger.info(f"Generating {config.num_examples} examples using {num_processes} processes...")
    
    # Initialize batch storage
    batch_examples = {"train": [], "val": [], "test": []}
    current_train = current_val = current_test = 0
    last_log_time = time.time()
    
    with Pool(processes=num_processes) as pool:
        # Generate examples in parallel with progress bar
        for example in tqdm(pool.imap_unordered(gen_func, seeds), total=config.num_examples):
            if example is not None:
                # Determine split based on current counts
                if current_train < train_size:
                    split = "train"
                    current_train += 1
                elif current_val < val_size:
                    split = "val"
                    current_val += 1
                else:
                    split = "test"
                    current_test += 1
                
                # Add example to batch
                batch_examples[split].append(example)
                
                # Update statistics
                generation_stats['successful_examples'] += 1
                generation_stats['avg_newton_basis_size'] += example['newton_basis_size']
                generation_stats['avg_actual_basis_size'] += example['actual_basis_size']
                generation_stats['avg_polynomial_length'] += len(example['polynomial_tokens'])
                
                # Save batch if it reaches BATCH_SIZE
                if len(batch_examples[split]) >= BATCH_SIZE:
                    save_dataset_batch(batch_examples[split], split, config.output_dir, num_variables=config.num_variables, basis_sampler=config.basis_sampler, max_degree=config.max_degree, num_monomials=config.num_monomials, matrix_sampler=config.matrix_sampler)
                    batch_examples[split] = []
                
                # Log to wandb periodically (every 60 seconds)
                if config.use_wandb and time.time() - last_log_time > 60:
                    wandb.log({
                        'examples_generated': generation_stats['successful_examples'],
                        'success_rate': generation_stats['successful_examples'] / (generation_stats['successful_examples'] + generation_stats['failed_examples'])
                    })
                    last_log_time = time.time()
            else:
                generation_stats['failed_examples'] += 1
                if config.use_wandb:
                    wandb.log({'generation_error': 1})
    
    # Save any remaining examples in batches
    for split, examples in batch_examples.items():
        if examples:
            save_dataset_batch(examples, split, config.output_dir, num_variables=config.num_variables, basis_sampler=config.basis_sampler, max_degree=config.max_degree, num_monomials=config.num_monomials, matrix_sampler=config.matrix_sampler)
    
    # Calculate final statistics
    if generation_stats['successful_examples'] > 0:
        generation_stats['avg_newton_basis_size'] /= generation_stats['successful_examples']
        generation_stats['avg_actual_basis_size'] /= generation_stats['successful_examples']
        generation_stats['avg_polynomial_length'] /= generation_stats['successful_examples']
    
    # Save config
    config_path = Path(config.output_dir) / f"n{config.num_variables}_{config.basis_sampler}_{config.matrix_sampler}_d{config.max_degree}_m{config.num_monomials}" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Final logging
    final_stats = {
        'total_examples': generation_stats['successful_examples'],
        'train_examples': current_train,
        'val_examples': current_val,
        'test_examples': current_test,
        **generation_stats
    }
    
    logger.info("Dataset generation complete!")
    for key, value in final_stats.items():
        logger.info(f"{key}: {value}")
    
    if config.use_wandb:
        wandb.log(final_stats)
        wandb.finish()
    
    return final_stats

if __name__ == "__main__":
    main() 