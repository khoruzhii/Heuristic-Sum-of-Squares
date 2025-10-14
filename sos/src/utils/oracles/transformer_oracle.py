import time
import os
import sys
from typing import List, Any, Dict
from torch.utils.data import DataLoader
from collections import Counter
from itertools import chain

# Add the transformer/src directory to Python path
transformer_src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'transformer', 'src')
sys.path.insert(0, transformer_src_path)

from .oracle_base import OracleBase
from data_generation.monomials.monomials import Polynomial
from utils.basis_extension import basis_extension


from loader.oracle import TransformerOracle as TransformerOracleImpl
from loader.checkpoint import load_pretrained_bag
from loader.data_format.processors.subprocessors import MonomialProcessorPlus
from misc.utils import to_cuda
from evaluation.generation import generation
from loader.data import load_data


class TransformerOracle(OracleBase):
    def __init__(self, use_basis_extension: bool = False, basis_extension_params: Dict = None, permutations: int = 1, mode: str = 'single', **transformer_kwargs):
        super().__init__(use_basis_extension, basis_extension_params)
        # Initialize the actual transformer oracle
        print(f"Transformer kwargs: {transformer_kwargs}")
        self.transformer_oracle = TransformerOracleImpl(**transformer_kwargs)
        print(f"Transformer oracle initialized successfully.")
        self.permutations = permutations
        self.mode = mode

    def _parse_basis_string(self, basis_str: str) -> List:
        """
        Parse the basis string returned by the transformer oracle.
        This is a placeholder - implement based on your specific format.
        """
        basis_list = basis_str.split()
        basis_terms = [term for term in Polynomial.from_sequence(basis_list).terms]

        return basis_terms
        
    
    def generate_from_string(self, tokenized_poly: str, max_length: int = 2048):
        """Wrapper to use the transformer oracle's generate_from_string method"""
        basis_string = self.transformer_oracle.generate_from_string(tokenized_poly, max_length)

        basis_terms = self._parse_basis_string(basis_str=basis_string)
        
        tokenized_poly = tokenized_poly.replace("[C]", "C1.0")
        polynomial = Polynomial.from_sequence(tokenized_poly.split())
        
        # do basis extension if necessary
        if self.use_basis_extension:
            basis_terms = basis_extension(basis_terms, polynomial)

        return basis_terms

    
    def generate_from_batch(self, batch, max_length: int = 2048):
        """Wrapper to use the transformer oracle's generate_from_batch method"""
        return self.transformer_oracle.generate_from_batch(batch, max_length)

    def predict_basis(self, **kwargs) -> Dict:
        # User must provide the Polynomial object as 'poly' in kwargs
        poly = kwargs.get('poly')
        poly_tokens = kwargs.get('poly_tokens')
        if poly is None:
            raise ValueError("TransformerOracle requires the 'poly' argument (Polynomial object)")
        if poly_tokens is None:
            raise ValueError("TransformerOracle requires the 'poly_tokens' argument")
            
        start_time = time.time()
        
        # Convert poly_tokens to string format expected by transformer oracle
        if isinstance(poly_tokens, list):
            # Convert token list to string format
            poly_str = " ".join([str(token) for token in poly_tokens])
        else:
            poly_str = str(poly_tokens)
        
        # Use the transformer oracle to generate basis
        if self.mode == "single":
            predicted_basis_str = self.transformer_oracle.generate_from_string(poly_str)
        elif self.mode == "permutation_union":
            predicted_basis_str = " + ".join(self.transformer_oracle.generate_from_string_with_permutations(poly_str, num_permutations=self.permutations)["union_basis"])
        elif self.mode == "permutation_intersection":
            predicted_basis_str = " + ".join(self.transformer_oracle.generate_from_string_with_permutations(poly_str, num_permutations=self.permutations)["intersection_basis"])
        elif self.mode == "permutation_all":
            # all_permuted_bases is a list of lists of strings; flatten to a list of strings
            all_permuted_bases = self.transformer_oracle.generate_from_string_with_permutations(poly_str, num_permutations=self.permutations)["all_permuted_bases"]
            # Flatten the list of lists
            all_permuted_bases = [" + ".join(basis) for basis in all_permuted_bases]
        
        # Convert the predicted basis string back to tokens/monomials
        if self.mode != "permutation_all":
            predicted_basis = self._parse_basis_string(predicted_basis_str)
        else:
            predicted_basis = [self._parse_basis_string(basis) for basis in all_permuted_bases]
        
        
        oracle_time = time.time() - start_time

        result = {
            'basis': predicted_basis,
            'time': oracle_time
        }

        # Optionally perform basis extension
        if self.use_basis_extension and self.mode != "permutation_all":
            ext_start = time.time()
            extended_basis = basis_extension(predicted_basis, poly, **self.basis_extension_params)
            ext_time = time.time() - ext_start
            result['basis'] = extended_basis
            result['basis_extension_time'] = ext_time
        elif self.use_basis_extension and self.mode == "permutation_all":
            ext_start = time.time()
            # Extend each basis individually
            extended_bases = [basis_extension(basis, poly, **self.basis_extension_params) for basis in predicted_basis]

            # Count frequency of each element across all extended base
            frequency_dict = Counter(chain.from_iterable(extended_bases))

            # Create a list of (monomial, frequency) tuples to sort by frequency
            # Convert each monomial to a string for hashing/sorting purposes
            monomial_freq_pairs = [(monomial, frequency_dict[tuple(monomial) if isinstance(monomial, list) else monomial]) 
                                 for monomial in frequency_dict.keys()]
            
            # Sort by frequency (descending) and extract just the monomials
            sorted_monomials = [monomial for monomial, freq in sorted(monomial_freq_pairs, key=lambda x: x[1], reverse=True)]

            # Take the union of all extended bases without sorting
            # union_extended_basis = list(set().union(*[set(b) for b in extended_bases]))

            ext_time = time.time() - ext_start

            # choose either union_extended_basis or sorted_monomials
            result['basis'] = sorted_monomials
            result['basis_extension_time'] = ext_time

        return result

    

if __name__ == "__main__":
    # Example usage
    model_path = "/home/htc/npelleriti/sum-of-squares-transformer/transformer-polynomial_old/notebooks/artifacts/model-expansion_sos-coefficients_m80000:v0"
    data_path = "/scratch/htc/npelleriti/data/sos-transformer/phase1/n4_sparse_uniform/test"
    num_variables = 4
    max_degree = 20
    max_coef = 10


    
    # Initialize oracle
    oracle = TransformerOracle(
        model_path=model_path,
        num_variables=num_variables,
        max_degree=max_degree,
        max_coef=max_coef,
        continuous_coefficient=True
    )

    # Load data
    test_dataset, data_collator = load_data(
        data_path=data_path,
        subprocessors={'monomial_ids': oracle.transformer_oracle.monomial_processor},
        splits=[{"name": "test", "batch_size": 32, "shuffle": False}],
        tokenizer=oracle.transformer_oracle.tokenizer,
        sample_size=None,  # or set a limit
        return_dataloader=False,
        data_collator_name='monomial',  # if you used monomial embedding
        aware_of_padding=False,
        data_format='polynomial_basis'  # or whatever you used
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        collate_fn=data_collator,
        shuffle=False
    )

    print(test_loader)
    """
    # Batch generation example
    print("=== Batch Generation Example ===")
    for i, batch in enumerate(test_loader):
        if i >= 2:  # Just show first 2 batches
            break
        break
            
        preds = oracle.generate_from_batch(batch, max_length=1024)
        
        # Decode targets for comparison
        labels = batch['labels']
        labels[labels == -100] = oracle.tokenizer.pad_token_id
        targets = oracle.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        print(f"Predictions: {preds}")
        print(f"Targets: {targets}")
    """

    # Single string generation example
    print("\n=== Single String Generation Example ===")

    polynomial_tokens = ["C5.7", "E0", "E0", "E0", "E0", "+", "C3.6", "E5", "E2", "E2", "E1", "+", "C-0.2", "E3", "E3", "E4", "E0", "+", "C-0.1", "E4", "E3", "E1", "E2", "+", "C-1.1", "E2", "E5", "E0", "E3", "+", "C-0.9", "E7", "E0", "E1", "E2", "+", "C1.8", "E6", "E0", "E0", "E1", "+", "C1.5", "E0", "E2", "E1", "E4", "+", "C5.9", "E1", "E2", "E4", "E2", "+", "C-0.2", "E3", "E4", "E0", "E0", "+", "C4.1", "E0", "E4", "E4", "E0", "+", "C1.0", "E3", "E0", "E3", "E1", "+", "C-2.6", "E2", "E6", "E0", "E1", "+", "C1.9", "E2", "E6", "E0", "E2", "+", "C-2.4", "E1", "E5", "E0", "E4", "+", "C-2.2", "E2", "E4", "E2", "E1", "+", "C-3.4", "E1", "E4", "E2", "E1", "+", "C3.0", "E6", "E0", "E0", "E3", "+", "C-2.1", "E0", "E1", "E4", "E0", "+", "C-1.3", "E3", "E2", "E1", "E3", "+", "C-2.2", "E3", "E1", "E0", "E0", "+", "C-0.9", "E4", "E1", "E2", "E0", "+", "C0.5", "E0", "E0", "E3", "E4", "+", "C-3.0", "E2", "E3", "E4", "E1", "+", "C-2.4", "E0", "E6", "E2", "E1", "+", "C5.7", "E10", "E4", "E4", "E2", "+", "C4.1", "E8", "E5", "E6", "E1", "+", "C2.3", "E9", "E5", "E3", "E3", "+", "C0.6", "E7", "E7", "E2", "E4", "+", "C-1.6", "E12", "E2", "E3", "E3", "+", "C3.6", "E11", "E2", "E2", "E2", "+", "C-0.1", "E5", "E4", "E3", "E5", "+", "C0.6", "E6", "E4", "E6", "E3", "+", "C1.6", "E8", "E6", "E2", "E1", "+", "C6.5", "E5", "E6", "E6", "E1", "+", "C1.6", "E8", "E2", "E5", "E2", "+", "C0.8", "E7", "E8", "E2", "E2", "+", "C-1.2", "E7", "E8", "E2", "E3", "+", "C-1.8", "E6", "E7", "E2", "E5", "+", "C-5.1", "E7", "E6", "E4", "E2", "+", "C-2.6", "E6", "E6", "E4", "E2", "+", "C3.3", "E11", "E2", "E2", "E4", "+", "C-2.5", "E5", "E3", "E6", "E1", "+", "C-1.7", "E8", "E4", "E3", "E4", "+", "C-0.8", "E8", "E3", "E2", "E1", "+", "C1.8", "E9", "E3", "E4", "E1", "+", "C-1.7", "E5", "E2", "E5", "E5", "+", "C-3.0", "E7", "E5", "E6", "E2", "+", "C0.6", "E5", "E8", "E4", "E2", "+", "C5.9", "E6", "E6", "E8", "E0", "+", "C-1.3", "E7", "E6", "E5", "E2", "+", "C-5.0", "E5", "E8", "E4", "E3", "+", "C-3.1", "E10", "E3", "E5", "E2", "+", "C1.9", "E3", "E5", "E5", "E4", "+", "C-2.2", "E4", "E5", "E8", "E2", "+", "C2.5", "E6", "E7", "E4", "E0", "+", "C-0.5", "E3", "E7", "E8", "E0", "+", "C0.2", "E6", "E3", "E7", "E1", "+", "C2.4", "E5", "E9", "E4", "E1", "+", "C-1.3", "E5", "E9", "E4", "E2", "+", "C-0.5", "E4", "E8", "E4", "E4", "+", "C-6.8", "E5", "E7", "E6", "E1", "+", "C2.1", "E4", "E7", "E6", "E1", "+", "C3.4", "E9", "E3", "E4", "E3", "+", "C-3.7", "E3", "E4", "E8", "E0", "+", "C1.6", "E6", "E5", "E5", "E3", "+", "C1.6", "E6", "E4", "E4", "E0", "+", "C2.9", "E7", "E4", "E6", "E0", "+", "C-1.1", "E3", "E3", "E7", "E4", "+", "C-1.3", "E5", "E6", "E8", "E1", "+", "C3.4", "E3", "E9", "E6", "E1", "+", "C3.5", "E8", "E6", "E2", "E4", "+", "C-0.8", "E6", "E8", "E1", "E5", "+", "C-0.1", "E11", "E3", "E2", "E4", "+", "C2.4", "E10", "E3", "E1", "E3", "+", "C-3.5", "E4", "E5", "E2", "E6", "+", "C-1.7", "E5", "E5", "E5", "E4", "+", "C-1.4", "E7", "E7", "E1", "E2", "+", "C0.2", "E4", "E7", "E5", "E2", "+", "C2.1", "E7", "E3", "E4", "E3", "+", "C0.5", "E6", "E9", "E1", "E3", "+", "C-2.5", "E6", "E9", "E1", "E4", "+", "C1.1", "E5", "E8", "E1", "E6", "+", "C-0.3", "E6", "E7", "E3", "E3", "+", "C-2.9", "E5", "E7", "E3", "E3", "+", "C2.5", "E10", "E3", "E1", "E5", "+", "C-0.8", "E4", "E4", "E5", "E2", "+", "C-2.4", "E7", "E5", "E2", "E5", "+", "C2.7", "E7", "E4", "E1", "E2", "+", "C-0.4", "E8", "E4", "E3", "E2", "+", "C0.6", "E4", "E3", "E4", "E6", "+", "C-3.9", "E6", "E6", "E5", "E3", "+", "C-0.4", "E4", "E9", "E3", "E3", "+", "C6.4", "E4", "E10", "E0", "E6", "+", "C-1.0", "E9", "E5", "E1", "E5", "+", "C-0.6", "E8", "E5", "E0", "E4", "+", "C-3.8", "E2", "E7", "E1", "E7", "+", "C3.0", "E3", "E7", "E4", "E5", "+", "C-3.8", "E5", "E9", "E0", "E3", "+", "C2.8", "E2", "E9", "E4", "E3", "+", "C2.7", "E5", "E5", "E3", "E4", "+", "C0.4", "E4", "E11", "E0", "E4", "+", "C-0.6", "E4", "E11", "E0", "E5", "+", "C-0.4", "E3", "E10", "E0", "E7", "+", "C3.8", "E4", "E9", "E2", "E4", "+", "C1.0", "E3", "E9", "E2", "E4", "+", "C-5.3", "E8", "E5", "E0", "E6", "+", "C-3.8", "E2", "E6", "E4", "E3", "+", "C1.8", "E5", "E7", "E1", "E6", "+", "C-4.0", "E5", "E6", "E0", "E3", "+", "C-0.2", "E6", "E6", "E2", "E3", "+", "C-2.4", "E2", "E5", "E3", "E7", "+", "C-3.8", "E2", "E11", "E2", "E4", "+", "C6.2", "E14", "E0", "E2", "E4", "+", "C-2.9", "E13", "E0", "E1", "E3", "+", "C1.7", "E7", "E2", "E2", "E6", "+", "C0.2", "E8", "E2", "E5", "E4", "+", "C1.6", "E10", "E4", "E1", "E2", "+", "C0.7", "E7", "E4", "E5", "E2", "+", "C-3.0", "E10", "E0", "E4", "E3", "+", "C-0.2", "E9", "E6", "E1", "E3", "+", "C-0.7", "E9", "E6", "E1", "E4", "+", "C-0.4", "E8", "E5", "E1", "E6", "+", "C3.1", "E9", "E4", "E3", "E3", "+", "C0.8", "E8", "E4", "E3", "E3", "+", "C3.6", "E13", "E0", "E1", "E5", "+", "C5.8", "E7", "E1", "E5", "E2", "+", "C1.6", "E10", "E2", "E2", "E5", "+", "C2.9", "E10", "E1", "E1", "E2", "+", "C-2.0", "E11", "E1", "E3", "E2", "+", "C-3.0", "E7", "E0", "E4", "E6", "+", "C-1.9", "E9", "E3", "E5", "E3", "+", "C-4.1", "E7", "E6", "E3", "E3", "+", "C4.5", "E12", "E0", "E0", "E2", "+", "C-1.7", "E6", "E2", "E1", "E5", "+", "C0.9", "E7", "E2", "E4", "E3", "+", "C3.1", "E9", "E4", "E0", "E1", "+", "C-2.0", "E6", "E4", "E4", "E1", "+", "C2.6", "E9", "E0", "E3", "E2", "+", "C1.5", "E8", "E6", "E0", "E2", "+", "C-0.2", "E8", "E6", "E0", "E3", "+", "C0.7", "E7", "E5", "E0", "E5", "+", "C-1.0", "E8", "E4", "E2", "E2", "+", "C-4.5", "E7", "E4", "E2", "E2", "+", "C-3.4", "E12", "E0", "E0", "E4", "+", "C1.6", "E6", "E1", "E4", "E1", "+", "C-1.5", "E9", "E2", "E1", "E4", "+", "C-0.5", "E9", "E1", "E0", "E1", "+", "C-2.1", "E10", "E1", "E2", "E1", "+", "C-1.4", "E6", "E0", "E3", "E5", "+", "C-3.5", "E8", "E3", "E4", "E2", "+", "C2.3", "E6", "E6", "E2", "E2", "+", "C5.3", "E0", "E4", "E2", "E8", "+", "C-0.4", "E1", "E4", "E5", "E6", "+", "C-1.2", "E3", "E6", "E1", "E4", "+", "C1.6", "E0", "E6", "E5", "E4", "+", "C-4.4", "E3", "E2", "E4", "E5", "+", "C-0.8", "E2", "E8", "E1", "E5", "+", "C2.2", "E2", "E8", "E1", "E6", "+", "C-0.6", "E1", "E7", "E1", "E8", "+", "C0.4", "E2", "E6", "E3", "E5", "+", "C-1.9", "E1", "E6", "E3", "E5", "+", "C0.5", "E6", "E2", "E1", "E7", "+", "C0.2", "E0", "E3", "E5", "E4", "+", "C3.2", "E3", "E4", "E2", "E7", "+", "C-0.2", "E3", "E3", "E1", "E4", "+", "C-3.2", "E4", "E3", "E3", "E4", "+", "C0.6", "E0", "E2", "E4", "E8", "+", "C1.6", "E2", "E5", "E5", "E5", "+", "C-0.1", "E0", "E8", "E3", "E5", "+", "C9.5", "E2", "E4", "E8", "E4", "+", "C-0.3", "E4", "E6", "E4", "E2", "+", "C5.4", "E1", "E6", "E8", "E2", "+", "C1.9", "E4", "E2", "E7", "E3", "+", "C-1.2", "E3", "E8", "E4", "E3", "+", "C2.9", "E3", "E8", "E4", "E4", "+", "C-0.8", "E2", "E7", "E4", "E6", "+", "C2.1", "E3", "E6", "E6", "E3", "+", "C-2.0", "E2", "E6", "E6", "E3", "+", "C6.0", "E7", "E2", "E4", "E5", "+", "C-2.2", "E1", "E3", "E8", "E2", "+", "C0.5", "E4", "E4", "E5", "E5", "+", "C-4.0", "E4", "E3", "E4", "E2", "+", "C-4.0", "E5", "E3", "E6", "E2", "+", "C0.9", "E1", "E2", "E7", "E6", "+", "C-1.1", "E3", "E5", "E8", "E3", "+", "C-1.0", "E1", "E8", "E6", "E3", "+", "C5.4", "E6", "E8", "E0", "E0", "+", "C-2.7", "E3", "E8", "E4", "E0", "+", "C3.2", "E6", "E4", "E3", "E1", "+", "C-0.4", "E5", "E10", "E0", "E1", "+", "C1.0", "E5", "E10", "E0", "E2", "+", "C-0.7", "E4", "E9", "E0", "E4", "+", "C-2.4", "E5", "E8", "E2", "E1", "+", "C1.1", "E4", "E8", "E2", "E1", "+", "C-0.8", "E9", "E4", "E0", "E3", "+", "C1.9", "E3", "E5", "E4", "E0", "+", "C-4.2", "E6", "E6", "E1", "E3", "+", "C-0.0", "E6", "E5", "E0", "E0", "+", "C0.3", "E7", "E5", "E2", "E0", "+", "C-3.5", "E3", "E4", "E3", "E4", "+", "C-1.8", "E5", "E7", "E4", "E1", "+", "C3.8", "E3", "E10", "E2", "E1", "+", "C9.7", "E0", "E8", "E8", "E0", "+", "C-1.5", "E3", "E4", "E7", "E1", "+", "C0.9", "E2", "E10", "E4", "E1", "+", "C5.7", "E2", "E10", "E4", "E2", "+", "C0.8", "E1", "E9", "E4", "E4", "+", "C2.1", "E2", "E8", "E6", "E1", "+", "C3.2", "E1", "E8", "E6", "E1", "+", "C5.5", "E6", "E4", "E4", "E3", "+", "C0.1", "E0", "E5", "E8", "E0", "+", "C5.8", "E3", "E6", "E5", "E3", "+", "C-3.4", "E4", "E5", "E6", "E0", "+", "C-2.4", "E0", "E4", "E7", "E4", "+", "C-0.1", "E2", "E7", "E8", "E1", "+", "C-1.6", "E0", "E10", "E6", "E1", "+", "C4.6", "E6", "E0", "E6", "E2", "+", "C-0.3", "E5", "E6", "E3", "E2", "+", "C-0.6", "E5", "E6", "E3", "E3", "+", "C-0.8", "E4", "E5", "E3", "E5", "+", "C-0.3", "E5", "E4", "E5", "E2", "+", "C-0.7", "E9", "E0", "E3", "E4", "+", "C-4.2", "E3", "E1", "E7", "E1", "+", "C-3.6", "E6", "E2", "E4", "E4", "+", "C-0.2", "E6", "E1", "E3", "E1", "+", "C-0.0", "E7", "E1", "E5", "E1", "+", "C-1.3", "E3", "E0", "E6", "E5", "+", "C-1.8", "E5", "E3", "E7", "E2", "+", "C1.7", "E3", "E6", "E5", "E2", "+", "C3.9", "E4", "E12", "E0", "E2", "+", "C-1.5", "E4", "E12", "E0", "E3", "+", "C0.9", "E3", "E11", "E0", "E5", "+", "C0.7", "E4", "E10", "E2", "E2", "+", "C1.7", "E3", "E10", "E2", "E2", "+", "C-0.5", "E8", "E6", "E0", "E4", "+", "C-0.2", "E2", "E7", "E4", "E1", "+", "C1.4", "E5", "E8", "E1", "E4", "+", "C-0.6", "E5", "E7", "E0", "E1", "+", "C0.3", "E6", "E7", "E2", "E1", "+", "C-0.6", "E4", "E9", "E4", "E2", "+", "C1.1", "E2", "E12", "E2", "E2", "+", "C3.4", "E4", "E12", "E0", "E4", "+", "C-1.9", "E3", "E11", "E0", "E6", "+", "C1.0", "E4", "E10", "E2", "E3", "+", "C0.9", "E3", "E10", "E2", "E3", "+", "C-0.0", "E8", "E6", "E0", "E5", "+", "C2.4", "E2", "E7", "E4", "E2", "+", "C4.0", "E5", "E8", "E1", "E5", "+", "C-1.0", "E5", "E7", "E0", "E2", "+", "C0.9", "E6", "E7", "E2", "E2", "+", "C-2.2", "E2", "E6", "E3", "E6", "+", "C2.5", "E4", "E9", "E4", "E3", "+", "C-2.0", "E2", "E12", "E2", "E3", "+", "C4.2", "E2", "E10", "E0", "E8", "+", "C3.3", "E3", "E9", "E2", "E5", "+", "C0.4", "E2", "E9", "E2", "E5", "+", "C-0.7", "E7", "E5", "E0", "E7", "+", "C2.8", "E1", "E6", "E4", "E4", "+", "C-2.0", "E4", "E7", "E1", "E7", "+", "C0.0", "E4", "E6", "E0", "E4", "+", "C-2.1", "E5", "E6", "E2", "E4", "+", "C4.2", "E1", "E5", "E3", "E8", "+", "C-1.1", "E3", "E8", "E4", "E5", "+", "C2.5", "E1", "E11", "E2", "E5", "+", "C8.1", "E4", "E8", "E4", "E2", "+", "C-0.6", "E3", "E8", "E4", "E2", "+", "C-1.6", "E8", "E4", "E2", "E4", "+", "C3.7", "E2", "E5", "E6", "E1", "+", "C5.1", "E5", "E6", "E3", "E4", "+", "E5", "E5", "E2", "E1", "+", "C-0.3", "E6", "E5", "E4", "E1", "+", "C-0.6", "E2", "E4", "E5", "E5", "+", "C-2.2", "E4", "E7", "E6", "E2", "+", "C6.0", "E2", "E8", "E4", "E2", "+", "C0.0", "E7", "E4", "E2", "E4", "+", "C0.1", "E1", "E5", "E6", "E1", "+", "C1.7", "E4", "E6", "E3", "E4", "+", "C-2.5", "E4", "E5", "E2", "E1", "+", "C3.6", "E5", "E5", "E4", "E1", "+", "C-2.7", "E1", "E4", "E5", "E5", "+", "C2.4", "E3", "E7", "E6", "E2", "+", "C1.6", "E1", "E10", "E4", "E2", "+", "C8.1", "E12", "E0", "E0", "E6", "+", "C-1.5", "E6", "E1", "E4", "E3", "+", "C-0.2", "E9", "E2", "E1", "E6", "+", "C2.2", "E9", "E1", "E0", "E3", "+", "C1.0", "E10", "E1", "E2", "E3", "+", "C3.4", "E6", "E0", "E3", "E7", "+", "C-3.5", "E8", "E3", "E4", "E4", "+", "C-0.5", "E6", "E6", "E2", "E4", "+", "C6.0", "E0", "E2", "E8", "E0", "+", "C0.8", "E3", "E3", "E5", "E3", "+", "C2.7", "E3", "E2", "E4", "E0", "+", "C-0.1", "E4", "E2", "E6", "E0", "+", "C-2.2", "E0", "E1", "E7", "E4", "+", "C-0.6", "E2", "E4", "E8", "E1", "+", "C-0.4", "E0", "E7", "E6", "E1", "+", "C8.0", "E6", "E4", "E2", "E6", "+", "C1.4", "E6", "E3", "E1", "E3", "+", "C-1.9", "E7", "E3", "E3", "E3", "+", "C-2.0", "E3", "E2", "E4", "E7", "+", "C-0.1", "E3", "E8", "E3", "E4", "+", "C4.3", "E6", "E2", "E0", "E0", "+", "C-1.5", "E7", "E2", "E2", "E0", "+", "C-1.1", "E3", "E1", "E3", "E4", "+", "C-1.9", "E5", "E4", "E4", "E1", "+", "C-1.6", "E3", "E7", "E2", "E1", "+", "C6.5", "E8", "E2", "E4", "E0", "+", "C-1.5", "E4", "E1", "E5", "E4", "+", "C0.6", "E6", "E4", "E6", "E1", "+", "C0.9", "E4", "E7", "E4", "E1", "+", "C5.5", "E0", "E0", "E6", "E8", "+", "C-1.3", "E2", "E3", "E7", "E5", "+", "C2.3", "E0", "E6", "E5", "E5", "+", "C5.2", "E4", "E6", "E8", "E2", "+", "C-5.1", "E2", "E9", "E6", "E2", "+", "C8.1", "E0", "E12", "E4", "E2"]
    # Concatenate the list and replace everything that starts with 'C' by '[C]', keeping a space between tokens
    example_tokenized_poly = " ".join(['[C]' if token.startswith('C') else token for token in polynomial_tokens])

    output = oracle.generate_from_string(example_tokenized_poly)
    print(f"Input: {example_tokenized_poly}")
    print(f"Generated: {output}")