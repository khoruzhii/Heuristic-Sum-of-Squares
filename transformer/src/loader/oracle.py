from typing import List, Tuple, Optional
import torch
from torch.utils.data import DataLoader
import random
import numpy as np

import os
import sys
from time import time  

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

print("Current working directory:", os.getcwd())


from src.loader.checkpoint import load_pretrained_bag
from src.loader.data_format.processors.subprocessors import MonomialProcessorPlus


from src.misc.utils import to_cuda
from src.evaluation.generation import generation
from src.loader.data import load_data


# Add the transformer/src directory to Python path
sos_src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'sos', 'src')
sys.path.insert(0, sos_src_path)
from utils.polynomial import permute_polynomial_tokens, permute_polynomial_object
from data_generation.monomials.monomials import Polynomial, Monomial
from utils.basis_extension import basis_extension




"""


model_path = "/home/htc/npelleriti/transformer-polynomial/training_pipeline/notebooks/artifacts/model-expansion_sos-coefficients_m80000:v0"

bag = load_pretrained_bag(model_path)
model = bag['model']
tokenizer = bag['tokenizer']
model_name = bag['model_name']

from src.loader.data_format.processors.subprocessors import MonomialProcessorPlus

monomial_processor = MonomialProcessorPlus(
    num_variables=4,  # set this to match your model
    max_degree=20,        # set this to match your model
    max_coef=10,     # or int(FIELD[2:]) if using GF
    rational_coefficients=None,   # or your rational_coefficients if used
    continuous_coefficient=True
)

from src.loader.data import load_data

data_path = "/scratch/htc/npelleriti/data/sos-transformer/phase1/n4_sparse_uniform/test"

test_dataset, data_collator = load_data(
    data_path=data_path,
    subprocessors={'monomial_ids': monomial_processor},
    splits=[{"name": "test", "batch_size": 32, "shuffle": False}],
    tokenizer=tokenizer,
    sample_size=None,  # or set a limit
    return_dataloader=False,
    data_collator_name='monomial',  # if you used monomial embedding
    aware_of_padding=False,
    data_format='polynomial_basis'  # or whatever you used
)



from torch.utils.data import DataLoader

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    collate_fn=data_collator,
    shuffle=False
)



from src.evaluation.generation import generation_accuracy

results = generation_accuracy(
    model,
    test_loader,
    max_length=2048,  # match your training
    tokenizer=tokenizer,
    monomial_processor=monomial_processor,
    compute_support_acc=True,  # or False if you don't need it
    th=0.01,  # for continuous coefficients
    model_name=model_name
)
print(results)

from src.misc.utils import to_cuda
from src.evaluation.generation import generation

from time import time  

num_samples = 3

for i, batch in enumerate(test_loader):
        print(len(batch['labels']))
        batch = to_cuda(batch)

        max_length = min(1024, batch['labels'].shape[1])
        
        start = time()
        preds = generation(model, model_name, batch, tokenizer, monomial_processor=monomial_processor, max_length=max_length)
        end = time()
        runtime = end - start
        
        
        labels = batch['labels']
        labels[labels == -100] = tokenizer.pad_token_id
        targets = tokenizer.batch_decode(labels, skip_special_tokens=True) 

        print(len(preds))

        print(preds)
        print(targets)

        if i >= num_samples:
            break
            
"""

class TransformerOracle:
    def __init__(
        self,
        model_path: str,
        num_variables: int,
        max_degree: int,
        max_coef: int,
        continuous_coefficient: bool = True,
        rational_coefficients=None,
        data_format: str = "polynomial_basis",
        batch_size: int = 32,
        device: str = "cuda"
    ):
        # Load model, tokenizer, model_name
        bag = load_pretrained_bag(model_path)
        self.model = bag['model'].to(device)
        self.tokenizer = bag['tokenizer']
        self.model_name = bag['model_name']
        self.device = device

        # Monomial processor
        self.monomial_processor = MonomialProcessorPlus(
            num_variables=num_variables,
            max_degree=max_degree,
            max_coef=max_coef,
            rational_coefficients=rational_coefficients,
            continuous_coefficient=continuous_coefficient
        )
        self.data_format = data_format
        self.batch_size = batch_size

        self.num_variables = num_variables

    def generate_from_batch(self, batch, max_length: int = 2048):
        """Generate predictions from a batch of data"""
        from src.misc.utils import to_cuda
        
        batch = to_cuda(batch)
        max_length = min(max_length, batch['labels'].shape[1])

        
        preds = generation(
            self.model, 
            self.model_name, 
            batch, 
            self.tokenizer, 
            monomial_processor=self.monomial_processor, 
            max_length=max_length
        )
        return preds

    def generate_from_string(self, tokenized_poly: str, max_length: int = 2048):
        """Generate prediction from a single tokenized polynomial string"""
        # Use monomial processor to tokenize instead of standard tokenizer
        if self.monomial_processor is not None:
            # Process using monomial processor (returns list of ProcessedMonomial)
            processed = self.monomial_processor([tokenized_poly])
            
            # Convert to tensor format
            batch_size = 1
            seq_length = len(processed[0])
            num_tokens_per_unit = len(processed[0][0].tokens) if hasattr(processed[0][0], 'tokens') else len(processed[0][0])
            
            # Create tensors
            input_ids = torch.zeros(batch_size, seq_length, num_tokens_per_unit, dtype=torch.long, device=self.device)
            coefficient_values = torch.zeros(batch_size, seq_length, dtype=torch.float, device=self.device)
            
            # Fill tensors
            for i, monomial in enumerate(processed[0]):
                if hasattr(monomial, 'tokens'):  # ProcessedMonomial
                    input_ids[0, i, :] = torch.tensor(monomial.tokens, device=self.device)
                    coefficient_values[0, i] = monomial.coefficient_value if monomial.coefficient_value is not None else 0.0
                else:  # Regular list
                    input_ids[0, i, :] = torch.tensor(monomial, device=self.device)
            
            # Create attention mask
            attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=self.device)
            
            # Create a batch with single item
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            # Add coefficient values if using continuous coefficients
            if self.monomial_processor.continuous_coefficient:
                batch['coefficient_values'] = coefficient_values
            
            # Use the same generation function as batch processing
            preds = generation(
                self.model,
                self.model_name,
                batch,
                self.tokenizer,
                monomial_processor=self.monomial_processor,
                max_length=max_length
            )
            return preds[0] if isinstance(preds, list) else preds
        else:
            # Fallback to standard tokenizer (if no monomial processor)
            inputs = self.tokenizer(
                tokenized_poly,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            print("inputs", inputs)
            # Generate
            with torch.no_grad():
                preds = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_beams=1,
                    tokenizer=self.tokenizer,
                    monomial_processor=self.monomial_processor,
                    do_sample=False
                )
            # Decode
            if self.monomial_processor is not None:
                decoded = self.monomial_processor.batch_decode(preds, skip_special_tokens=True)
            else:
                decoded = self.tokenizer.batch_decode(preds.long().cpu().numpy(), skip_special_tokens=True)
            return decoded[0] if isinstance(decoded, list) else decoded

    def generate_from_string_with_permutations(self, tokenized_poly: str, max_length: int = 2048, num_permutations: int = 5):
        """
        Generate prediction using multiple permutation-based approaches.
        Args:
            tokenized_poly (str): Tokenized polynomial string
            max_length (int): Maximum generation length
            num_permutations (int): Number of random permutations to try (default 5)
        Returns:
            dict: Contains 'union_basis', 'intersection_basis', 'original_basis', 'all_permuted_bases', 'permutations', 'inverse_permutations'
        """
        tokens = tokenized_poly.split()
        original_output = self.generate_from_string(tokenized_poly, max_length)
        original_basis = self._extract_basis_from_output(original_output)
        all_permuted_bases = []
        permutations = []
        inverse_permutations = []
        for _ in range(num_permutations):
            permutation = list(range(self.num_variables))
            random.shuffle(permutation)
            inverse_permutation = [0] * self.num_variables
            for i, pos in enumerate(permutation):
                inverse_permutation[pos] = i
            poly = Polynomial.from_sequence(tokens)
            permuted_poly = permute_polynomial_object(poly, permutation, num_vars=self.num_variables)
            permuted_string = " ".join(permuted_poly.to_sequence(num_vars=self.num_variables, sort_polynomials=True))
            #print(f"Permuted string: {_}: {permuted_string}")
            permuted_output = self.generate_from_string(permuted_string, max_length)
            permuted_poly_out = Polynomial.from_sequence(permuted_output.split())
            unpermuted_poly = permute_polynomial_object(permuted_poly_out, inverse_permutation, num_vars=self.num_variables)
            unpermuted_output = " ".join(unpermuted_poly.to_sequence(num_vars=self.num_variables))
            unpermuted_basis = self._extract_basis_from_output(unpermuted_output)
            all_permuted_bases.append(unpermuted_basis)
            permutations.append(permutation)
            inverse_permutations.append(inverse_permutation)
        # Union and intersection across all bases (including original)
        all_bases = [set(original_basis)] + [set(b) for b in all_permuted_bases]
        union_basis = list(set().union(*all_bases))
        intersection_basis = list(set.intersection(*all_bases)) if all_bases else []
        return {
            'union_basis': union_basis,
            'intersection_basis': intersection_basis,
            'original_basis': original_basis,
            'all_permuted_bases': all_permuted_bases,
            'permutations': permutations,
            'inverse_permutations': inverse_permutations
        }
    
    def _extract_basis_from_output(self, output_string: str) -> List[str]:
        """
        Extract basis monomials from the model output string.
        This is a simplified extraction - adjust based on your actual output format.
        
        Args:
            output_string (str): Model output string
            
        Returns:
            List[str]: List of basis monomial strings
        """
        # Split by common separators and extract monomials
        # This is a basic implementation - you may need to adjust based on your output format
        tokens = output_string.split()
        basis = []
        
        # Simple extraction: look for patterns that look like monomials
        # Adjust this based on your actual output format
        current_monomial = []
        for token in tokens:
            if token.startswith('C') or token.startswith('E'):
                current_monomial.append(token)
            elif token == '+':
                if current_monomial:
                    basis.append(" ".join(current_monomial))
                    current_monomial = []
            else:
                # Handle other tokens as needed
                pass
        
        # Don't forget the last monomial
        if current_monomial:
            basis.append(" ".join(current_monomial))
        
        return basis

if __name__ == "__main__":
    # Example usage
    model_path = "/scratch/htc/npelleriti/models/sos-transformer_sparse_uniform_n4"
    data_path = "/scratch/htc/npelleriti/data/sos-transformer/phase1/n4_clique/train"
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
        subprocessors={'monomial_ids': oracle.monomial_processor},
        splits=[{"name": "test", "batch_size": 32, "shuffle": False}],
        tokenizer=oracle.tokenizer,
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

    # Batch generation example
    print("=== Batch Generation Example ===")
    for i, batch in enumerate(test_loader):
        if i >= 2:  # Just show first 2 batches
            break
            
        print(f"\nBatch {i}:")
        print(batch.keys())
        print(batch['input_ids'][0])
        break
        preds = oracle.generate_from_batch(batch, max_length=1024)
        
        # Decode targets for comparison
        labels = batch['labels']
        labels[labels == -100] = oracle.tokenizer.pad_token_id
        targets = oracle.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        print(f"Predictions: {preds}")
        print(f"Targets: {targets}")

    # Single string generation example
    print("\n=== Single String Generation Example ===")

    polynomial_tokens = ["C5.1", "E4", "E0", "E6", "E0", "+", "C-0.3", "E2", "E1", "E3", "E0", "+", "C-3.0", "E2", "E0", "E3", "E0", "+", "C2.0", "E2", "E2", "E5", "E0", "+", "C-2.6", "E2", "E1", "E6", "E0", "+", "C2.6", "E5", "E1", "E3", "E0", "+", "C0.1", "E2", "E0", "E6", "E0", "+", "C-1.9", "E6", "E1", "E3", "E0", "+", "C-2.5", "E3", "E0", "E3", "E2", "+", "C0.4", "E3", "E4", "E3", "E0", "+", "C-3.5", "E6", "E0", "E3", "E0", "+", "C0.7", "E2", "E4", "E3", "E0", "+", "C1.7", "E5", "E0", "E3", "E1", "+", "C-1.1", "E2", "E2", "E3", "E0", "+", "C-0.0", "E2", "E0", "E8", "E0", "+", "C-0.7", "E4", "E0", "E4", "E0", "+", "C5.8", "E2", "E3", "E3", "E0", "+", "C1.3", "E2", "E5", "E3", "E0", "+", "C2.6", "E3", "E0", "E3", "E0", "+", "C-1.3", "E2", "E1", "E7", "E0", "+", "C-3.5", "E2", "E3", "E5", "E0", "+", "C2.2", "E3", "E3", "E4", "E0", "+", "C3.0", "E2", "E0", "E4", "E0", "+", "C4.4", "E0", "E2", "E0", "E0", "+", "C1.5", "E0", "E1", "E0", "E0", "+", "C2.9", "E0", "E3", "E2", "E0", "+", "C-1.9", "E0", "E2", "E3", "E0", "+", "C-2.1", "E3", "E2", "E0", "E0", "+", "C-2.1", "E0", "E1", "E3", "E0", "+", "C-1.3", "E4", "E2", "E0", "E0", "+", "C2.7", "E1", "E1", "E0", "E2", "+", "C5.0", "E1", "E5", "E0", "E0", "+", "C2.9", "E4", "E1", "E0", "E0", "+", "C-0.6", "E0", "E5", "E0", "E0", "+", "C2.5", "E3", "E1", "E0", "E1", "+", "C-4.5", "E0", "E3", "E0", "E0", "+", "C1.9", "E0", "E1", "E5", "E0", "+", "C1.0", "E2", "E1", "E1", "E0", "+", "C0.7", "E0", "E4", "E0", "E0", "+", "C9.1", "E0", "E6", "E0", "E0", "+", "C-1.7", "E1", "E1", "E0", "E0", "+", "C4.3", "E0", "E2", "E4", "E0", "+", "C-1.2", "E0", "E4", "E2", "E0", "+", "C6.2", "E1", "E4", "E1", "E0", "+", "C1.9", "E0", "E1", "E1", "E0", "+", "C4.0", "E0", "E0", "E0", "E0", "+", "C2.6", "E0", "E2", "E2", "E0", "+", "C-1.9", "E3", "E1", "E0", "E0", "+", "C-1.7", "E0", "E0", "E3", "E0", "+", "C0.9", "E1", "E0", "E0", "E2", "+", "C1.3", "E1", "E4", "E0", "E0", "+", "C5.6", "E4", "E0", "E0", "E0", "+", "C-1.3", "E3", "E0", "E0", "E1", "+", "C1.8", "E0", "E0", "E5", "E0", "+", "C-0.2", "E2", "E0", "E1", "E0", "+", "C2.2", "E1", "E0", "E0", "E0", "+", "C-0.2", "E0", "E1", "E4", "E0", "+", "C-0.7", "E1", "E3", "E1", "E0", "+", "C-0.4", "E0", "E0", "E1", "E0", "+", "C4.8", "E0", "E4", "E4", "E0", "+", "C6.1", "E0", "E3", "E5", "E0", "+", "C-1.5", "E3", "E3", "E2", "E0", "+", "C-3.7", "E0", "E2", "E5", "E0", "+", "C-1.4", "E4", "E3", "E2", "E0", "+", "C1.8", "E1", "E2", "E2", "E2", "+", "C5.2", "E1", "E6", "E2", "E0", "+", "C3.2", "E4", "E2", "E2", "E0", "+", "C2.4", "E0", "E6", "E2", "E0", "+", "C0.6", "E3", "E2", "E2", "E1", "+", "C1.0", "E0", "E2", "E7", "E0", "+", "C-0.7", "E0", "E5", "E2", "E0", "+", "C6.2", "E0", "E7", "E2", "E0", "+", "C3.1", "E1", "E2", "E2", "E0", "+", "C2.2", "E0", "E3", "E6", "E0", "+", "C0.8", "E0", "E5", "E4", "E0", "+", "C-7.1", "E1", "E5", "E3", "E0", "+", "C4.9", "E0", "E2", "E6", "E0", "+", "C-1.6", "E3", "E2", "E3", "E0", "+", "C-1.9", "E0", "E1", "E6", "E0", "+", "C-2.9", "E4", "E2", "E3", "E0", "+", "C-2.8", "E1", "E1", "E3", "E2", "+", "C3.3", "E4", "E1", "E3", "E0", "+", "C1.4", "E0", "E5", "E3", "E0", "+", "C-1.3", "E3", "E1", "E3", "E1", "+", "C-2.2", "E0", "E3", "E3", "E0", "+", "C1.1", "E0", "E1", "E8", "E0", "+", "C0.6", "E2", "E1", "E4", "E0", "+", "C-0.9", "E0", "E4", "E3", "E0", "+", "C-4.1", "E0", "E6", "E3", "E0", "+", "C0.1", "E1", "E1", "E3", "E0", "+", "C-2.3", "E0", "E4", "E5", "E0", "+", "C-3.9", "E1", "E4", "E4", "E0", "+", "C5.8", "E6", "E2", "E0", "E0", "+", "C1.1", "E3", "E1", "E3", "E0", "+", "C-0.1", "E7", "E2", "E0", "E0", "+", "C-1.2", "E4", "E1", "E0", "E2", "+", "C-3.4", "E4", "E5", "E0", "E0", "+", "C0.8", "E7", "E1", "E0", "E0", "+", "C-2.0", "E3", "E5", "E0", "E0", "+", "C-3.3", "E6", "E1", "E0", "E1", "+", "C0.1", "E3", "E3", "E0", "E0", "+", "C1.4", "E3", "E1", "E5", "E0", "+", "C-1.1", "E5", "E1", "E1", "E0", "+", "C3.3", "E3", "E4", "E0", "E0", "+", "C-4.0", "E3", "E6", "E0", "E0", "+", "C-2.2", "E3", "E2", "E4", "E0", "+", "C2.9", "E3", "E4", "E2", "E0", "+", "C-0.5", "E4", "E4", "E1", "E0", "+", "C-0.4", "E3", "E1", "E1", "E0", "+", "C4.4", "E0", "E0", "E6", "E0", "+", "C2.2", "E1", "E0", "E3", "E2", "+", "C1.9", "E1", "E4", "E3", "E0", "+", "C-0.6", "E4", "E0", "E3", "E0", "+", "C-1.2", "E3", "E0", "E3", "E1", "+", "C-2.7", "E0", "E0", "E8", "E0", "+", "C-1.0", "E1", "E0", "E3", "E0", "+", "C-1.0", "E0", "E1", "E7", "E0", "+", "C0.6", "E1", "E3", "E4", "E0", "+", "C-0.8", "E0", "E0", "E4", "E0", "+", "C5.8", "E8", "E2", "E0", "E0", "+", "C2.3", "E5", "E1", "E0", "E2", "+", "C1.6", "E5", "E5", "E0", "E0", "+", "C1.0", "E8", "E1", "E0", "E0", "+", "C0.8", "E7", "E1", "E0", "E1", "+", "C-0.9", "E4", "E3", "E0", "E0", "+", "C1.4", "E4", "E1", "E5", "E0", "+", "C2.9", "E6", "E1", "E1", "E0", "+", "C-3.1", "E4", "E4", "E0", "E0", "+", "C2.8", "E4", "E6", "E0", "E0", "+", "C-0.2", "E5", "E1", "E0", "E0", "+", "C-2.1", "E4", "E2", "E4", "E0", "+", "C7.9", "E4", "E4", "E2", "E0", "+", "C1.8", "E5", "E4", "E1", "E0", "+", "C-3.5", "E4", "E1", "E1", "E0", "+", "C3.6", "E2", "E0", "E0", "E4", "+", "C1.1", "E2", "E4", "E0", "E2", "+", "C0.5", "E5", "E0", "E0", "E2", "+", "C2.2", "E1", "E4", "E0", "E2", "+", "C-0.6", "E4", "E0", "E0", "E3", "+", "C-1.5", "E1", "E2", "E0", "E2", "+", "C-0.6", "E1", "E0", "E5", "E2", "+", "C-0.2", "E3", "E0", "E1", "E2", "+", "C-1.4", "E1", "E3", "E0", "E2", "+", "C5.1", "E1", "E5", "E0", "E2", "+", "C-0.2", "E2", "E0", "E0", "E2", "+", "C1.4", "E1", "E1", "E4", "E2", "+", "C3.4", "E1", "E3", "E2", "E2", "+", "C-1.0", "E2", "E3", "E1", "E2", "+", "C-1.8", "E1", "E0", "E1", "E2", "+", "C8.8", "E2", "E8", "E0", "E0", "+", "C4.8", "E5", "E4", "E0", "E0", "+", "C3.3", "E1", "E8", "E0", "E0", "+", "C1.9", "E4", "E4", "E0", "E1", "+", "C-0.5", "E1", "E6", "E0", "E0", "+", "C-2.4", "E1", "E4", "E5", "E0", "+", "C-2.3", "E3", "E4", "E1", "E0", "+", "C3.6", "E1", "E7", "E0", "E0", "+", "C4.7", "E1", "E9", "E0", "E0", "+", "C3.3", "E2", "E4", "E0", "E0", "+", "C-3.0", "E1", "E5", "E4", "E0", "+", "C0.7", "E1", "E7", "E2", "E0", "+", "C-2.0", "E2", "E7", "E1", "E0", "+", "C6.3", "E8", "E0", "E0", "E0", "+", "C-2.4", "E7", "E0", "E0", "E1", "+", "C0.7", "E4", "E0", "E5", "E0", "+", "C-0.1", "E6", "E0", "E1", "E0", "+", "C0.7", "E5", "E0", "E0", "E0", "+", "C-2.4", "E4", "E1", "E4", "E0", "+", "C-1.5", "E5", "E3", "E1", "E0", "+", "C2.7", "E4", "E0", "E1", "E0", "+", "C8.8", "E0", "E8", "E0", "E0", "+", "C4.2", "E3", "E4", "E0", "E1", "+", "C2.9", "E2", "E4", "E1", "E0", "+", "C2.8", "E0", "E7", "E0", "E0", "+", "C5.3", "E0", "E9", "E0", "E0", "+", "C-0.5", "E1", "E7", "E1", "E0", "+", "C0.2", "E0", "E4", "E1", "E0", "+", "C4.4", "E6", "E0", "E0", "E2", "+", "C-1.9", "E3", "E2", "E0", "E1", "+", "C0.2", "E3", "E0", "E5", "E1", "+", "C0.2", "E5", "E0", "E1", "E1", "+", "C3.2", "E3", "E3", "E0", "E1", "+", "C3.8", "E3", "E5", "E0", "E1", "+", "C0.6", "E4", "E0", "E0", "E1", "+", "C-0.9", "E3", "E1", "E4", "E1", "+", "C-2.1", "E3", "E3", "E2", "E1", "+", "C5.8", "E4", "E3", "E1", "E1", "+", "C-0.8", "E3", "E0", "E1", "E1", "+", "C1.6", "E2", "E2", "E1", "E0", "+", "C0.6", "E1", "E2", "E0", "E0", "+", "C-1.6", "E0", "E3", "E4", "E0", "+", "C-1.5", "E1", "E5", "E1", "E0", "+", "C0.0", "E0", "E2", "E1", "E0", "+", "C3.8", "E0", "E0", "E10", "E0", "+", "C-3.4", "E0", "E5", "E5", "E0", "+", "C-1.1", "E1", "E0", "E5", "E0", "+", "C-2.6", "E0", "E1", "E9", "E0", "+", "C-1.2", "E0", "E3", "E7", "E0", "+", "C-0.6", "E1", "E3", "E6", "E0", "+", "C3.3", "E4", "E0", "E2", "E0", "+", "C0.4", "E2", "E3", "E1", "E0", "+", "C-0.1", "E2", "E5", "E1", "E0", "+", "C-0.3", "E3", "E0", "E1", "E0", "+", "C0.3", "E2", "E1", "E5", "E0", "+", "C-0.0", "E2", "E0", "E2", "E0", "+", "C3.3", "E1", "E3", "E0", "E0", "+", "C1.0", "E1", "E6", "E1", "E0", "+", "C0.1", "E0", "E3", "E1", "E0", "+", "C6.0", "E0", "E10", "E0", "E0", "+", "C12.2", "E0", "E6", "E4", "E0", "+", "C2.0", "E0", "E8", "E2", "E0", "+", "C0.1", "E1", "E8", "E1", "E0", "+", "C-2.0", "E0", "E5", "E1", "E0", "+", "C7.9", "E2", "E0", "E0", "E0", "+", "C-1.1", "E1", "E1", "E4", "E0", "+", "C-1.6", "E1", "E3", "E2", "E0", "+", "C-3.6", "E1", "E0", "E1", "E0", "+", "C4.9", "E0", "E2", "E8", "E0", "+", "C0.6", "E0", "E4", "E6", "E0", "+", "C0.1", "E1", "E6", "E3", "E0", "+", "C6.2", "E2", "E6", "E2", "E0", "+", "C4.0", "E0", "E0", "E2", "E0"]
    # Concatenate the list and replace everything that starts with 'C' by '[C]', keeping a space between tokens


    example_tokenized_poly = " ".join(polynomial_tokens)

    actual_basis = ["C1.0", "E2", "E0", "E3", "E0", "+", "C1.0", "E0", "E1", "E0", "E0", "+", "C1.0", "E0", "E0", "E0", "E0", "+", "C1.0", "E0", "E2", "E2", "E0", "+", "C1.0", "E0", "E1", "E3", "E0", "+", "C1.0", "E3", "E1", "E0", "E0", "+", "C1.0", "E0", "E0", "E3", "E0", "+", "C1.0", "E4", "E1", "E0", "E0", "+", "C1.0", "E1", "E0", "E0", "E2", "+", "C1.0", "E1", "E4", "E0", "E0", "+", "C1.0", "E4", "E0", "E0", "E0", "+", "C1.0", "E0", "E4", "E0", "E0", "+", "C1.0", "E3", "E0", "E0", "E1", "+", "C1.0", "E0", "E2", "E0", "E0", "+", "C1.0", "E0", "E0", "E5", "E0", "+", "C1.0", "E2", "E0", "E1", "E0", "+", "C1.0", "E0", "E3", "E0", "E0", "+", "C1.0", "E0", "E5", "E0", "E0", "+", "C1.0", "E1", "E0", "E0", "E0", "+", "C1.0", "E0", "E1", "E4", "E0", "+", "C1.0", "E0", "E3", "E2", "E0", "+", "C1.0", "E1", "E3", "E1", "E0", "+", "C1.0", "E0", "E0", "E1", "E0"]

    actual_basis = oracle._extract_basis_from_output(" ".join(actual_basis))

    output = oracle.generate_from_string(example_tokenized_poly)
    print(f"Input: {example_tokenized_poly}")
    print(f"Generated: {output}")

    # Permutation-based generation example
    print("\n=== Permutation-based Generation Example ===")
    permutation_result = oracle.generate_from_string_with_permutations(example_tokenized_poly, num_permutations=5)
    print(f"Original basis: {permutation_result['original_basis']}")
    print(f"All permuted bases: {permutation_result['all_permuted_bases']}")
    print(f"Union basis: {permutation_result['union_basis']}")
    print(f"Intersection basis: {permutation_result['intersection_basis']}")
    print(f"Permutations used: {permutation_result['permutations']}")
    print(f"Inverse permutations: {permutation_result['inverse_permutations']}")


    # Compute set differences between actual_basis and permutation results
    # Convert all basis lists to sets of tuples for comparison (tuplify for hashability)
    def basis_to_set(basis):
        # If basis is a list of Polynomial.Term or similar, convert to tuple or string
        # Here, assume each term is a tuple or can be converted to tuple
        return set(tuple(term) if not isinstance(term, str) else term for term in basis)

    actual_basis_set = basis_to_set(actual_basis)
    original_basis_set = basis_to_set(permutation_result['original_basis'])
    permuted_basis_set = basis_to_set(permutation_result['all_permuted_bases'][2])
    union_basis_set = basis_to_set(permutation_result['union_basis'])

    diff_actual_vs_original = actual_basis_set - original_basis_set
    diff_actual_vs_permuted = actual_basis_set - permuted_basis_set
    diff_actual_vs_union = actual_basis_set - union_basis_set
    print(f"Count (actual_basis - original_basis): {len(diff_actual_vs_original)}")
    print(f"Count (actual_basis - permuted_basis): {len(diff_actual_vs_permuted)}")
    print(f"Count (actual_basis - union_basis): {len(diff_actual_vs_union)}")

    # compute basis extensions
    print([Monomial.from_sequence(term.split())[0] for term in permutation_result['all_permuted_bases'][2]], Polynomial.from_sequence(polynomial_tokens))
    basis_extended = basis_extension([Monomial.from_sequence(term.split())[0] for term in permutation_result['all_permuted_bases'][2]], Polynomial.from_sequence(polynomial_tokens))
    print(f"Basis extension: {basis_extended}")
    basis_extended_set = basis_to_set([" ".join(term.to_sequence(4)) for term in basis_extended])
    print(basis_extended_set)
    diff_actual_vs_basis_extended = actual_basis_set - basis_extended_set
    print(f"Count (actual_basis - basis_extended): {len(diff_actual_vs_basis_extended)}")
