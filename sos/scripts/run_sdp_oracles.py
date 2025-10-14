import json
import numpy as np
import time
import wandb
import argparse
from tqdm import tqdm
import os
import sys

# Add the sos/src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation.monomials.monomials import Monomial, Polynomial
from sdp_solver.cvxpy_solver import CVXPYSOSSolver
from utils.polynomial import get_newton_polytope_basis, parse_monomial_str, parse_polynomial_str, verify_basis_reconstruction
from utils.oracles import TransformerOracle, NewtonOracle, OriginalOracle, DegreeOracle

def get_oracle(oracle_type, use_basis_extension, basis_extension_params, model_path = None, num_variables = None, max_degree = None, max_coef = None, permutations = 1, mode = 'single'):
    if oracle_type == 'transformer':
        return TransformerOracle(use_basis_extension=use_basis_extension, basis_extension_params=basis_extension_params, model_path=model_path, num_variables=num_variables, max_degree=max_degree, max_coef=max_coef, permutations=permutations, mode=mode)
    elif oracle_type == 'newton':
        return NewtonOracle(use_basis_extension=use_basis_extension, basis_extension_params=basis_extension_params)
    elif oracle_type == 'original':
        return OriginalOracle(use_basis_extension=use_basis_extension, basis_extension_params=basis_extension_params)
    elif oracle_type == 'degree':
        return DegreeOracle(max_degree=max_degree, num_variables=num_variables, use_basis_extension=use_basis_extension, basis_extension_params=basis_extension_params)
    else:
        raise ValueError(f"Unknown oracle type: {oracle_type}")

def process_example_with_oracle(data, solver, solver_options, oracle, oracle_type):
    try:
        poly = Polynomial.from_sequence(data["polynomial_tokens"])
        basis_tokens = data["basis_tokens"]
        original_basis = list(Polynomial.from_sequence(basis_tokens).terms.keys())

         # Oracle call
        oracle_kwargs = {'poly_tokens': data["polynomial_tokens"], 'poly': poly}
        if oracle_type == 'original':
            oracle_kwargs['original_basis'] = original_basis
        oracle_result = oracle.predict_basis(**oracle_kwargs)
        predicted_basis = oracle_result['basis']
        oracle_time = oracle_result['time']
        basis_extension_time = oracle_result.get('basis_extension_time', None)

        # Calculate false positives and false negatives
        original_basis_set = set(original_basis)
        predicted_basis_set = set(predicted_basis)
        
        # False positives: terms in predicted basis but not in original basis
        false_positives = predicted_basis_set - original_basis_set
        false_negatives = original_basis_set - predicted_basis_set
        
        num_false_positives = len(false_positives)
        num_false_negatives = len(false_negatives)

        # check if original basis is a subset of predicted basis
        if not set(original_basis).issubset(set(predicted_basis)):
            missing = set(original_basis) - set(predicted_basis)
            print(f"Original basis is not a subset of predicted basis")
            print(f"Missing from predicted basis: {missing}")
            print(f"Predicted basis: {predicted_basis}")
            print(f"Original basis: {original_basis}")
            #raise ValueError("Original basis is not a subset of predicted basis")
        else:
            pass#print(f"Original basis is a subset of predicted basis")

        # SDP solve
        t0 = time.time()
        is_sos, Q = solver.solve_sos_feasibility(poly, basis=predicted_basis, solver_options=solver_options)
        sdp_time = time.time() - t0

        results = {
            'success': is_sos,
            'oracle_time': oracle_time,
            'sdp_time': sdp_time,
            'basis_extension_time': basis_extension_time,
            'predicted_basis_size': len(predicted_basis),
            'num_terms': len(poly.terms),
            'max_degree': max(sum(m.exponents) for m in poly.terms.keys()),
            'error': None,
            'original_basis_size': len(original_basis),
            'false_positives': num_false_positives,
            'false_negatives': num_false_negatives,
        }
        return results
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Run SDP experiments with oracle-supported basis prediction')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--run_name', type=str, default='sdp-oracle', help='Name for the wandb run')
    parser.add_argument('--oracle', type=str, choices=['transformer', 'newton', 'original', 'degree'], default='transformer', help='Which oracle to use')
    parser.add_argument('--use_basis_extension', action='store_true', help='Whether to use basis extension after oracle prediction')
    parser.add_argument('--max_examples', type=int, default=100, help='Only process the first N examples from the dataset')
    parser.add_argument('--basis_extension_max_iter', type=int, default=10, help='Max iterations for basis extension')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model file')
    parser.add_argument('--num_variables', type=int, default=None, help='Number of variables in the polynomial')
    parser.add_argument('--max_degree', type=int, default=None, help='Max degree of the polynomial')
    parser.add_argument('--max_coef', type=int, default=None, help='Max coefficient of the polynomial')
    parser.add_argument('--permutations', type=int, default=1, help='Number of permutations to use for the oracle (only relevant for transformer oracle)')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'permutation_union', 'permutation_all', 'permutation_intersection'], help='Oracle prediction mode')
    args = parser.parse_args()

    wandb.init(project="sos-transformer", name=args.run_name)

    solver = CVXPYSOSSolver(solver='SCS', verbose=False)
    print("Using SCS solver")

    solver_options = {
        'max_iters': 50000, # Increased from default 2500
        'eps': 1e-8, # Significantly tighter than default 1e-4
        'alpha': 1.5, # Default
        'acceleration_lookback': 50, # Default
        'scale': 5.0, # Default
        'normalize': False, # Default
        'use_indirect': False, # Default
        'use_quad_obj': True, # Default
    }

    oracle = get_oracle(
        args.oracle,
        use_basis_extension=args.use_basis_extension,
        basis_extension_params={'max_iter': args.basis_extension_max_iter},
        model_path=args.model_path,
        num_variables=args.num_variables,
        max_degree=args.max_degree,
        max_coef=args.max_coef,
        permutations=args.permutations,
        mode=args.mode
    )
    print(f"Oracle initialized successfully.")

    total_examples = 0
    successful_examples = 0
    failed_examples = 0

    with open(args.input_path, 'r') as f:
        for i, line in enumerate(tqdm(f, total=args.max_examples, desc="Processing examples")):
            if i >= args.max_examples:
                break
            try:
                data = json.loads(line)
                metrics = process_example_with_oracle(
                    data,
                    solver,
                    solver_options,
                    oracle,
                    args.oracle
                )
                total_examples += 1
                if metrics['success']:
                    successful_examples += 1
                else:
                    failed_examples += 1
                metrics['example_idx'] = i
                wandb.log(metrics)
            except Exception as e:
                print(f"Error processing example {i}: {str(e)}")
                wandb.log({
                    'example_idx': i,
                    'success': False,
                    'error': str(e)
                })
                failed_examples += 1
                continue

    final_stats = {
        'total_examples': total_examples,
        'successful_examples': successful_examples,
        'failed_examples': failed_examples,
        'success_rate': successful_examples / total_examples if total_examples > 0 else 0
    }
    wandb.log(final_stats)

    print("\nFinal Statistics:")
    print(f"Total examples processed: {total_examples}")
    print(f"Successful examples: {successful_examples}")
    print(f"Failed examples: {failed_examples}")
    print(f"Success rate: {final_stats['success_rate']:.2%}")

    wandb.finish()

if __name__ == "__main__":
    main() 