import json
import numpy as np
import time
import wandb
import argparse
from tqdm import tqdm
import os
import sys
import re
from pathlib import Path

# Add the sos/src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation.monomials.monomials import Monomial, Polynomial
from sdp_solver.cvxpy_solver import CVXPYSOSSolver
from utils.polynomial import get_newton_polytope_basis, parse_monomial_str, parse_polynomial_str, verify_basis_reconstruction
from utils.oracles import TransformerOracle, NewtonOracle, OriginalOracle, DegreeOracle
from utils.basis_extension import basis_extension

def load_solver_config(config_file_path, solver_name, config_name="default"):
    """
    Load solver configuration from JSONL file.
    
    Args:
        config_file_path (str): Path to the solver configuration JSONL file
        solver_name (str): Name of the solver (e.g., "mosek")
        config_name (str): Configuration name (e.g., "default", "high_precision")
    
    Returns:
        dict: Solver parameters or None if not found
    """
    try:
        with open(config_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    config = json.loads(line)
                    if (config.get('solver', '').lower() == solver_name.lower() and 
                        config.get('config', '') == config_name):
                        return config.get('params', {})
        print(f"Warning: Configuration for solver '{solver_name}' with config '{config_name}' not found in {config_file_path}")
        return None
    except FileNotFoundError:
        print(f"Warning: Solver configuration file not found: {config_file_path}")
        return None
    except Exception as e:
        print(f"Warning: Error loading solver configuration: {e}")
        return None

def extract_params_from_path(model_path, input_path):
    """
    Extract num_variables, max_degree, and generate corresponding model path and run name.
    
    Expected format: /path/to/n{num_variables}_{basis_sampler}_{matrix_sampler}_d{max_degree}_m{num_monomials}/test.jsonl
    Example: /scratch/htc/npelleriti/data/sos-transformer/star/n4_sparse_uniform_simple_random_d5_m20/test.jsonl
    
    Returns:
        tuple: (num_variables, max_degree, num_monomials, model_path, run_name)
    """
    # Get the directory name from the path
    path_obj = Path(input_path)
    dir_name = path_obj.parent.name
    
    # Extract using regex pattern: n{num_variables}_...d{max_degree}_m{num_monomials}
    pattern = r'n(\d+)_.*_d(\d+)_m(\d+)'
    match = re.search(pattern, dir_name)
    
    if match:
        num_variables = int(match.group(1))
        max_degree = int(match.group(2))
        num_monomials = int(match.group(3))
        
        # Generate model path based on the directory name
        model_path = f"{model_path}/{dir_name}"
        
        # Generate run name based on the directory name
        run_name = f"cascading-oracle-{dir_name}"
        
        return num_variables, max_degree, num_monomials, model_path, run_name
    else:
        raise ValueError(f"Could not extract parameters from path: {input_path}")

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

def process_example_with_cascading_oracles(data, solver, solver_options, transformer_oracle_single, transformer_oracle_permutation, newton_oracle, expansion_factor, min_expansion_size, verbose=False):
    """
    Process a single example with the cascading oracle approach:
    1. Try transformer oracle with mode='single', permutations=1, with repair (basis extension)
    2. If SDP fails, get ordered list from transformer oracle with permutation mode
    3. Iteratively expand basis size using the ordered list (factor rho > 1 or constant increment)
    4. If ordered list is exhausted without success, fall back to Newton oracle
    """
    try:
        poly = Polynomial.from_sequence(data["polynomial_tokens"])
        basis_tokens = data["basis_tokens"]
        original_basis = list(Polynomial.from_sequence(basis_tokens).terms.keys())
        
        results = {
            'success': False,
            'final_oracle_used': None,
            'num_expansion_iterations': 0,
            'total_oracle_time': 0,
            'total_sdp_time': 0,
            'total_basis_extension_time': 0,
            'predicted_basis_size': 0,
            'num_terms': len(poly.terms),
            'max_degree': max(sum(m.exponents) for m in poly.terms.keys()),
            'error': None,
            'original_basis_size': len(original_basis),
            'false_positives': 0,
            'false_negatives': 0,
            'stage1_attempted': False,
            'stage2_attempted': False, 
            'stage3_attempted': False,
            'stage4_attempted': False,
            'ordered_basis_size': 0,
            'filtered_ordered_basis_size': 0,
            'max_basis_subset_tried': 0,
            'stage1_sdp_time': 0,
            'stage3_sdp_times': [],
            'stage4_sdp_time': 0,
        }
        
        oracle_kwargs = {'poly_tokens': data["polynomial_tokens"], 'poly': poly}
        
        # Stage 1: Transformer oracle with single mode and basis extension (repair)
        if verbose:
            print(f"Stage 1: Trying transformer oracle (single mode) with basis extension...")
        results['stage1_attempted'] = True
        
        stage1_start = time.time()
        oracle_result = transformer_oracle_single.predict_basis(**oracle_kwargs)
        predicted_basis = oracle_result['basis']
        stage1_oracle_time = oracle_result['time']
        stage1_basis_extension_time = oracle_result.get('basis_extension_time', 0)
        stage1_total_time = time.time() - stage1_start
        
        results['total_oracle_time'] += stage1_oracle_time
        results['total_basis_extension_time'] += stage1_basis_extension_time
        results['predicted_basis_size'] = len(predicted_basis)
        
        # Calculate false positives and false negatives for stage 1
        original_basis_set = set(original_basis)
        predicted_basis_set = set(predicted_basis)
        false_positives = predicted_basis_set - original_basis_set
        false_negatives = original_basis_set - predicted_basis_set
        results['false_positives'] = len(false_positives)
        results['false_negatives'] = len(false_negatives)
        
        # Try SDP solve with stage 1 basis
        if verbose:
            print(f"Stage 1: Trying SDP solve with basis of size {len(predicted_basis)}...")
        sdp_start = time.time()
        is_sos, Q = solver.solve_sos_feasibility(poly, basis=predicted_basis, solver_options=solver_options)
        sdp_time = time.time() - sdp_start
        results['total_sdp_time'] += sdp_time
        results['stage1_sdp_time'] = sdp_time
        
        if is_sos:
            if verbose:
                print(f"Stage 1: Success! SDP solved with transformer single mode + basis extension")
            results['success'] = True
            results['final_oracle_used'] = 'transformer_single_with_repair'
            return results
        
        # Stage 2: Get ordered basis from transformer oracle with permutation mode
        if verbose:
            print(f"Stage 1 failed. Stage 2: Getting ordered basis from transformer oracle with permutation mode...")
        results['stage2_attempted'] = True
        
        stage2_start = time.time()
        oracle_result = transformer_oracle_permutation.predict_basis(**oracle_kwargs)
        ordered_basis = oracle_result['basis']
        stage2_oracle_time = oracle_result['time'] 
        stage2_basis_extension_time = oracle_result.get('basis_extension_time', 0)
        
        results['total_oracle_time'] += stage2_oracle_time
        results['total_basis_extension_time'] += stage2_basis_extension_time
        results['ordered_basis_size'] = len(ordered_basis)
        
        if verbose:
            print(f"Stage 2: Obtained ordered basis of size {len(ordered_basis)}")
        
        # Get Newton basis for intersection and potential fallback
        newton_result = newton_oracle.predict_basis(**oracle_kwargs)
        newton_basis = newton_result['basis']
        newton_basis_set = set(newton_basis)
        
        # Stage 3: Intersect ordered basis with Newton polytope, preserving order
        ordered_basis = [m for m in ordered_basis if m in newton_basis_set]
        results['filtered_ordered_basis_size'] = len(ordered_basis)
        if verbose:
            print(f"Stage 3: Filtered ordered basis with Newton polytope, size: {len(ordered_basis)}")
        
        # Stage 3: Iteratively expand basis size using filtered ordered list
        if verbose:
            print(f"Stage 3: Iteratively expanding basis size using filtered ordered list...")
        results['stage3_attempted'] = True
        
        # Start with initial basis size (either factor of stage 1 size or minimum size)
        stage1_basis_size = len(predicted_basis)
        current_basis_size = max(int(stage1_basis_size * expansion_factor), min_expansion_size)
        current_basis_size = min(current_basis_size, len(ordered_basis))  # Don't exceed available basis
        
        expansion_iter = 0
        while current_basis_size <= len(ordered_basis):
            expansion_iter += 1
            if verbose:
                print(f"Stage 3: Expansion iteration {expansion_iter}, trying basis size {current_basis_size}/{len(ordered_basis)}")
            
            # Take first current_basis_size elements from ordered basis
            current_basis = ordered_basis[:current_basis_size]
            results['predicted_basis_size'] = len(current_basis)
            results['max_basis_subset_tried'] = current_basis_size
            results['num_expansion_iterations'] = expansion_iter
            
            # Update false positives/negatives
            predicted_basis_set = set(current_basis)
            false_positives = predicted_basis_set - original_basis_set
            false_negatives = original_basis_set - predicted_basis_set
            results['false_positives'] = len(false_positives)
            results['false_negatives'] = len(false_negatives)
            
            # Try SDP solve with current basis subset
            if verbose:
                print(f"Stage 3: Trying SDP solve with basis subset of size {len(current_basis)}...")
            sdp_start = time.time()
            is_sos, Q = solver.solve_sos_feasibility(poly, basis=current_basis, solver_options=solver_options)
            sdp_time = time.time() - sdp_start
            results['total_sdp_time'] += sdp_time
            results['stage3_sdp_times'].append(sdp_time)
            
            if is_sos:
                if verbose:
                    print(f"Stage 3: Success! SDP solved with basis subset of size {len(current_basis)} after {expansion_iter} iterations")
                results['success'] = True
                results['final_oracle_used'] = 'transformer_ordered_expansion'
                return results
            
            # Expand basis size for next iteration
            if expansion_factor > 1:
                # Use multiplicative factor
                current_basis_size = min(int(current_basis_size * expansion_factor), len(ordered_basis))
            else:
                # Use additive increment
                current_basis_size = min(current_basis_size + min_expansion_size, len(ordered_basis))
            
            # If we've reached the maximum size, try the full basis once
            if current_basis_size >= len(ordered_basis) and len(current_basis) < len(ordered_basis):
                current_basis_size = len(ordered_basis)
            elif current_basis_size >= len(ordered_basis):
                # We've tried the full basis, exit loop
                break
        
        if verbose:
            print(f"Stage 3: Exhausted ordered basis without finding SOS certificate")
        
        # Stage 4: Newton oracle fallback (using previously computed Newton basis)
        if verbose:
            print(f"Stage 3 failed. Stage 4: Using Newton oracle basis...")
        results['stage4_attempted'] = True
        
        # Newton basis was already computed for Stage 3 intersection
        newton_oracle_time = newton_result['time']
        results['total_oracle_time'] += newton_oracle_time
        results['predicted_basis_size'] = len(newton_basis)
        
        # Update false positives/negatives for Newton
        predicted_basis_set = set(newton_basis)
        false_positives = predicted_basis_set - original_basis_set
        false_negatives = original_basis_set - predicted_basis_set
        results['false_positives'] = len(false_positives)
        results['false_negatives'] = len(false_negatives)
        
        # Add Newton-specific metrics
        if 'vertex_bound' in newton_result:
            results['vertex_bound'] = newton_result['vertex_bound']
        if 'combinatorial_bound' in newton_result:
            results['combinatorial_bound'] = newton_result['combinatorial_bound']
        
        # Try SDP solve with Newton basis
        if verbose:
            print(f"Stage 4: Trying SDP solve with Newton basis of size {len(newton_basis)}...")
        sdp_start = time.time()
        is_sos, Q = solver.solve_sos_feasibility(poly, basis=newton_basis, solver_options=solver_options)
        sdp_time = time.time() - sdp_start
        results['total_sdp_time'] += sdp_time
        results['stage4_sdp_time'] = sdp_time
        
        if is_sos:
            if verbose:
                print(f"Stage 4: Success! SDP solved with Newton oracle")
            results['success'] = True
            results['final_oracle_used'] = 'newton'
        else:
            if verbose:
                print(f"Stage 4: Failed. All cascading oracle approaches unsuccessful")
            results['success'] = False
            results['final_oracle_used'] = 'none'
            
        return results
        
    except Exception as e:
        print(f"Error processing example: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'final_oracle_used': 'error',
            'stage1_attempted': False,
            'stage2_attempted': False,
            'stage3_attempted': False,
            'stage4_attempted': False,
        }

def main():
    parser = argparse.ArgumentParser(description='Run SDP experiments with cascading oracle approach')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--use_basis_extension', type=lambda x: x.lower() in ['true', '1', 'yes', 'on'], default=True, help='Whether to use basis extension in transformer single mode (repair)')
    parser.add_argument('--max_examples', type=int, default=100, help='Only process the first N examples from the dataset')
    parser.add_argument('--basis_extension_max_iter', type=int, default=10, help='Max iterations for basis extension in transformer single mode')
    parser.add_argument('--expansion_factor', type=float, default=1.5, help='Factor (rho > 1) to expand basis size in stage 3, or 1.0 to use constant increment')
    parser.add_argument('--min_expansion_size', type=int, default=10, help='Minimum expansion size (constant increment) or minimum initial size')
    parser.add_argument('--model_path', type=str, required=True, help='Base path to the model directory')
    parser.add_argument('--max_coef', type=int, default=None, help='Max coefficient of the polynomial')
    parser.add_argument('--permutations', type=int, default=2, help='Number of permutations to use for transformer permutation oracle')
    parser.add_argument('--permutation_mode', type=str, default='permutation_all', choices=['permutation_union', 'permutation_all', 'permutation_intersection'], help='Permutation mode for stage 2')
    parser.add_argument('--solver', type=str, default='SCS', choices=['SCS', 'MOSEK', 'CLARABEL', 'SDPA'], help='Solver to use')
    parser.add_argument('--solver_config', type=str, default='default', help='Solver configuration name to load from config file (e.g., default, high_precision)')
    parser.add_argument('--ood', type=lambda x: x.lower() in ['true', '1', 'yes', 'on'], default=False, help='Whether to use out-of-distribution model')
    parser.add_argument('--verbose', type=lambda x: x.lower() in ['true', '1', 'yes', 'on'], default=False, help='Enable verbose output printing')
    
    args = parser.parse_args()
    
    # Extract parameters, model path, and run name from the input path
    try:
        num_variables, max_degree, num_monomials, model_path, run_name = extract_params_from_path(args.model_path, args.input_path)
        print(f"Extracted from path: num_variables={num_variables}, max_degree={max_degree}, num_monomials={num_monomials}")
        print(f"Generated model_path: {model_path}")
        print(f"Generated run_name: {run_name}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Use model of the given path
    if args.ood:
        model_path = args.model_path

    # Initialize wandb with configuration
    wandb.init(
        project="sos-transformer", 
        name=run_name,
        config={
            'input_path': args.input_path,
            'cascading_approach': True,
            'max_examples': args.max_examples,
            'use_basis_extension': args.use_basis_extension,
            'basis_extension_max_iter': args.basis_extension_max_iter,
            'expansion_factor': args.expansion_factor,
            'min_expansion_size': args.min_expansion_size,
            'model_path': model_path,
            'num_variables': num_variables,
            'max_degree': max_degree,
            'num_monomials': num_monomials,
            'max_coef': args.max_coef,
            'permutations': args.permutations,
            'permutation_mode': args.permutation_mode,
            'solver': args.solver,
            'solver_config': args.solver_config,
            'ood': args.ood,
            'verbose': args.verbose
        }
    )

    # Initialize solver
    if args.solver == 'MOSEK':
        solver = CVXPYSOSSolver(solver='MOSEK')
        print(f"Using MOSEK solver with config: {args.solver_config}")
        
        # Load MOSEK configuration from file
        config_file_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'solvers', 'solver_settings.jsonl')
        mosek_params = load_solver_config(config_file_path, 'mosek', args.solver_config)
        
        if mosek_params is not None:
            solver_options = {'mosek_params': mosek_params}
            print(f"Loaded MOSEK parameters from config file: {list(mosek_params.keys())}")
        else:
            # Fallback to default hardcoded parameters
            print("Warning: Using fallback hardcoded MOSEK parameters")
            solver_options = {
                'mosek_params': {
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-3,
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-3,
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-3,
                    'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-3,
                    'MSK_DPAR_INTPNT_CO_TOL_INFEAS': 1e-3,
                    'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-3,
                    'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-3,
                    'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-3,
                    'MSK_DPAR_INTPNT_TOL_MU_RED': 1e-3,
                    'MSK_DPAR_INTPNT_TOL_INFEAS': 1e-3,
                    'MSK_IPAR_INTPNT_MAX_ITERATIONS': 10000,
                    'MSK_IPAR_INTPNT_SCALING': 1,
                    'MSK_IPAR_LOG_INTPNT': 1,
                }
            }
    elif args.solver == 'SCS': 
        solver = CVXPYSOSSolver(solver='SCS')
        print(f"Using SCS solver with config: {args.solver_config}")
        
        # Load SCS configuration from file
        config_file_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'solvers', 'solver_settings.jsonl')
        scs_params = load_solver_config(config_file_path, 'scs', args.solver_config)
        
        if scs_params is not None:
            solver_options = scs_params
            print(f"Loaded SCS parameters from config file: {list(scs_params.keys())}")
        else:
            # Fallback to default hardcoded parameters
            print("Warning: Using fallback hardcoded SCS parameters")
            solver_options = {
                'max_iters': 50000,
                'eps': 1e-8,
                'alpha': 1.5,
                'acceleration_lookback': 50,
                'scale': 5.0,
                'normalize': False,
                'use_indirect': False,
                'use_quad_obj': True,
            }
    elif args.solver == 'CLARABEL':
        solver = CVXPYSOSSolver(solver='CLARABEL')
        print(f"Using CLARABEL solver with config: {args.solver_config}")
        
        # Load CLARABEL configuration from file
        config_file_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'solvers', 'solver_settings.jsonl')
        clarabel_params = load_solver_config(config_file_path, 'clarabel', args.solver_config)
        
        if clarabel_params is not None:
            solver_options = clarabel_params
            print(f"Loaded CLARABEL parameters from config file: {list(clarabel_params.keys())}")
        else:
            # Fallback to default hardcoded parameters
            print("Warning: Using fallback hardcoded CLARABEL parameters")
            solver_options = {
                'max_iter': 10000,
                'tol_gap_abs': 1e-3,
                'tol_gap_rel': 1e-3,
                'tol_feas': 1e-3,
                'tol_infeas_abs': 1e-3,
                'tol_infeas_rel': 1e-3,
                'tol_ktratio': 1e-6,
                'equilibrate_enable': True,
                'presolve_enable': True,
                'chordal_decomposition_enable': True,
            }
    elif args.solver == 'SDPA':
        solver = CVXPYSOSSolver(solver='SDPA')
        print(f"Using SDPA solver with config: {args.solver_config}")
        
        # Load SDPA configuration from file
        config_file_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'solvers', 'solver_settings.jsonl')
        sdpa_params = load_solver_config(config_file_path, 'sdpa', args.solver_config)
        
        if sdpa_params is not None:
            solver_options = sdpa_params
            print(f"Loaded SDPA parameters from config file: {list(sdpa_params.keys())}")
        else:
            # Fallback to default hardcoded parameters
            print("Warning: Using fallback hardcoded SDPA parameters")
            solver_options = {
                'maxIteration': 10000,
                'epsilonStar': 1e-3,
                'epsilonDash': 1e-3,
            }
    else:
        raise ValueError(f"Unknown solver: {args.solver}")

    # Initialize oracles for different stages
    if args.verbose:
        print("Initializing cascading oracles...")
    
    # Stage 1: Transformer oracle with single mode and basis extension (repair)
    transformer_oracle_single = get_oracle(
        'transformer',
        use_basis_extension=args.use_basis_extension,
        basis_extension_params={'max_iter': args.basis_extension_max_iter},
        model_path=model_path,
        num_variables=num_variables,
        max_degree=max_degree,
        max_coef=args.max_coef,
        permutations=1,
        mode='single'
    )
    
    # Stage 2: Transformer oracle with permutation mode for larger basis
    transformer_oracle_permutation = get_oracle(
        'transformer',
        use_basis_extension=args.use_basis_extension,
        basis_extension_params={'max_iter': args.basis_extension_max_iter},
        model_path=model_path,
        num_variables=num_variables,
        max_degree=max_degree,
        max_coef=args.max_coef,
        permutations=args.permutations,
        mode=args.permutation_mode
    )
    
    # Stage 4: Newton oracle for fallback
    newton_oracle = get_oracle(
        'newton',
        use_basis_extension=False,  # Newton oracle doesn't support basis extension
        basis_extension_params={},
        num_variables=num_variables,
        max_degree=max_degree
    )
    
    if args.verbose:
        print("All oracles initialized successfully.")

    # Processing statistics
    total_examples = 0
    successful_examples = 0
    failed_examples = 0
    
    # Stage-wise statistics
    stage1_successes = 0
    stage3_successes = 0
    stage4_successes = 0

    with open(args.input_path, 'r') as f:
        for i, line in enumerate(tqdm(f, total=args.max_examples, desc="Processing examples")):
            if i >= args.max_examples:
                break
            try:
                data = json.loads(line)
                metrics = process_example_with_cascading_oracles(
                    data,
                    solver,
                    solver_options,
                    transformer_oracle_single,
                    transformer_oracle_permutation,
                    newton_oracle,
                    args.expansion_factor,
                    args.min_expansion_size,
                    verbose=args.verbose
                )
                
                total_examples += 1
                if metrics['success']:
                    successful_examples += 1
                    # Count stage-wise successes
                    if metrics['final_oracle_used'] == 'transformer_single_with_repair':
                        stage1_successes += 1
                    elif metrics['final_oracle_used'] == 'transformer_ordered_expansion':
                        stage3_successes += 1
                    elif metrics['final_oracle_used'] == 'newton':
                        stage4_successes += 1
                else:
                    failed_examples += 1
                    
                metrics['example_idx'] = i
                
                # Add summary statistics for stage 3 SDP times
                if metrics['stage3_sdp_times']:
                    metrics['stage3_avg_sdp_time'] = sum(metrics['stage3_sdp_times']) / len(metrics['stage3_sdp_times'])
                    metrics['stage3_min_sdp_time'] = min(metrics['stage3_sdp_times'])
                    metrics['stage3_max_sdp_time'] = max(metrics['stage3_sdp_times'])
                
                wandb.log(metrics)
                
            except Exception as e:
                print(f"Error processing example {i}: {str(e)}")
                wandb.log({
                    'example_idx': i,
                    'success': False,
                    'error': str(e),
                    'final_oracle_used': 'error'
                })
                failed_examples += 1
                continue

    # Final statistics
    final_stats = {
        'total_examples': total_examples,
        'successful_examples': successful_examples,
        'failed_examples': failed_examples,
        'success_rate': successful_examples / total_examples if total_examples > 0 else 0,
        'stage1_successes': stage1_successes,
        'stage3_successes': stage3_successes,
        'stage4_successes': stage4_successes,
        'stage1_success_rate': stage1_successes / total_examples if total_examples > 0 else 0,
        'stage3_success_rate': stage3_successes / total_examples if total_examples > 0 else 0,
        'stage4_success_rate': stage4_successes / total_examples if total_examples > 0 else 0,
    }
    wandb.log(final_stats)

    print("\nFinal Statistics:")
    print(f"Total examples processed: {total_examples}")
    print(f"Successful examples: {successful_examples}")
    print(f"Failed examples: {failed_examples}")
    print(f"Overall success rate: {final_stats['success_rate']:.2%}")
    print(f"\nStage-wise breakdown:")
    print(f"Stage 1 (Transformer single + repair): {stage1_successes} ({final_stats['stage1_success_rate']:.2%})")
    print(f"Stage 3 (Ordered basis expansion): {stage3_successes} ({final_stats['stage3_success_rate']:.2%})")
    print(f"Stage 4 (Newton fallback): {stage4_successes} ({final_stats['stage4_success_rate']:.2%})")

    wandb.finish()

if __name__ == "__main__":
    main()
