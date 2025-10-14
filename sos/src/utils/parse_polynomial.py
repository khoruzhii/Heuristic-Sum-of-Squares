"""
Script to parse a polynomial string and extract exponent vectors.
"""

import re
import json
from typing import List, Tuple, Dict

def parse_term(term: str) -> Tuple[List[int], float]:
    """Parse a single term of the form 'coeff*x1^a*x2^b...' into exponents and coefficient."""
    # Initialize exponents for 5 variables
    exponents = [0, 0, 0, 0, 0]
    
    # Handle negative terms
    term = term.strip()
    if term.startswith('-'):
        term = term[1:]
        sign = -1
    else:
        sign = 1
    
    # Extract coefficient
    parts = term.split('*')
    try:
        coeff = float(parts[0]) * sign
    except ValueError:
        if parts[0] == "1":
            coeff = 1.0 * sign
        else:
            raise ValueError(f"Invalid coefficient: {parts[0]}")
    
    # If term is just a constant, return
    if len(parts) == 1:
        return exponents, coeff
    
    # Extract variable parts (e.g., "x1^2", "x4", etc.)
    var_parts = parts[1:]
    
    for part in var_parts:
        if '^' in part:
            # Handle terms with explicit power (e.g., "x1^2")
            var, power = part.split('^')
            var_idx = int(var[1]) - 1  # x1 -> index 0
            exponents[var_idx] = int(power)
        else:
            # Handle terms with implicit power 1 (e.g., "x1")
            var_idx = int(part[1]) - 1
            exponents[var_idx] = 1
            
    return exponents, coeff

def parse_tokenized_polynomial(tokens: List[str]) -> Tuple[List[List[int]], List[float]]:
    """Parse a tokenized polynomial into exponents and coefficients."""
    exponents_list = []
    coeffs_list = []
    i = 0
    while i < len(tokens):
        coeff_token = tokens[i]
        monomial_token = tokens[i+1]

        # Extract coefficient
        coeff = int(coeff_token.split('_')[1])
        coeffs_list.append(float(coeff))

        # Extract exponents
        exponent_parts = monomial_token.split('_')[1:]
        exponents = [int(p) for p in exponent_parts]
        exponents_list.append(exponents)
        
        i += 3 # Move to the next C_, M_, + group, or end
    return exponents_list, coeffs_list

def get_high_dim_example() -> List[List[int]]:
    """Returns a high-dimensional example polynomial's exponents as a list of lists."""
    # Example polynomial tokens (this is a complex polynomial in 5 variables)
    poly_tokens = ["C_25", "M_0_0_0_0_0", "+", "C_14", "M_2_0_0_1_0", "+", "C_-2", "M_0_0_0_1_4", "+", "C_-7", "M_2_0_0_0_2", "+", "C_-3", "M_0_1_0_3_1", "+", "C_-8", "M_0_1_0_3_0", "+", "C_6", "M_0_0_0_3_1", "+", "C_14", "M_3_0_0_1_1", "+", "C_-6", "M_0_0_2_0_2", "+", "C_2", "M_0_2_0_0_0", "+", "C_-12", "M_3_0_0_1_0", "+", "C_0", "M_2_0_1_1_0", "+", "C_13", "M_5_0_0_0_0", "+", "C_-2", "M_0_1_0_2_0", "+", "C_-8", "M_1_0_1_0_1", "+", "C_0", "M_1_1_1_0_0", "+", "C_-11", "M_3_1_0_0_0", "+", "C_-27", "M_2_1_0_0_0", "+", "C_-4", "M_0_0_3_0_2", "+", "C_-7", "M_0_1_2_0_0", "+", "C_26", "M_4_0_0_2_0", "+", "C_-7", "M_2_0_0_2_4", "+", "C_-12", "M_4_0_0_1_2", "+", "C_-8", "M_2_1_0_4_1", "+", "C_-3", "M_2_1_0_4_0", "+", "C_-12", "M_2_0_0_4_1", "+", "C_-1", "M_5_0_0_2_1", "+", "C_4", "M_2_0_2_1_2", "+", "C_-13", "M_2_2_0_1_0", "+", "C_-29", "M_5_0_0_2_0", "+", "C_0", "M_4_0_1_2_0", "+", "C_-6", "M_7_0_0_1_0", "+", "C_12", "M_2_1_0_3_0", "+", "C_9", "M_3_0_1_1_1", "+", "C_4", "M_3_1_1_1_0", "+", "C_2", "M_5_1_0_1_0", "+", "C_-10", "M_4_1_0_1_0", "+", "C_-12", "M_2_0_3_1_2", "+", "C_7", "M_2_1_2_1_0", "+", "C_15", "M_0_0_0_2_8", "+", "C_4", "M_2_0_0_1_6", "+", "C_-6", "M_0_1_0_4_5", "+", "C_2", "M_0_1_0_4_4", "+", "C_-9", "M_0_0_0_4_5", "+", "C_10", "M_3_0_0_2_5", "+", "C_14", "M_0_0_2_1_6", "+", "C_-4", "M_0_2_0_1_4", "+", "C_5", "M_3_0_0_2_4", "+", "C_-4", "M_2_0_1_2_4", "+", "C_3", "M_5_0_0_1_4", "+", "C_-2", "M_0_1_0_3_4", "+", "C_-5", "M_1_0_1_1_5", "+", "C_6", "M_1_1_1_1_4", "+", "C_7", "M_3_1_0_1_4", "+", "C_19", "M_2_1_0_1_4", "+", "C_11", "M_0_0_3_1_6", "+", "C_-16", "M_0_1_2_1_4", "+", "C_13", "M_4_0_0_0_4", "+", "C_-12", "M_2_1_0_3_3", "+", "C_4", "M_2_1_0_3_2", "+", "C_5", "M_2_0_0_3_3", "+", "C_-9", "M_5_0_0_1_3", "+", "C_5", "M_2_0_2_0_4", "+", "C_4", "M_2_2_0_0_2", "+", "C_21", "M_5_0_0_1_2", "+", "C_-5", "M_4_0_1_1_2", "+", "C_10", "M_7_0_0_0_2", "+", "C_-8", "M_2_1_0_2_2", "+", "C_-5", "M_3_0_1_0_3", "+", "C_0", "M_3_1_1_0_2", "+", "C_2", "M_5_1_0_0_2", "+", "C_14", "M_4_1_0_0_2", "+", "C_12", "M_2_0_3_0_4", "+", "C_5", "M_2_1_2_0_2", "+", "C_49", "M_0_2_0_6_2", "+", "C_26", "M_0_2_0_6_1", "+", "C_16", "M_0_1_0_6_2", "+", "C_-8", "M_3_1_0_4_2", "+", "C_-9", "M_0_1_2_3_3", "+", "C_-2", "M_0_3_0_3_1", "+", "C_5", "M_3_1_0_4_1", "+", "C_19", "M_2_1_1_4_1", "+", "C_-6", "M_5_1_0_3_1", "+", "C_12", "M_0_2_0_5_1", "+", "C_5", "M_1_1_1_3_2", "+", "C_4", "M_1_2_1_3_1", "+", "C_3", "M_3_2_0_3_1", "+", "C_-12", "M_2_2_0_3_1", "+", "C_2", "M_0_1_3_3_3", "+", "C_-23", "M_0_2_2_3_1", "+", "C_14", "M_0_2_0_6_0", "+", "C_7", "M_0_1_0_6_1", "+", "C_7", "M_0_1_2_3_2", "+", "C_4", "M_0_3_0_3_0", "+", "C_2", "M_3_1_0_4_0", "+", "C_7", "M_2_1_1_4_0", "+", "C_-4", "M_5_1_0_3_0", "+", "C_15", "M_0_2_0_5_0", "+", "C_4", "M_1_1_1_3_1", "+", "C_0", "M_1_2_1_3_0", "+", "C_3", "M_3_2_0_3_0", "+", "C_-2", "M_2_2_0_3_0", "+", "C_1", "M_0_1_3_3_2", "+", "C_-6", "M_0_2_2_3_0", "+", "C_14", "M_0_0_0_6_2", "+", "C_-5", "M_3_0_0_4_2", "+", "C_-3", "M_0_0_2_3_3", "+", "C_1", "M_0_2_0_3_1", "+", "C_7", "M_3_0_0_4_1", "+", "C_-6", "M_2_0_1_4_1", "+", "C_5", "M_5_0_0_3_1", "+", "C_2", "M_0_1_0_5_1", "+", "C_-8", "M_1_0_1_3_2", "+", "C_17", "M_3_1_0_3_1", "+", "C_-10", "M_2_1_0_3_1", "+", "C_-4", "M_0_0_3_3_3", "+", "C_-6", "M_0_1_2_3_1", "+", "C_27", "M_6_0_0_2_2", "+", "C_6", "M_3_0_2_1_3", "+", "C_2", "M_3_2_0_1_1", "+", "C_7", "M_6_0_0_2_1", "+", "C_-12", "M_5_0_1_2_1", "+", "C_-13", "M_8_0_0_1_1", "+", "C_-8", "M_4_1_1_1_1", "+", "C_0", "M_6_1_0_1_1", "+", "C_4", "M_5_1_0_1_1", "+", "C_-1", "M_3_0_3_1_3", "+", "C_-18", "M_3_1_2_1_1", "+", "C_21", "M_0_0_4_0_4", "+", "C_-7", "M_0_2_2_0_2", "+", "C_-1", "M_3_0_2_1_2", "+", "C_5", "M_5_0_2_0_2", "+", "C_9", "M_0_1_2_2_2", "+", "C_17", "M_1_0_3_0_3", "+", "C_-6", "M_1_1_3_0_2", "+", "C_13", "M_3_1_2_0_2", "+", "C_7", "M_0_0_5_0_4", "+", "C_-13", "M_0_1_4_0_2", "+", "C_18", "M_0_4_0_0_0", "+", "C_2", "M_3_2_0_1_0", "+", "C_-1", "M_2_2_1_1_0", "+", "C_6", "M_5_2_0_0_0", "+", "C_4", "M_0_3_0_2_0", "+", "C_-1", "M_1_2_1_0_1", "+", "C_-5", "M_1_3_1_0_0", "+", "C_-1", "M_3_3_0_0_0", "+", "C_-8", "M_2_3_0_0_0", "+", "C_5", "M_0_2_3_0_2", "+", "C_15", "M_0_3_2_0_0", "+", "C_32", "M_6_0_0_2_0", "+", "C_-1", "M_5_0_1_2_0", "+", "C_6", "M_8_0_0_1_0", "+", "C_-5", "M_3_1_0_3_0", "+", "C_-17", "M_4_0_1_1_1", "+", "C_2", "M_4_1_1_1_0", "+", "C_-3", "M_6_1_0_1_0", "+", "C_12", "M_3_0_3_1_2", "+", "C_-2", "M_3_1_2_1_0", "+", "C_16", "M_4_0_2_2_0", "+", "C_12", "M_7_0_1_1_0", "+", "C_8", "M_2_1_1_3_0", "+", "C_13", "M_3_0_2_1_1", "+", "C_-7", "M_5_1_1_1_0", "+", "C_2", "M_2_0_4_1_2", "+", "C_5", "M_2_1_3_1_0", "+", "C_17", "M_10_0_0_0_0", "+", "C_-6", "M_5_1_0_2_0", "+", "C_-3", "M_6_0_1_0_1", "+", "C_-9", "M_6_1_1_0_0", "+", "C_-15", "M_8_1_0_0_0", "+", "C_-7", "M_7_1_0_0_0", "+", "C_1", "M_5_0_3_0_2", "+", "C_-6", "M_5_1_2_0_0", "+", "C_19", "M_0_2_0_4_0", "+", "C_19", "M_1_1_1_2_1", "+", "C_-10", "M_1_2_1_2_0", "+", "C_1", "M_3_2_0_2_0", "+", "C_-14", "M_2_2_0_2_0", "+", "C_3", "M_0_1_3_2_2", "+", "C_-1", "M_0_2_2_2_0", "+", "C_20", "M_2_0_2_0_2", "+", "C_-7", "M_2_1_2_0_1", "+", "C_4", "M_4_1_1_0_1", "+", "C_-6", "M_3_1_1_0_1", "+", "C_4", "M_1_0_4_0_3", "+", "C_7", "M_1_1_3_0_1", "+", "C_9", "M_2_2_2_0_0", "+", "C_8", "M_4_2_1_0_0", "+", "C_1", "M_3_2_1_0_0", "+", "C_6", "M_1_1_4_0_2", "+", "C_-3", "M_1_2_3_0_0", "+", "C_15", "M_6_2_0_0_0", "+", "C_5", "M_3_1_3_0_2", "+", "C_-2", "M_3_2_2_0_0", "+", "C_24", "M_4_2_0_0_0", "+", "C_12", "M_2_1_3_0_2", "+", "C_15", "M_0_0_6_0_4", "+", "C_-2", "M_0_1_5_0_2", "+", "C_28", "M_0_2_4_0_0"]
    
    # Parse the polynomial and return only the exponents
    exponents_list, _ = parse_tokenized_polynomial(poly_tokens)
    return exponents_list

def get_polynomial_info(exponents_list: List[List[int]]) -> Dict:
    """Get information about the polynomial from its exponents."""
    info = {}
    info['num_terms'] = len(exponents_list)
    if exponents_list:
        num_variables = len(exponents_list[0])
        max_degrees = [max(exp[i] for exp in exponents_list if len(exp) > i) for i in range(num_variables)]
        info['max_degrees'] = {f"x{i+1}": max_degrees[i] for i in range(num_variables)}
    return info

def main():
    # Example usage
    exponents = get_high_dim_example()
    info = get_polynomial_info(exponents)
    
    print("# Exponents for the complex polynomial")
    print("complex_exponents = [")
    for exp in exponents:
        print(f"    {exp},")
    print("]")
    
    print(f"\nTotal number of terms: {info['num_terms']}")
    degree_str = ", ".join([f"{var}:{deg}" for var, deg in info['max_degrees'].items()])
    print(f"Maximum degrees: {degree_str}")

if __name__ == "__main__":
    main() 