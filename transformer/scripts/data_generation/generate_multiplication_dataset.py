import sympy as sp
import random
from pathlib import Path
import argparse
from fractions import Fraction
import numpy as np

def create_monomial(variables, max_degree, coefficient_type="rational"):
    """Create a single monomial with given parameters."""
    # Generate random exponents - ensure they're small enough that products won't exceed max_degree
    # Since we're multiplying polynomials, each exponent should be at most max_degree/2
    safe_max_degree = min(max_degree // 2, 30)  # 30 is half of max allowed (62)
    exponents = [random.randint(0, safe_max_degree) for _ in range(len(variables))]
    
    # Generate coefficient based on type
    if coefficient_type == "rational":
        # Generate rational coefficient as fraction of small integers
        numerator = random.randint(1, 2)
        denominator = random.randint(1, 2)
        coeff = Fraction(numerator, denominator)
    else:  # real
        # Generate real coefficient between 0.1 and 5.0
        coeff = round(random.uniform(0.1, 5.0), 1)
    
    # Create the monomial
    monomial = sp.Mul(coeff, *[var**exp for var, exp in zip(variables, exponents)])
    return monomial

def create_polynomial(variables, max_terms, max_degree, coefficient_type="rational"):
    """Create a polynomial with given parameters."""
    num_terms = random.randint(1, max_terms)
    terms = [create_monomial(variables, max_degree, coefficient_type) for _ in range(num_terms)]
    return sum(terms)

def format_monomial(term, variables, coefficient_type):
    """Format a monomial in the required token format."""
    try:
        # Extract coefficient and exponents
        expanded = sp.expand(term)
        # Get the coefficient by dividing by the variable parts
        var_part = sp.Mul(*[var**sp.degree(expanded, var) for var in variables])
        if var_part == 0:
            coeff = float(expanded)
        else:
            coeff = float(expanded / var_part)
        
        # Get exponents
        exponents = [sp.degree(expanded, var) for var in variables]
        
        # Format coefficient based on type
        if coefficient_type == "rational":
            # Always convert to fraction for rational mode
            frac = Fraction(coeff).limit_denominator()
            coeff_str = f"C{frac.numerator}_{frac.denominator}"
        else:  # real
            # Always use decimal format for real mode
            coeff_str = f"C{coeff:.1f}"
        
        # Format exponents
        exp_str = " ".join(f"E{exp}" for exp in exponents)
        
        return f"{coeff_str} {exp_str}"
    except Exception as e:
        print(f"Error in format_monomial:")
        print(f"Term: {term}")
        print(f"Variables: {variables}")
        print(f"Expanded: {sp.expand(term)}")
        raise e

def format_polynomial(poly, variables, coefficient_type):
    """Format entire polynomial in the required token format."""
    expanded = sp.expand(poly)
    terms = sp.Add.make_args(expanded)
    formatted_terms = [format_monomial(term, variables, coefficient_type) for term in terms]
    return " + ".join(formatted_terms)

def validate_polynomial(poly, variables, max_allowed_exponent=62):
    """Validate that a polynomial's exponents are within bounds."""
    expanded = sp.expand(poly)
    terms = sp.Add.make_args(expanded)
    
    for term in terms:
        for var in variables:
            if sp.degree(term, var) > max_allowed_exponent:
                return False
    return True

def generate_example(variables, max_terms, max_degree, coefficient_type="rational", max_attempts=10):
    """Generate a single multiplication example."""
    for _ in range(max_attempts):
        try:
            # Create two random polynomials
            poly1 = create_polynomial(variables, max_terms, max_degree, coefficient_type)
            poly2 = create_polynomial(variables, max_terms, max_degree, coefficient_type)
            
            # Compute their product
            product = sp.expand(poly1 * poly2)
            
            # Validate the product's exponents
            if not validate_polynomial(product, variables):
                continue
            
            # Format all polynomials
            input_str = f"{format_polynomial(poly1, variables, coefficient_type)} [SEP] {format_polynomial(poly2, variables, coefficient_type)}"
            output_str = format_polynomial(product, variables, coefficient_type)
            
            return input_str, output_str
            
        except Exception as e:
            continue
    
    raise ValueError(f"Could not generate valid example after {max_attempts} attempts")

def main():
    parser = argparse.ArgumentParser(description="Generate polynomial multiplication dataset")
    parser.add_argument("--num_examples", type=int, default=10000, help="Number of examples to generate")
    parser.add_argument("--num_vars", type=int, default=3, help="Number of variables")
    parser.add_argument("--max_terms", type=int, default=3, help="Maximum number of terms per polynomial")
    parser.add_argument("--max_degree", type=int, default=2, help="Maximum degree per variable")
    parser.add_argument("--coeff_type", choices=["rational", "real"], default="rational", help="Type of coefficients")
    parser.add_argument("--output_dir", type=str, default="data/multiplication", help="Output directory")
    parser.add_argument("--split", type=float, default=0.8, help="Train/test split ratio")
    args = parser.parse_args()

    # Create variables
    variables = sp.symbols(f'x:{args.num_vars}')
    
    # Create output directories
    output_dir = Path(args.output_dir) / args.coeff_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate examples
    examples = []
    for _ in range(args.num_examples):
        input_str, output_str = generate_example(
            variables=variables,
            max_terms=args.max_terms,
            max_degree=args.max_degree,
            coefficient_type=args.coeff_type
        )
        examples.append((input_str, output_str))
    
    # Split into train and test
    split_idx = int(args.num_examples * args.split)
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]
    
    # Save train set
    with open(output_dir / "train.infix", "w") as f:
        for input_str, output_str in train_examples:
            f.write(f"{input_str} # {output_str}\n")
    
    # Save test set
    with open(output_dir / "test.infix", "w") as f:
        for input_str, output_str in test_examples:
            f.write(f"{input_str} # {output_str}\n")
    
    print(f"Generated {len(train_examples)} training examples and {len(test_examples)} test examples")
    print(f"Files saved in {output_dir}")
    
    # Print a few examples
    print("\nExample from training set:")
    print(f"Input: {train_examples[0][0]}")
    print(f"Output: {train_examples[0][1]}")

if __name__ == "__main__":
    main() 