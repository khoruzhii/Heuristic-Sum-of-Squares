module DataLoader

using JSON3
using DynamicPolynomials

export read_jsonl_polynomials, create_variables

"""
    create_variables(n_vars::Int)

Create a vector of polynomial variables with names x1, x2, ..., xn.
"""
function create_variables(n_vars::Int)
    # Use @polyvar to create variables with indexed notation
    @polyvar x[1:n_vars]
    
    # Convert to a regular vector
    vars = [x[i] for i in 1:n_vars]
    
    return vars
end

"""
    parse_tokenized_polynomial(tokens, n_vars::Int=8)

Parse a tokenized polynomial representation into a DynamicPolynomials polynomial.
The tokens should be in the format: C{coeff}E{exp1}E{exp2}...E{expN}+

# Arguments
- `tokens`: Vector of strings representing the tokenized polynomial
- `n_vars`: Number of variables (default: 8)

# Returns
- Polynomial in DynamicPolynomials format, or `nothing` if parsing fails
"""
function parse_tokenized_polynomial(tokens, n_vars::Int=8)
    # Create variables
    vars = create_variables(n_vars)
    
    # Initialize the polynomial
    f = 0
    
    # Process tokens in groups
    i = 1
    while i <= length(tokens)
        if startswith(tokens[i], "C")
            # Extract coefficient
            coeff = parse(Float64, tokens[i][2:end])
            
            # Extract exponents (next n_vars tokens should be E0, E1, etc.)
            exponents = []
            for j in 1:n_vars
                if i + j <= length(tokens) && startswith(tokens[i + j], "E")
                    exp_val = parse(Int, tokens[i + j][2:end])
                    push!(exponents, exp_val)
                else
                    println("Error: Expected exponent at position ", i + j)
                    return nothing
                end
            end
            
            # Build the monomial term
            term = coeff
            for (var, exp) in zip(vars, exponents)
                if exp > 0
                    term *= var^exp
                end
            end
            
            f += term
            
            # Skip to next term (after the + sign)
            i += n_vars + 2  # coefficient + n_vars exponents + 1 for the "+"
        else
            i += 1
        end
    end
    
    return f
end

"""
    read_jsonl_polynomials(filepath::String, n_vars::Int=8; max_polynomials::Union{Int, Nothing}=nothing)

Read polynomials from a JSONL file and parse them into DynamicPolynomials format.

# Arguments
- `filepath`: Path to the JSONL file
- `n_vars`: Number of variables (default: 8)
- `max_polynomials`: Maximum number of polynomials to load (default: nothing, loads all)

# Returns
- Vector of parsed polynomials
"""
function read_jsonl_polynomials(filepath::String, n_vars::Int=8; max_polynomials::Union{Int, Nothing}=nothing)
    polynomials = []
    
    open(filepath, "r") do file
        for line in eachline(file)
            # Stop if we've reached the maximum number of polynomials
            if max_polynomials !== nothing && length(polynomials) >= max_polynomials
                break
            end
            
            if !isempty(strip(line))
                try
                    # Parse JSON line
                    data = JSON3.read(line)
                    
                    # Extract tokenized polynomial
                    if haskey(data, "polynomial_tokens")
                        tokens = data["polynomial_tokens"]
                        poly = parse_tokenized_polynomial(tokens, n_vars)
                        
                        if poly !== nothing
                            push!(polynomials, poly)
                        else
                            println("Warning: Failed to parse polynomial from line: ", line)
                        end
                    else
                        println("Warning: No 'polynomial_tokens' field found in line: ", line)
                    end
                catch e
                    println("Error parsing JSON line: ", e)
                    println("Line: ", line)
                end
            end
        end
    end
    
    return polynomials
end


end # module
