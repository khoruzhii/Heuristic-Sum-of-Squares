#!/usr/bin/env julia

# Simplified SOS test script for JSONL polynomials
# Usage: julia test_jsonl_sos.jl <jsonl_path> <num_variables> [solver] [config]
# Example: julia test_jsonl_sos.jl data.jsonl 4 mosek default
# Example: julia test_jsonl_sos.jl data.jsonl 4 mosek high_precision
# Example: julia test_jsonl_sos.jl data.jsonl 4 scs default
println("Testing SOS Solver on JSONL Polynomials...")

# Parse command line arguments
if length(ARGS) >= 2
    const JSONL_PATH = ARGS[1]
    const NUM_VARIABLES = parse(Int, ARGS[2])
    const SOLVER_NAME = length(ARGS) >= 3 ? lowercase(ARGS[3]) : "mosek"
    const SOLVER_CONFIG = length(ARGS) >= 4 ? ARGS[4] : "default"
    println("Using command line arguments:")
    println("  JSONL Path: ", JSONL_PATH)
    println("  Number of variables: ", NUM_VARIABLES)
    println("  Solver: ", SOLVER_NAME)
    println("  Solver config: ", SOLVER_CONFIG)
else
    # Fallback to default values if no arguments provided
    const JSONL_PATH = "/scratch/llm/ais2t/sos/ood/n6_sparse_uniform_simple_random_d10_m30/test.jsonl"
    const NUM_VARIABLES = 6
    const SOLVER_NAME = "mosek"
    const SOLVER_CONFIG = "default"
    println("Using default values:")
    println("  JSONL Path: ", JSONL_PATH)
    println("  Number of variables: ", NUM_VARIABLES)
    println("  Solver: ", SOLVER_NAME)
    println("  Solver config: ", SOLVER_CONFIG)
end

# Other configuration variables
const MAX_POLYNOMIALS = 10
const VERBOSE = true

# Add the current directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

# Import the module
using SOSBenchmarks
using DynamicPolynomials
using SumOfSquares
using JSON3

println("✓ Module imported successfully")

# Load solver settings from JSONL
println("\n=== Loading Solver Settings ===")
solver_config_path = "/home/htc/npelleriti/sum-of-squares-transformer/sos/configs/solvers/solver_settings.jsonl"
println("Loading solver settings from: ", solver_config_path)
println("Looking for solver: ", SOLVER_NAME, ", config: ", SOLVER_CONFIG)

solver_params = nothing
found_config = false

if isfile(solver_config_path)
    open(solver_config_path, "r") do file
        for line in eachline(file)
            isempty(strip(line)) && continue
            try
                settings = JSON3.read(line)
                if get(settings, "solver", "") == SOLVER_NAME && get(settings, "config", "") == SOLVER_CONFIG
                    raw_params = get(settings, "params", Dict())
                    # Convert all keys to strings to ensure compatibility
                    global solver_params = Dict(string(k) => v for (k, v) in raw_params)
                    global found_config = true
                    println("✓ Loaded $(uppercase(SOLVER_NAME)) parameters:")
                    for (key, value) in solver_params
                        println("  $key => $value")
                    end
                    break
                end
            catch e
                @warn "Failed to parse line in config file: $line"
                continue
            end
        end
    end
    
    if !found_config
        @warn "Could not find configuration for solver '$(SOLVER_NAME)' with config '$(SOLVER_CONFIG)'"
        println("⚠️  Will use solver defaults")
    end
else
    println("⚠️  Solver config file not found, will use solver defaults")
end

# Load polynomials from JSONL
println("\n=== Loading Polynomials ===")
println("Loading polynomials from: ", JSONL_PATH)

polynomials = read_jsonl_polynomials(JSONL_PATH, NUM_VARIABLES, max_polynomials=MAX_POLYNOMIALS)
println("Loaded ", length(polynomials), " polynomials")

# Create variables
vars = create_variables(NUM_VARIABLES)

# Test each polynomial
println("\n=== Testing Polynomials ===")
for i in 1:length(polynomials)
    println("\n--- Polynomial $i ---")
    f = polynomials[i]
    
    # Print polynomial info
    println("Degree: ", maxdegree(f))
    println("Terms: ", length(terms(f)))
    
    # Run SOS benchmark
    results = solve_sos(f, vars, verbose=VERBOSE, solver=SOLVER_NAME, solver_params=solver_params)
    
    
end

println("\n=== Test Completed ===")
