module SOSBenchmarks

# Import the data loader module
include("data_loader.jl")
using .DataLoader

# Import the SOS solver module
include("sos_solver.jl")
using .SOSolver

# Re-export only the functions needed for test_jsonl_sos.jl
export read_jsonl_polynomials, create_variables, solve_sos

end # module 