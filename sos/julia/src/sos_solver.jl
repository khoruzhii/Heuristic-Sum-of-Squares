module SOSolver

using JuMP
using SumOfSquares
using DynamicPolynomials
using MosekTools

export solve_sos


"""
    _get_fallback_mosek_params()

Get fallback MOSEK parameters in case config file loading fails.
"""
function _get_fallback_mosek_params()
    return Dict(
        # Conic interior-point tolerances (most relevant for SDPs)
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS" => 1e-3,
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => 1e-3,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => 1e-3,
        "MSK_DPAR_INTPNT_CO_TOL_MU_RED" => 1e-3,
        "MSK_DPAR_INTPNT_CO_TOL_INFEAS" => 1e-3,
        # Non-conic tolerances (harmless; sometimes still used internally)
        "MSK_DPAR_INTPNT_TOL_DFEAS" => 1e-3,
        "MSK_DPAR_INTPNT_TOL_PFEAS" => 1e-3,
        "MSK_DPAR_INTPNT_TOL_REL_GAP" => 1e-3,
        "MSK_DPAR_INTPNT_TOL_MU_RED" => 1e-3,
        "MSK_DPAR_INTPNT_TOL_INFEAS" => 1e-3,
        # Iterations / scaling / logging
        "MSK_IPAR_INTPNT_MAX_ITERATIONS" => 10_000,
        "MSK_IPAR_INTPNT_SCALING" => 1,
        "MSK_IPAR_LOG_INTPNT" => 1,
    )
end

# Default MOSEK parameters (fallback when none provided)
function get_default_mosek_params()
    return _get_fallback_mosek_params()
end

# Build a Mosek optimizer with attributes. You can pass extra overrides.
function _mosek_with(params::Union{Dict, Nothing}=nothing)
    if params === nothing
        params = get_default_mosek_params()
    end
    return optimizer_with_attributes(Mosek.Optimizer, (k=>v for (k,v) in params)...)
end

"""
    SOSResult

Result structure for SOS computations.
"""
struct SOSResult
    status::String
    setup_time::Float64
    solve_time::Float64
    total_time::Float64
    model::Union{JuMP.Model, Nothing}
    info::Union{Dict, Nothing}
end

"""
    solve_sos(f, vars; verbose=true, mosek_params=nothing)

Solve SOS decomposition for polynomial f using SumOfSquares.jl.
Let the solver automatically choose the monomial basis.

# Arguments
- `f`: Polynomial to decompose
- `vars`: Vector of variables
- `verbose`: Whether to print progress information
- `mosek_params`: Dictionary of MOSEK solver parameters

# Returns
- SOSResult with timing and status information
"""
function solve_sos(f, vars; verbose=true, mosek_params::Union{Dict, Nothing}=nothing)
    if mosek_params === nothing
        mosek_params = get_default_mosek_params()
    end
    
    if verbose
        println("=== SOS Decomposition (Auto Basis) ===")
        #println("Polynomial: ", f)
        println("Variables: ", vars)
    end
    
    # Setup time
    setup_time = @elapsed begin
        model = SOSModel(_mosek_with(mosek_params))
        # No objective - just feasibility
        @objective(model, Max, 0)
        @constraint(model, f in SOSCone(), sparsity=Sparsity.Monomial(ChordalCompletion()))
    end
    
    if verbose
        println("Setup time: ", setup_time, " seconds")
    end
    
    # Solve time
    solve_time = @elapsed begin
        optimize!(model)
    end
    
    if verbose
        println("Solve time: ", solve_time, " seconds")
        println("Total time: ", setup_time + solve_time, " seconds")
        println("Status: ", termination_status(model))
    end
    
    return SOSResult(
        string(termination_status(model)),
        setup_time,
        solve_time,
        setup_time + solve_time,
        model,
        nothing
    )
end

"""
    solve_sos(f, vars; verbose=true, solver="mosek", solver_params=nothing)

Generic solve SOS decomposition for polynomial f using SumOfSquares.jl with configurable solver.

# Arguments
- `f`: Polynomial to decompose
- `vars`: Vector of variables
- `verbose`: Whether to print progress information
- `solver`: Solver name ("mosek", "scs", "clarabel", "sdpa")
- `solver_params`: Dictionary of solver-specific parameters

# Returns
- SOSResult with timing and status information
"""
function solve_sos(f, vars; verbose=true, solver::String="mosek", solver_params::Union{Dict, Nothing}=nothing, mosek_params::Union{Dict, Nothing}=nothing)
    # For backward compatibility, handle old mosek_params keyword
    if mosek_params !== nothing && solver == "mosek"
        solver_params = mosek_params
    end
    
    if verbose
        println("=== SOS Decomposition ($(uppercase(solver)) Solver) ===")
        println("Variables: ", vars)
    end
    
    # Setup optimizer based on solver type
    setup_time = @elapsed begin
        if solver == "mosek"
            if solver_params === nothing
                solver_params = get_default_mosek_params()
            end
            optimizer = _mosek_with(solver_params)
        else
            # For other solvers, we'll stick with MOSEK as fallback for now
            # This can be extended when other Julia solver packages are available
            @warn "Solver '$solver' not fully supported in Julia, falling back to MOSEK"
            if solver_params !== nothing && solver == "mosek"
                optimizer = _mosek_with(solver_params)
            else
                optimizer = _mosek_with(get_default_mosek_params())
            end
        end
        
        model = SOSModel(optimizer)
        @objective(model, Max, 0)
        @constraint(model, f in SOSCone(), sparsity=Sparsity.Monomial(ChordalCompletion()))
    end
    
    if verbose
        println("Setup time: ", setup_time, " seconds")
    end
    
    # Solve time
    solve_time = @elapsed begin
        optimize!(model)
    end
    
    if verbose
        println("Solve time: ", solve_time, " seconds")
        println("Total time: ", setup_time + solve_time, " seconds")
        println("Status: ", termination_status(model))
    end
    
    return SOSResult(
        string(termination_status(model)),
        setup_time,
        solve_time,
        setup_time + solve_time,
        model,
        nothing
    )
end

end # module 