Base.@kwdef mutable struct NewtonSolver{T, LS<:AbstractLinearSolver{T}} <: AbstractNonlinearSolver{T}
    prob::NonlinearProblem{T}
    linalg::LS
    maxiters::Int
    P_tol::T    
    r_tol::T    
    iter::Int
    norm_r::T
    dP_rel::T
    P::Vector{T}
    P_old::Vector{T}
    r::Vector{T}    
    success::Bool
end

function NewtonSolver{T}(prob::NonlinearProblem{T}, linalg::AbstractLinearSolver{T}, opts) where {T}
    Nt = size(prob.C, 2)
    kwargs = (
        maxiters = opts["maxiters"],
        P_tol = opts["P_tol"],
        r_tol = opts["r_tol"],
        iter = 0,
        norm_r = zero(T),
        dP_rel = zero(T),
        P = Array{T}(undef, Nt),
        P_old = Array{T}(undef, Nt),
        r = Array{T}(undef, Nt),
        success = false,
    )
    NewtonSolver{T, typeof(linalg)}(; prob, linalg, kwargs...)
end

function perform_step!(solver::NewtonSolver, Δt, n)

    @unpack P, P_old, r, linalg, prob, P_tol, r_tol = solver
    
    iter, success = 0, false
    for outer iter = 1:solver.maxiters
        J = getjac(linalg, n)
        val_and_jac!(r, J, P, Δt, prob, n)
        
        copyto!(P_old, P)
        solve!(P, r, n, linalg)        
        @turbo for i = 1:length(P)
            P[i] = P_old[i] - P[i]
        end

        x_converged, f_converged = assess_convergence(P, P_old, r, P_tol, r_tol)
        (success = x_converged | f_converged) && break
    end
    setjac!(linalg, n)
    norm_r = maximum(abs, r)
    dP_rel = maxdiff(P, P_old) / maximum(abs, P) 

    @pack! solver = iter, norm_r, dP_rel, success
    
    return solver
end