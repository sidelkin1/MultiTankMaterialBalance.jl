Base.@kwdef mutable struct NewtonSolver{T, M} <: AbstractNonlinearSolver{T, M}
    maxiters::Int
    P_tol::T    
    r_tol::T
    prob::NonlinearProblem{T}
    alg::NewtonAlgorithm{M}
    cache::ProblemCache{T}
    iter::Int
    norm_r::T
    dP_rel::T
    P::Vector{T}
    P_old::Vector{T}
    r::Vector{T}
    J::M
    F::SuiteSparse.CHOLMOD.Factor
    success::Bool
end

function init(prob::NonlinearProblem{T}, alg::NewtonAlgorithm{M}; maxiters, P_tol, r_tol) where {T, M}
    Nt = size(prob.params.Pcalc, 1)
    kwargs = (   
        cache = ProblemCache{T}(prob),
        iter = 0,
        norm_r = zero(T),
        dP_rel = zero(T),
        P = Array{T}(undef, Nt),
        P_old = Array{T}(undef, Nt),
        r = Array{T}(undef, Nt),
        J = zeros(T, Nt, Nt),
        F = cholesky(spdiagm(ones(T, Nt))),
        success = false,
    )
    NewtonSolver{T, M}(; maxiters, P_tol, r_tol, prob, alg, kwargs...)
end

function perform_step!(solver::NewtonSolver, ::DenseNewtonAlgorithm, params::ModelParameters)

    @unpack P, P_old, r, J = solver

    iter, success = 0, false
    @fastmath for outer iter = 1:solver.maxiters
        val_and_jac!(r, J, P, solver.cache, params)
        
        copyto!(P_old, P)
        if size(J, 1) ≤ 100
            # RecursiveFactorization seems to be consistantly winning below 100
            # https://discourse.julialang.org/t/ann-recursivefactorization-jl/39213
            F = RecursiveFactorization.lu!(J, Val(true); threshold=100)
        else
            F = cholesky!(J, Val(true))
        end        
        ldiv!(P, F, r)
        @. P = P_old - P

        x_converged, f_converged = assess_convergence(P, P_old, r, solver.P_tol, solver.r_tol)
        (success = x_converged | f_converged) && break
    end
    norm_r = maximum(abs, r)
    dP_rel = maxdiff(P, P_old) / maximum(abs, P) 

    @pack! solver = iter, norm_r, dP_rel, success
    
    return solver
end

function perform_step!(solver::NewtonSolver, ::SparseNewtonAlgorithm, params::ModelParameters)

    @unpack P, P_old, r, J, F = solver

    iter, success = 0, false
    @fastmath for outer iter = 1:solver.maxiters
        val_and_jac!(r, J, P, solver.cache, params)
        
        copyto!(P_old, P)
        cholesky!(F, J) 
        P .= F \ r
        @. P = P_old - P

        x_converged, f_converged = assess_convergence(P, P_old, r, solver.P_tol, solver.r_tol)
        (success = x_converged | f_converged) && break
    end
    norm_r = maximum(abs, r)
    dP_rel = maxdiff(P, P_old) / maximum(abs, P) 

    @pack! solver = iter, norm_r, dP_rel, success
    
    return solver
end