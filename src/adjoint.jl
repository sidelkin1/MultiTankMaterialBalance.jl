Base.@kwdef mutable struct AdjointSolver{T}
    prob::NonlinearProblem{T}
    targ::TargetFunction{T}
    μ::Vector{T}
    g::Vector{T}
end

function AdjointSolver{T}(prob::NonlinearProblem{T}, targ::TargetFunction{T}) where {T}
    Nt = size(prob.params.Pcalc, 1)
    kwargs = (
        μ = Array{T}(undef, Nt),
        g = Array{T}(undef, Nt),
    )
    AdjointSolver{T}(; prob, targ, kwargs...)
end

function solve!(solver::AdjointSolver)

    @unpack μ, g = solver

    update_targ!(solver.targ)    
    fill!(μ, 0)
    for (n, (params, jac, terms...)) ∈ enumerate(solver.itr)
        @unpack jac_next = params
        gradP!(g, targ, terms)
        @. g = -(jac_next * μ + g)
        μ .= jac \ g
        println("n: $n, mu: $μ")
    end

    return solver
end

function Base.getproperty(obj::AdjointSolver, sym::Symbol)
    if sym === :itr
        @unpack pviews, jacs = obj.prob
        terms = @_ map(_.pviews, Base.front(obj.targ.terms))
        return Iterators.reverse(zip(pviews, jacs, terms...))
    else # fallback to getfield
        return getfield(obj, sym)
    end
end