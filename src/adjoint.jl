Base.@kwdef struct AdjointSolver{T, LS<:AbstractLinearSolver{T}, FS<:FittingSet{T}}
    prob::NonlinearProblem{T}
    targ::TargetFunction{T}
    linalg::LS
    fset::FS
    μ::Vector{T}
    gp::Vector{T}
    g::Vector{T}    
end

function AdjointSolver{T}(prob::NonlinearProblem{T}, targ::TargetFunction{T}, linalg::AbstractLinearSolver{T}, fset::FittingSet{T}) where {T}
    Nt = size(prob.C, 2)
    Nx = length(fset.cache.xbuf)
    kwargs = (
        μ = Array{T}(undef, Nt),
        gp = Array{T}(undef, Nt),
        g = Array{T}(undef, Nx),        
    )
    AdjointSolver{T, typeof(linalg), typeof(fset)}(; prob, targ, linalg, fset, kwargs...)
end

function solve!(solver::AdjointSolver{T}; verbose=false) where {T}

    @unpack μ, g, gp, prob, linalg, targ, fset = solver
    
    fill!(μ, zero(T))
    fill!(g, zero(T))
    @inbounds for n = length(prob.pviews):-1:2
        @unpack jac_next = prob.pviews[n]
        jac = linalg.Fbuf[n]        

        # Вычисление сопряженного вектора
        grad!(gp, targ, n)
        @simd for i = 1:length(gp)
            gp[i] = -(jac_next[i] * μ[i] + gp[i])
        end
        # TODO: Почему-то быстрее, чем 'ldiv!'
        μ .= jac \ gp
        
        # Вычисление градиента целевой функции
        grad!(g, fset, prob, targ, μ, n)
                
        verbose && println("n: $n, mu: $μ")
    end

    # Градиент члена L2-регуляризации
    grad!(g, targ.terms.L2)    
    # Коррекция градиента в соответствии с масштабированием параметров
    unscaleg!(g, fset.scale)

    return solver
end