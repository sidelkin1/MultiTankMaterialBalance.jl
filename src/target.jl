Base.@kwdef struct TargetTermParameters{A<:AbstractArray{<:AbstractFloat}, B<:AbstractArray{<:Real}, C<:AbstractArray{Bool}}
    Pobs::A
    Wobs::B
    Wnan::C
end

Base.@kwdef struct FloatTargetTerm{S, T} <: AbstractTargetTerm{S, T}
    params::TargetTermParameters{Matrix{T}, Matrix{T}, BitMatrix}
    pviews::Vector{TargetTermParameters{ColumnSlice{T}, ColumnSlice{T}, ColumnSliceBool}}
    α::T
end

Base.@kwdef struct BoolTargetTerm{S, T} <: AbstractTargetTerm{S, T}
    params::TargetTermParameters{Matrix{T}, BitMatrix, BitMatrix}
    pviews::Vector{TargetTermParameters{ColumnSlice{T}, ColumnSliceBool, ColumnSliceBool}}
    α::T
end

Base.@kwdef struct L2TargetTerm{T} <: AbstractTargetTerm{:L2, T}
    x::Vector{T}
    αₓ::Vector{T}
    α::T
end

Base.@kwdef struct TargetFunction{T<:AbstractFloat}
    terms::@NamedTuple begin
        Pres::FloatTargetTerm{:Pres, T}
        Pbhp::FloatTargetTerm{:Pbhp, T}
        Pinj::FloatTargetTerm{:Pinj, T}
        Pmin::BoolTargetTerm{:Pmin, T}
        Pmax::BoolTargetTerm{:Pmax, T}
        L2::L2TargetTerm{T}
    end
    prob::NonlinearProblem{T}
end

getpcalc(prob::NonlinearProblem, ::Val{S}) where {S} = nothing
getpcalc(prob::NonlinearProblem, ::Val{:Pres}) = prob.params.Pcalc
getpcalc(prob::NonlinearProblem, ::Val{:Pbhp}) = prob.params.Pbhp
getpcalc(prob::NonlinearProblem, ::Val{:Pinj}) = prob.params.Pinj
getpcalc(prob::NonlinearProblem, ::Val{:Pmin}) = prob.params.Pcalc
getpcalc(prob::NonlinearProblem, ::Val{:Pmax}) = prob.params.Pcalc
getpcalc(prob::NonlinearProblem, n, ::Val{S}) where {S} = nothing
getpcalc(prob::NonlinearProblem, n, ::Val{:Pres}) = @inbounds prob.pviews[n].Pcalc
getpcalc(prob::NonlinearProblem, n, ::Val{:Pbhp}) = @inbounds prob.pviews[n].Pbhp
getpcalc(prob::NonlinearProblem, n, ::Val{:Pinj}) = @inbounds prob.pviews[n].Pinj
getpcalc(prob::NonlinearProblem, n, ::Val{:Pmin}) = @inbounds prob.pviews[n].Pcalc
getpcalc(prob::NonlinearProblem, n, ::Val{:Pmax}) = @inbounds prob.pviews[n].Pcalc

function FloatTargetTerm{S, T}(Pobs, Wobs, α, N, M) where {S, T}
    Wnan = @. !ismissing(Pobs) & !ismissing(Wobs)
    data = (
        Pobs = build_rate_matrix(T, Pobs, N, M),
        Wobs = build_rate_matrix(T, Wobs, N, M),
        Wnan = (permutedims∘reshape)(Wnan, M, N),
    )
    params, pviews = params_and_views(TargetTermParameters, data)
    FloatTargetTerm{S, T}(; params, pviews, α)
end

function BoolTargetTerm{S, T}(Pobs, α, N, M) where {S, T}
    Wnan = @. !ismissing(Pobs)
    data = (
        Pobs = build_rate_matrix(T, Pobs, N, M),
        Wobs = falses(N, M),
        Wnan = (permutedims∘reshape)(Wnan, M, N),
    )
    params, pviews = params_and_views(TargetTermParameters, data)
    BoolTargetTerm{S, T}(; params, pviews, α)
end

function TargetFunction{T}(df_rates::AbstractDataFrame, prob::NonlinearProblem{T}, fset::FittingSet{T}, α) where {T}
    
    Nt = size(prob.C, 2)
    Nd = length(prob.pviews)

    terms = @with df_rates begin (
        Pres = FloatTargetTerm{^(:Pres), T}(:Pres, :Wres, α["alpha_resp"], Nt, Nd),
        Pbhp = FloatTargetTerm{^(:Pbhp), T}(:Pbhp_prod, :Wbhp_prod, α["alpha_bhp"], Nt, Nd),
        Pinj = FloatTargetTerm{^(:Pinj), T}(:Pbhp_inj, :Wbhp_inj, α["alpha_inj"], Nt, Nd),
        Pmin = BoolTargetTerm{^(:Pmin), T}(:Pres_min, α["alpha_lb"], Nt, Nd),
        Pmax = BoolTargetTerm{^(:Pmax), T}(:Pres_max, α["alpha_ub"], Nt, Nd),
        L2 = L2TargetTerm{T}(fset.cache.ybuf, fset.α, α["alpha_l2"]),
    ) end
    
    TargetFunction{T}(; terms, prob)
end

function update_targ!(targ::TargetFunction)
    update_term!(targ.terms.Pmin, targ.prob)
    update_term!(targ.terms.Pmax, targ.prob)
    return targ
end

function update_term!(term::BoolTargetTerm{:Pmin}, prob::NonlinearProblem)
    @unpack Pcalc = prob.params
    @unpack Wobs, Pobs = term.params
    @inbounds @simd for i = 1:length(Pcalc)
        Wobs[i] = Pcalc[i] < Pobs[i]
    end
    return term
end

function update_term!(term::BoolTargetTerm{:Pmax}, prob::NonlinearProblem)
    @unpack Pcalc = prob.params
    @unpack Wobs, Pobs = term.params
    @inbounds @simd for i = 1:length(Pcalc)
        Wobs[i] = Pcalc[i] > Pobs[i]
    end
    return term
end

function grad!(g, term::AbstractTargetTerm{S, T}, prob::NonlinearProblem, n) where {S, T}
    @unpack Pobs, Wobs, Wnan = @inbounds term.pviews[n]
    Pcalc = getpcalc(prob, n, Val(S))    
    @inbounds @simd for i = 1:length(Pcalc)
        g[i] += Wnan[i] * (T(2) * term.α * Wobs[i] * (Pcalc[i] - Pobs[i]))
    end
    return g
end

function grad!(g, targ::TargetFunction{T}, n) where {T}    
    fill!(g, zero(T))
    terms = @inbounds values(targ.terms)[1:end-1]    
    # FIXED: Использование 'map' вместо 'for' сохраняет 'type-stability'
    map(term -> grad!(g, term, targ.prob, n), terms)
    return g
end

getvalue(targ::TargetFunction) = sum(term -> getvalue(term, targ.prob), values(targ.terms))
getvalues(targ::TargetFunction) = map(term -> getvalue(term, targ.prob), targ.terms)

function getvalue(term::L2TargetTerm{T}, prob::NonlinearProblem) where {T}
    @unpack x, αₓ = term
    val = zero(T)
    @inbounds @simd for i = 1:length(x)
        val += αₓ[i] * x[i]^2
    end
    return term.α * val
end

function getvalue(term::AbstractTargetTerm{S, T}, prob::NonlinearProblem) where {S, T}
    @unpack Pobs, Wobs, Wnan = term.params
    Pcalc = getpcalc(prob, Val(S))    
    val = zero(T)
    @inbounds @simd for i = 1:length(Pcalc)
        val += Wnan[i] * (Wobs[i] * (Pcalc[i] - Pobs[i])^2)
    end
    return term.α * val
end