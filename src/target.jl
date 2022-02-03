Base.@kwdef struct TargetTermParameters{A<:AbstractArray{<:AbstractFloat}, B<:AbstractArray{<:Real}, C<:AbstractArray{Bool}}
    Pcalc::A
    Pobs::A
    Wobs::B
    Wnan::C
end

Base.@kwdef struct FloatTargetTerm{T, S} <: AbstractTargetTerm{T}
    params::TargetTermParameters{Matrix{T}, Matrix{T}, BitMatrix}
    pviews::Vector{TargetTermParameters{ColumnSlice{T}, ColumnSlice{T}, ColumnSliceBool}}
    α::T
end

Base.@kwdef struct BoolTargetTerm{T, S} <: AbstractTargetTerm{T}
    params::TargetTermParameters{Matrix{T}, BitMatrix, BitMatrix}
    pviews::Vector{TargetTermParameters{ColumnSlice{T}, ColumnSliceBool, ColumnSliceBool}}
    α::T
end

Base.@kwdef struct L2TargetTerm{T} <: AbstractTargetTerm{T}
    fset::FittingSet{T}
    α::T
end

Base.@kwdef struct TargetFunction{T<:AbstractFloat}
    terms::@NamedTuple begin
        Pres::FloatTargetTerm{T, :Pres}
        Pbhp::FloatTargetTerm{T, :Pbhp}
        Pinj::FloatTargetTerm{T, :Pinj}
        Pmin::BoolTargetTerm{T, :Pmin}
        Pmax::BoolTargetTerm{T, :Pmax}
        L2::L2TargetTerm{T}
    end    
    g::Vector{T}
end

function FloatTargetTerm{T, S}(Pcalc, Pobs, Wobs, α, N, M) where {T, S}
    Wnan = @. !ismissing(Pobs) & !ismissing(Wobs)
    data = (
        Pcalc = Pcalc,
        Pobs = build_rate_matrix(T, Pobs, N, M),
        Wobs = build_rate_matrix(T, Wobs, N, M),
        Wnan = (permutedims∘reshape)(Wnan, M, N),
    )
    params, pviews = params_and_views(TargetTermParameters, data)
    FloatTargetTerm{T, S}(; params, pviews, α)
end

function BoolTargetTerm{T, S}(Pcalc, Pobs, α, N, M) where {T, S}
    Wnan = @. !ismissing(Pobs)
    data = (
        Pcalc = Pcalc,
        Pobs = build_rate_matrix(T, Pobs, N, M),
        Wobs = falses(N, M),
        Wnan = (permutedims∘reshape)(Wnan, M, N),
    )
    params, pviews = params_and_views(TargetTermParameters, data)
    BoolTargetTerm{T, S}(; params, pviews, α)
end

function TargetFunction{T}(df_rates::AbstractDataFrame, prob::NonlinearProblem, fset::FittingSet, α) where {T}
    
    @unpack Pcalc, Pbhp, Pinj = prob.params

    Nt = (length∘unique)(df_rates.Tank::Vector{String})
    Nd = (length∘unique)(df_rates.Date::Vector{Date})

    terms = @with df_rates begin (
        Pres = FloatTargetTerm{T, ^(:Pres)}(Pcalc, :Pres, :Wres, α.Pres, Nt, Nd),
        Pbhp = FloatTargetTerm{T, ^(:Pbhp)}(Pbhp, :Pbhp_prod, :Wbhp_prod, α.Pbhp, Nt, Nd),
        Pinj = FloatTargetTerm{T, ^(:Pinj)}(Pinj, :Pbhp_inj, :Wbhp_inj, α.Pinj, Nt, Nd),
        Pmin = BoolTargetTerm{T, ^(:Pmin)}(Pcalc, :Pres_min, α.Pmin, Nt, Nd),
        Pmax = BoolTargetTerm{T, ^(:Pmax)}(Pcalc, :Pres_max, α.Pmax, Nt, Nd),
        L2 = L2TargetTerm{T}(fset, α.L2),
    ) end

    g = Array{T}(undef, Nt)
    TargetFunction{T}(; terms, g)
end

function update_targ!(targ::TargetFunction)
    update_term!(targ.terms.Pmin)
    update_term!(targ.terms.Pmax)
    return targ
end

function update_term!(term::BoolTargetTerm{T, :Pmin}) where {T}
    @unpack Pcalc, Wobs, Pobs = term.params
    @. Wobs = Pcalc < Pobs
    return term
end

function update_term!(term::BoolTargetTerm{T, :Pmax}) where {T}
    @unpack Pcalc, Wobs, Pobs = term.params
    @. Wobs = Pcalc > Pobs
    return term
end

function gradP!(g, term::AbstractTargetTerm, params::TargetTermParameters)
    @unpack Pcalc, Pobs, Wobs, Wnan = params
    @. g = Wnan * (2 * term.α * Wobs * (Pcalc - Pobs))
    return g
end

function gradP!(g, targ::TargetFunction, nparams::NTuple{N, TargetTermParameters}) where {N}
    (term, params), rest = Iterators.peel(zip(targ.terms, nparams))
    gradP!(g, term, params)
    for (term, params) ∈ rest
        gradP!(targ.g, term, params)
        g .+= targ.g
    end
    return g
end

getvalue(targ::TargetFunction) = sum(getvalue, targ.terms)
getvalues(targ::TargetFunction) = map(getvalue, targ.terms)
function getvalue(term::L2TargetTerm)
    @unpack xoptim, α = fset
    return term.α * @_ sum(_[1] * _[2]^2, zip(α, xoptim))
end
function getvalue(term::AbstractTargetTerm)
    @unpack Pcalc, Pobs, Wobs, Wnan = term.params
    itr = zip(Wnan, Wobs, Pcalc, Pobs)
    return term.α * @_ sum(_[1] * (_[2] * (_[3] - _[4])^2), itr)
end