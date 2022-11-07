Base.@kwdef struct WellIndexSolver{T}
    J⁻¹::Vector{T}
    J⁻¹min::Vector{T}
    J⁻¹max::Vector{T}  
    B::SparseMatrixCSC{T, Int}    
    W::SparseMatrixCSC{T, Int}
    lb::BitVector
    ub::BitVector
    ΔPmin::Vector{T}
    ΔPmax::Vector{T}
    lbviews::VectorViewBool
    ubviews::VectorViewBool    
end

Base.@kwdef struct PresTargetTerm{T} <: AbstractTargetTerm{T}
    Pobs::Vector{T}
    Wobs::Vector{T}
    ΔP::Vector{T}
    g::Matrix{T}
    gviews::Vector{ColumnSlice{T}}
end

Base.@kwdef struct PresBoundTerm{S, T} <: AbstractTargetTerm{T}
    Pobs::Vector{T}
    Wobs::Vector{T}
    ΔP::Vector{T}
    g::Matrix{T}
    gviews::Vector{ColumnSlice{T}}
end

Base.@kwdef struct PbhpTargetTerm{T} <: AbstractTargetTerm{T}
    Pobs::Vector{T}  
    Wobs::Vector{T}
    ΔP::Vector{T}  
    g::Matrix{T}    
    gviews::Vector{ColumnSlice{T}}    
    solver::WellIndexSolver{T}
end

Base.@kwdef struct PinjTargetTerm{T} <: AbstractTargetTerm{T}
    Pobs::Vector{T}
    Wobs::Vector{T}
    ΔP::Vector{T}  
    g::Matrix{T}
    gλ::Matrix{T}
    gviews::Vector{ColumnSlice{T}}
    gλviews::Vector{ColumnSlice{T}}
    solver::WellIndexSolver{T}
    J⁻¹min_h::Vector{T}
    J⁻¹max_h::Vector{T}
    λviews::CartesianView{T} 
end

Base.@kwdef struct L2TargetTerm{T} <: AbstractTargetTerm{T}
    x::Vector{T}
    g::Vector{T}
    αₓ::Vector{T}
end

Base.@kwdef struct WellIndexViews{S, T<:AbstractFloat}
    xviews::VectorView{T}
    pviews::Vector{RowRange{T}}
    yviews::CartesianView{T}
    λviews::CartesianView{T}
end

Base.@kwdef struct TargetFunction{T<:AbstractFloat}
    terms::@NamedTuple begin
        Pres::PresTargetTerm{T}        
        Pbhp::PbhpTargetTerm{T}
        Pinj::PinjTargetTerm{T}
        Pmin::PresBoundTerm{:Pmin, T}
        Pmax::PresBoundTerm{:Pmax, T}
        L2::L2TargetTerm{T}
    end
    wviews::@NamedTuple begin
        Jp::WellIndexViews{:Jp, T}
        Jinj::WellIndexViews{:Jinj, T}
        Gw::WellIndexViews{:Gw, T}
    end
    prob::NonlinearProblem{T}
end

function WellIndexSolver{T}(df::AbstractDataFrame, Qobs, Wobs, shape, α) where {T}

    # Indices of fixing intervals
    idx = @with df begin
        rng = UnitRange.(:Jstart, :Jstop)        
        getindex.(Ref(LinearIndices(shape)), :Istart, rng)
    end
    
    # Fill in system matrix of rates (Q ⋅ J⁻¹ = ΔP)
    Q = spzeros(T, length(Qobs), nrow(df))    
    @_ map(_1[_2] .= Qobs[_2], eachcol(Q), idx)

    # Diagonal weight matrix
    Dw = Diagonal(replace(Wobs, missing => zero(T)))
    
    # Matrix for calculating reciprocal PI (Ĵ⁻¹ = B ⋅ ΔP)
    B = dropzeros(Q' * Dw * Q)
    map!(inv, B.nzval, B.nzval)
    B = dropzeros(B * Q' * Dw)

    # Matrix for calculating weighted sum of squared residuals (β = ΔPᵀ ⋅ W ⋅ ΔP)
    lmul!(T(2) * α, Dw)
    W = I - Q * B
    W = dropzeros(W' * Dw * W)

    # Parameters for calculating PI
    params = (      
        J⁻¹ = Array{T}(undef, nrow(df)),
        lb = falses(nrow(df)),
        ub = falses(nrow(df)),
        J⁻¹min = inv.(df.Max_value),
        J⁻¹max = inv.(df.Min_value),
        ΔPmin = Q * inv.(df.Max_value),
        ΔPmax = Q * inv.(df.Min_value),
    )
    
    # References to flags that indicate that PI (ΔP) is out of bounds [lb, ub]    
    nums = Array{Int}(undef, length(Qobs))
    @_ map(nums[_1] .= _2, idx, axes(df, 1))
    lbviews = view(params.lb, nums)
    ubviews = view(params.ub, nums)

    return WellIndexSolver{T}(; B, W, lbviews, ubviews, params...)
end

function WellIndexViews{S, T}(df::AbstractDataFrame, prob::NonlinearProblem{T}, J⁻¹) where {S, T}
    
    # PI matrix
    J = getfield(prob.params, S)
    
    # References to fitting elements of the vector of reciprocal PI
    df_S = getparams(df, Val(S))
    xviews = view(J⁻¹, axes(df_S, 1)[.!df_S.Ignore])

    # References to fitting elements of the PI matrix
    df_S = getparams(df, Val(S), Val(:var))
    pviews, yviews = @with df_S begin
        view.(Ref(J), :Istart, UnitRange.(:Jstart, :Jstop)),
        view(J, CartesianIndex.(:Istart, :Jstart))
    end

    # References to injection efficiency factor (needed only for S == Jinj)
    idx = replace(df_S.Link, missing => 1)
    df_λ = view(getparams(df, Val(:λ)), idx, :)
    λviews = @with df_λ begin
        view(prob.params.λ, CartesianIndex.(:Istart, :Jstart))
    end

    return WellIndexViews{S, T}(; xviews, pviews, yviews, λviews)
end

function PresTargetTerm{T}(df::AbstractDataFrame, idx, α) where {T}
    Pobs, Wobs = @with df begin
        :Pres[vec(idx)], :Wres[vec(idx)]
    end
    params = (
        Pobs = replace(Pobs, missing => zero(T)),
        Wobs = T(2) .* α .* replace(Wobs, missing => zero(T)),
        ΔP = Array{T}(undef, length(idx)),
        g = Array{T}(undef, size(idx)),
    )
    gviews = collect(eachcol(params.g))
    PresTargetTerm{T}(; gviews, params...)
end

getpobs(df, idx, ::Val{:Pmin}) = @with(df, :Pres_min[idx])
getpobs(df, idx, ::Val{:Pmax}) = @with(df, :Pres_max[idx])

function PresBoundTerm{S, T}(df::AbstractDataFrame, idx, α) where {S, T}
    Pobs = getpobs(df, vec(idx), Val(S))
    params = @with df begin (
        Pobs = replace(Pobs, missing => zero(T)),
        Wobs = T(2) .* α .* .!ismissing.(Pobs),
        ΔP = Array{T}(undef, length(idx)),
        g = Array{T}(undef, size(idx)),
    ) end
    gviews = collect(eachcol(params.g))
    PresBoundTerm{S, T}(; gviews, params...)
end

function PbhpTargetTerm{T}(df_rates::AbstractDataFrame, df_params::AbstractDataFrame, idx, α) where {T}

    # List of all values of PI
    df_view = getparams(df_params, Val(:Gw))
    Pobs, Wobs, Qobs = @with df_rates begin
        :Pbhp_prod[vec(idx)], 
        :Wbhp_prod[vec(idx)], 
        (:Qliq ./ :Total_mobility)[vec(idx)]
    end
    
    # Solver for PI
    solver = WellIndexSolver{T}(df_view, Qobs, Wobs, size(idx), α)
   
    # Parameters for calculating PI
    params = (
        Pobs = replace(Pobs, missing => zero(T)),
        Wobs = T(2) .* α .* replace(Wobs, missing => zero(T)),
        ΔP = Array{T}(undef, length(idx)), 
        g = Array{T}(undef, size(idx)),
    )
    gviews = collect(eachcol(params.g))
    
    PbhpTargetTerm{T}(; gviews, solver, params...)
end

function PinjTargetTerm{T}(df_rates::AbstractDataFrame, df_params::AbstractDataFrame, prob::NonlinearProblem{T}, idx, α) where {T}

    # List of all values of PI
    df_view = getparams(df_params, Val(:Jinj))
    Pobs, Wobs, Qobs = @with df_rates begin
        :Pbhp_inj[vec(idx)],
        :Wbhp_inj[vec(idx)],
        .-:Qinj[vec(idx)]
    end

    # Solver for PI
    solver = WellIndexSolver{T}(df_view, Qobs, Wobs, size(idx), α)

    # Parameters needed for calculating PI
    params = (
        Pobs = replace(Pobs, missing => zero(T)),
        Wobs = T(2) .* α .* replace(Wobs, missing => zero(T)),
        ΔP = Array{T}(undef, length(idx)), 
        g = Array{T}(undef, size(idx)),
        gλ = Array{T}(undef, size(idx)),
        J⁻¹min_h = copy(solver.J⁻¹min),
        J⁻¹max_h = copy(solver.J⁻¹max),
    )
    gviews = collect(eachcol(params.g))
    gλviews = collect(eachcol(params.gλ))
    
    # References to injection efficiency factors
    df_λ = view(getparams(df_params, Val(:λ)), df_view.Link, :)
    λviews = @with df_λ begin
        view(prob.params.λ, CartesianIndex.(:Istart, :Jstart))
    end

    PinjTargetTerm{T}(; gviews, gλviews, solver, λviews, params...)
end

function L2TargetTerm{T}(df::AbstractDataFrame, x, α) where {T}
    df_view = getparams(df, Val(:tanks), Val(:var))
    params = (
        αₓ = T(2) .* α .* replace(df_view.alpha, missing => zero(T)),
        g = Array{T}(undef, length(x)),
    )
    return L2TargetTerm{T}(; x, params...)
end

function TargetFunction{T}(df_rates::AbstractDataFrame, df_params::AbstractDataFrame, prob::NonlinearProblem{T}, fset::FittingSet{T}, α) where {T}
    
    # Linear indexing
    Nt = size(prob.C, 2)
    Nd = length(prob.pviews)
    idx = LinearIndices((Nd, Nt))'

    # Separate terms of objective function
    terms = @with df_rates begin (
        Pres = PresTargetTerm{T}(df_rates, idx, α["alpha_resp"]),
        Pbhp = PbhpTargetTerm{T}(df_rates, df_params, idx, α["alpha_bhp"]),
        Pinj = PinjTargetTerm{T}(df_rates, df_params, prob, idx, α["alpha_inj"]),
        Pmin = PresBoundTerm{^(:Pmin), T}(df_rates, idx, α["alpha_lb"]),
        Pmax = PresBoundTerm{^(:Pmax), T}(df_rates, idx, α["alpha_ub"]),        
        L2 = L2TargetTerm{T}(df_params, fset.cache.ybuf, α["alpha_l2"]),
    ) end

    # References to well parameters
    wviews = (
        Jp = WellIndexViews{:Jp, T}(df_params, prob, terms.Pbhp.solver.J⁻¹),
        Jinj = WellIndexViews{:Jinj, T}(df_params, prob, terms.Pinj.solver.J⁻¹),
        Gw = WellIndexViews{:Gw, T}(df_params, prob, terms.Pbhp.solver.J⁻¹)
    )
   
    TargetFunction{T}(; terms, wviews, prob)
end

function update_targ!(targ::TargetFunction)
    # FIXED: Using 'map' instead of 'for' preserves 'type-stability'
    map(term -> update_term!(term, targ.prob), values(targ.terms))
    return targ
end

function update_term!(term::PresTargetTerm, prob::NonlinearProblem)
    @unpack Pcalc = prob.params
    @unpack Wobs, Pobs, ΔP, g = term
    @turbo for i ∈ eachindex(Pcalc)
        ΔP[i] = Pcalc[i] - Pobs[i]
        g[i] = Wobs[i] * ΔP[i]
    end
    return term
end

function update_term!(term::PresBoundTerm{:Pmin}, prob::NonlinearProblem)
    @unpack Pcalc = prob.params
    @unpack Pobs, Wobs, ΔP, g = term
    @turbo for i ∈ eachindex(Pcalc)
        ΔP[i] = Pcalc[i] - Pobs[i]
        g[i] = (Pcalc[i] < Pobs[i]) * Wobs[i] * ΔP[i]
    end
    return term
end

function update_term!(term::PresBoundTerm{:Pmax}, prob::NonlinearProblem)
    @unpack Pcalc = prob.params
    @unpack Pobs, Wobs, ΔP, g = term
    @turbo for i ∈ eachindex(Pcalc)
        ΔP[i] = Pcalc[i] - Pobs[i]
        g[i] = (Pcalc[i] > Pobs[i]) * Wobs[i] * ΔP[i]
    end
    return term
end

function solve!(g, solver::WellIndexSolver, ΔP)
    @unpack J⁻¹, J⁻¹min, J⁻¹max, B, W, lb, ub = solver
    mul!(J⁻¹, B, ΔP)    
    @turbo for i ∈ eachindex(J⁻¹)
        lb[i] = J⁻¹[i] < J⁻¹min[i]
        ub[i] = J⁻¹[i] > J⁻¹max[i]
        J⁻¹[i] = clamp(J⁻¹[i], J⁻¹min[i], J⁻¹max[i])
    end
    mul!(g, W, ΔP)
    return g
end

function update_term!(term::PbhpTargetTerm, prob::NonlinearProblem)
    @unpack Pcalc = prob.params
    @unpack Pobs, Wobs, ΔP, g = term
    @unpack W, ΔPmin, ΔPmax, lbviews, ubviews = term.solver
    @turbo for i ∈ eachindex(Pcalc)
        ΔP[i] = Pcalc[i] - Pobs[i]
    end
    solve!(vec(g), term.solver, ΔP)
    @inbounds @simd for i ∈ eachindex(ΔP)
        if lbviews[i]
            ΔP[i] -= ΔPmin[i]
            g[i] = Wobs[i] * ΔP[i]
        elseif ubviews[i]
            ΔP[i] -= ΔPmax[i]
            g[i] = Wobs[i] * ΔP[i]        
        end        
    end
    return term
end

function update_term!(term::PinjTargetTerm{T}, prob::NonlinearProblem) where {T}
    @unpack Pcalc, λ = prob.params
    @unpack Pobs, Wobs, ΔP, g = term
    @unpack gλ, J⁻¹min_h, J⁻¹max_h, λviews = term
    @unpack W, ΔPmin, ΔPmax, lbviews, ubviews, J⁻¹min, J⁻¹max = term.solver
    @turbo for i ∈ eachindex(Pcalc)
        ΔP[i] = Pcalc[i] - Pobs[i]
    end
    @inbounds @simd for i ∈ eachindex(λviews)
        J⁻¹min[i] = λviews[i] * J⁻¹min_h[i]
        J⁻¹max[i] = λviews[i] * J⁻¹max_h[i]
    end
    solve!(vec(g), term.solver, ΔP)
    @inbounds @simd for i ∈ eachindex(ΔP)        
        if lbviews[i]
            ΔP[i] -= λ[i] * ΔPmin[i]
            g[i] = Wobs[i] * ΔP[i]
            gλ[i] = -ΔPmin[i] * g[i]
        elseif ubviews[i]
            ΔP[i] -= λ[i] * ΔPmax[i]
            g[i] = Wobs[i] * ΔP[i]
            gλ[i] = -ΔPmax[i] * g[i]
        else            
            gλ[i] = zero(T)
        end        
    end
    return term
end

function update_term!(term::L2TargetTerm, prob::NonlinearProblem)
    @unpack g, x, αₓ = term
    @turbo for i ∈ eachindex(g)
        g[i] = αₓ[i] * x[i]
    end
end

function grad!(g, targ::TargetFunction{T}, n) where {T}    
    fill!(g, zero(T))
    terms = @inbounds values(targ.terms)[1:end-1]
    # FIXED: Using 'map' instead of 'for' preserves 'type-stability'
    map(term -> (g .+= @inbounds term.gviews[n]), terms)
    return g
end

getvalue(targ::TargetFunction) = sum(getvalue, values(targ.terms))
getvalues(targ::TargetFunction) = map(getvalue, targ.terms)

function getvalue(term::AbstractTargetTerm{T}) where {T}
    @unpack ΔP, g = term
    val = zero(T)
    @turbo for i ∈ eachindex(ΔP)
        val += ΔP[i] * g[i]
    end
    return T(0.5) * val
end

function getvalue(term::L2TargetTerm{T}) where {T}
    @unpack x, g = term
    val = zero(T)
    @turbo for i ∈ eachindex(x)
        val += x[i] * g[i]
    end
    return T(0.5) * val
end

function calc_well_index!(targ::TargetFunction)
    calc_well_index!(targ.prob, targ.wviews.Gw)
    calc_well_index!(targ.prob, targ.wviews.Jinj)
end

function calc_well_index!(prob::NonlinearProblem, wviews::WellIndexViews{:Gw})
    @unpack xviews, pviews = wviews
    @unpack Pcalc, Pbhp, Jp, M, Gw, Qliq_h = prob.params
    @inbounds @simd for i ∈ eachindex(xviews)
        fill!(pviews[i], inv(xviews[i]))
    end
    @turbo for i ∈ eachindex(Pcalc)
        Jp[i] = Gw[i] * M[i]
        Pbhp[i] = Pcalc[i] - Qliq_h[i] / Jp[i]
    end
end

function calc_well_index!(prob::NonlinearProblem, wviews::WellIndexViews{:Jinj})
    @unpack xviews, pviews, λviews = wviews
    @unpack Pcalc, Pinj, Jinj, Qinj_h, λ = prob.params
    @inbounds @simd for i ∈ eachindex(xviews)
        fill!(pviews[i], λviews[i] * inv(xviews[i]))
    end
    @turbo for i ∈ eachindex(Pcalc)        
        Pinj[i] = Pcalc[i] + λ[i] * Qinj_h[i] / Jinj[i]
    end
end