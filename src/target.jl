Base.@kwdef struct PresTargetTerm{S, T} <: AbstractTargetTerm{T}
    Pobs::Vector{T}
    Wobs::Vector{T}
    r::Vector{T}
    g::Matrix{T}
    gviews::Vector{ColumnSlice{T}}
    α::T
end

Base.@kwdef struct PresBoundTerm{S, T} <: AbstractTargetTerm{T}
    Pobs::Vector{T}
    Wobs::BitVector
    r::Vector{T}
    g::Matrix{T}
    gviews::Vector{ColumnSlice{T}}
    α::T
end

Base.@kwdef struct BHPTargetTerm{S, T} <: AbstractTargetTerm{T}
    Pobs::Vector{T}
    Wobs::Vector{T}
    r::Vector{T}  
    ΔP::Vector{T}
    ΔPlim::Vector{T}
    ΔPh::Vector{T}
    ΔPmin::Vector{T}
    ΔPmax::Vector{T}
    J⁻¹::Vector{T}
    B::SparseMatrixCSC{T, Int}    
    H::SparseMatrixCSC{T, Int}
    W::SparseMatrixCSC{T, Int}
    g::Matrix{T}
    gλ::Matrix{T}
    gviews::Vector{ColumnSlice{T}}
    gλviews::Vector{ColumnSlice{T}}
    α::T
end

Base.@kwdef struct L2TargetTerm{T} <: AbstractTargetTerm{T}
    x::Vector{T}
    αₓ::Vector{T}
    α::T
end

Base.@kwdef struct WellIndexViews{S, T<:AbstractFloat}
    xviews::VectorView{T}
    pviews::Vector{RowRange{T}}
    yviews::CartesianView{T}
end

Base.@kwdef struct TargetFunction{T<:AbstractFloat}
    terms::@NamedTuple begin
        Pres::PresTargetTerm{:Pres, T}        
        Pbhp::BHPTargetTerm{:Pbhp, T}
        Pinj::BHPTargetTerm{:Pinj, T}
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
    ΔPmin_h::Vector{T}
    ΔPmax_h::Vector{T}
end

function build_bhp_solver(df::AbstractDataFrame, q, w, N, M)

    # Разбивка дебитов на итервалы закрепления, представленных
    # в виде исходного вектора ('src') и в виде матрицы ('dst')
    src, dst = @with df begin
        rng = UnitRange.(:Jstart, :Jstop)        
        getindex.(Ref(LinearIndices((M, N))), rng, :Istart),
        getindex.(Ref(LinearIndices((N, M))), :Istart, rng)
    end    

    # Заполнение системной матрицы дебитов (Q * J⁻¹ = ΔP)
    Q = spzeros(eltype(q), length(q), nrow(df))    
    @_ map(_1[_3] .= q[_2], eachcol(Q), src, dst)

    # Переупорядочивание вектора весов
    Dw = similar(w, nonmissingtype(eltype(w)))    
    Dw[vcat(dst...)] .= replace(.√w, missing => zero(eltype(Dw)))
    
    # Матрица для расчет обратного Кпрод (Ĵ⁻¹ = B * ΔP)
    # TODO: Используется QR-разложение с учетом весов,
    # почему-то не работает с 'spdiagm', 'Diagonal'  в правой части
    B = sparse((Diagonal(Dw) * Q) \ diagm(Dw))

    # Матрица (hat matrix) для расчета депрессии/репрессии (ΔP̂ = H * ΔP)
    H = dropzeros(Q * B)
    W = dropzeros((I - H)' * Diagonal(Dw.^2) * (I - H))

    # Предельные значения депрессии/репрессии
    J⁻¹min, J⁻¹max = @with(df, (inv.(:Max_value), inv.(:Min_value)))
    if all(q .≥ 0)
        ΔPmin, ΔPmax = Q * J⁻¹min, Q * J⁻¹max
    else
        ΔPmin, ΔPmax = Q * J⁻¹max, Q * J⁻¹min
    end
    
    return B, H, W, ΔPmin, ΔPmax
end

function WellIndexViews{S, T}(df::AbstractDataFrame, prob::NonlinearProblem{T}, J⁻¹) where {S, T<:AbstractFloat}
    
    # Матрица Кпрод
    J = getfield(prob.params, S)
    
    # Ссылка на изменяемые элементы вектора обратных значений Кпрод
    df_view = @view df[df.Parameter .=== S, :]
    xviews = view(J⁻¹, axes(df_view, 1)[.!df_view.Ignore])

    # Ссылка на изменяемые элементы матрицы Кпрод
    df_view = @view df[(df.Parameter .=== S) .& .!df.Ignore, :]
    pviews, yviews = @with df_view begin
        view.(Ref(J), :Istart, UnitRange.(:Jstart, :Jstop)),
        view(J, CartesianIndex.(:Istart, :Jstart))
    end

    return WellIndexViews{S, T}(; xviews, pviews, yviews)
end

function PresTargetTerm{S, T}(Pobs, Wobs, α, N, M) where {S, T}
    idx = vec(LinearIndices((M, N))')
    params = (
        Pobs = replace(Pobs[idx], missing => zero(T)),
        Wobs = replace(Wobs[idx], missing => zero(T)),
        r = Array{T}(undef, length(idx)),
        g = Array{T}(undef, N, M),
    )
    gviews = collect(eachcol(params.g))
    PresTargetTerm{S, T}(; gviews, α, params...)
end

function PresBoundTerm{S, T}(Pobs, α, N, M) where {S, T}
    idx = vec(LinearIndices((M, N))')
    params = (
        Pobs = replace(Pobs[idx], missing => zero(T)),
        Wobs = .!ismissing.(Pobs[idx]),
        r = Array{T}(undef, length(idx)),
        g = Array{T}(undef, N, M),
    )
    gviews = collect(eachcol(params.g))
    PresBoundTerm{S, T}(; gviews, α, params...)
end

function BHPTargetTerm{S, T}(df::AbstractDataFrame, Pobs, Wobs, Q, α, N, M) where {S, T}

    # Индексы для переупорядочивания векторов
    idx = vec(LinearIndices((M, N))')

    # Список всех значений Кпрод
    name = S === :Pbhp ? :Gw : :Jinj
    df_view = @view df[df.Parameter .=== name, :]

    # Вспомогательные матрицы для расчета Рзаб
    B, H, W, ΔPmin, ΔPmax = build_bhp_solver(df_view, Q, Wobs, N, M)

    # Параметры для расчета Кпрод
    params = (
        Pobs = replace(Pobs[idx], missing => zero(T)),
        Wobs = replace(Wobs[idx], missing => zero(T)),
        r = Array{T}(undef, length(idx)),
        ΔP = Array{T}(undef, length(idx)),
        ΔPlim = Array{T}(undef, length(idx)),
        ΔPh = Array{T}(undef, length(idx)),        
        J⁻¹ = Array{T}(undef, nrow(df_view)),
        g = Array{T}(undef, N, M),
        gλ = Array{T}(undef, N, M),
    )
    gviews = collect(eachcol(params.g))
    gλviews = collect(eachcol(params.gλ))
    
    BHPTargetTerm{S, T}(; B, H, W, ΔPmin, ΔPmax, gviews, gλviews, α, params...)
end

function L2TargetTerm{T}(df::AbstractDataFrame, x, α) where {T}
    df_view = @view df[(df.Parameter .∉ Ref((:Gw, :Jinj, :Jp))) .& .!df.Const, :]
    αₓ = replace(df_view.alpha, missing => zero(T))
    return L2TargetTerm{T}(; x, αₓ, α)
end

function TargetFunction{T}(df_rates::AbstractDataFrame, df_params::AbstractDataFrame, prob::NonlinearProblem{T}, fset::FittingSet{T}, α) where {T<:AbstractFloat}
    
    Nt = size(prob.C, 2)
    Nd = length(prob.pviews)

    terms = @with df_rates begin (
        Pres = PresTargetTerm{^(:Pres), T}(:Pres, :Wres, α["alpha_resp"], Nt, Nd),
        Pbhp = BHPTargetTerm{^(:Pbhp), T}(df_params, :Pbhp_prod, :Wbhp_prod, :Qliq ./ :Total_mobility, α["alpha_bhp"], Nt, Nd),
        Pinj = BHPTargetTerm{^(:Pinj), T}(df_params, :Pbhp_inj, :Wbhp_inj, .-:Qinj, α["alpha_inj"], Nt, Nd),
        Pmin = PresBoundTerm{^(:Pmin), T}(:Pres_min, α["alpha_lb"], Nt, Nd),
        Pmax = PresBoundTerm{^(:Pmax), T}(:Pres_max, α["alpha_ub"], Nt, Nd),        
        L2 = L2TargetTerm{T}(df_params, fset.cache.ybuf, α["alpha_l2"]),
    ) end

    wviews = (
        Jp = WellIndexViews{:Jp, T}(df_params, prob, terms.Pbhp.J⁻¹),
        Jinj = WellIndexViews{:Jinj, T}(df_params, prob, terms.Pinj.J⁻¹),
        Gw = WellIndexViews{:Gw, T}(df_params, prob, terms.Pbhp.J⁻¹)
    )

    ΔPmin_h = copy(terms.Pinj.ΔPmin)
    ΔPmax_h = copy(terms.Pinj.ΔPmax)
    
    TargetFunction{T}(; terms, wviews, prob, ΔPmin_h, ΔPmax_h)
end

function update_targ!(targ::TargetFunction)

    @unpack λ = targ.prob.params
    @unpack ΔPmin_h, ΔPmax_h  = targ
    @unpack ΔPmin, ΔPmax = targ.terms.Pinj
    @inbounds @simd for i = 1:length(λ)
        ΔPmin[i] = λ[i] * ΔPmin_h[i]
        ΔPmax[i] = λ[i] * ΔPmax_h[i]
    end 

    terms = @inbounds values(targ.terms)[1:end-1]
    # FIXED: Использование 'map' вместо 'for' сохраняет 'type-stability'
    map(term -> update_term!(term, targ), terms)
    return targ
end

function update_term!(term::PresTargetTerm{S, T}, targ::TargetFunction) where {S, T}
    @unpack Pcalc = targ.prob.params
    @unpack Wobs, Pobs, r, g, α = term
    @inbounds @simd for i = 1:length(Pcalc)
        r[i] = Pcalc[i] - Pobs[i]
        g[i] = T(2) * α * Wobs[i] * r[i]
    end
    return term
end

compare(Pcalc, Pobs, ::Val{:Pmin}) = Pcalc < Pobs
compare(Pcalc, Pobs, ::Val{:Pmax}) = Pcalc > Pobs

function update_term!(term::PresBoundTerm{S, T}, targ::TargetFunction) where {S, T}
    @unpack Pcalc = targ.prob.params
    @unpack Pobs, Wobs, r, g, α = term
    @inbounds @simd for i = 1:length(Pcalc)
        W = Wobs[i] & compare(Pcalc[i], Pobs[i], Val(S))
        r[i] = Pcalc[i] - Pobs[i]
        g[i] = T(2) * W * α * r[i]
    end
    return term
end

function update_term!(term::BHPTargetTerm{S, T}, targ::TargetFunction) where {S, T}
    @unpack Pcalc = targ.prob.params
    @unpack Pobs, Wobs, r, ΔP, ΔPh, ΔPmin, ΔPmax, W, H, g, gλ, α = term
    @inbounds @simd for i = 1:length(Pcalc)
        ΔPh[i] = Pcalc[i] - Pobs[i]
    end
    mul!(ΔP, H, ΔPh)
    mul!(vec(g), W, ΔPh, T(2) * α, zero(T))
    @inbounds @simd for i = 1:length(r)
        # r[i] = ΔPh[i] - clamp(ΔP[i], ΔPmin[i], ΔPmax[i])                
        # g[i] = T(2) * Wobs[i] * α * r[i]
        r[i] = ΔP[i] < ΔPmin[i] ? ΔPh[i] - ΔPmin[i] : ΔPh[i]
        r[i] = ΔP[i] > ΔPmax[i] ? ΔPh[i] - ΔPmax[i] : r[i]
        g[i] = ΔP[i] < ΔPmin[i] ? T(2) * Wobs[i] * α * r[i] : g[i]
        g[i] = ΔP[i] > ΔPmax[i] ? T(2) * Wobs[i] * α * r[i] : g[i]
        gλ[i] = ΔP[i] < ΔPmin[i] ? -targ.ΔPmin_h[i] * g[i] : zero(T)
        gλ[i] = ΔP[i] > ΔPmax[i] ? -targ.ΔPmax_h[i] * g[i] : gλ[i]
    end
    return term
end

function grad!(g, targ::TargetFunction{T}, n) where {T}    
    fill!(g, zero(T))
    terms = @inbounds values(targ.terms)[1:end-1]
    # FIXED: Использование 'map' вместо 'for' сохраняет 'type-stability'
    map(term -> (g .+= @inbounds term.gviews[n]), terms)
    return g
end

getvalue(targ::TargetFunction) = sum(getvalue, values(targ.terms))
getvalues(targ::TargetFunction) = map(getvalue, targ.terms)

function getvalue(term::AbstractTargetTerm{T}) where {T}
    @unpack r, g = term
    val = zero(T)
    @inbounds @simd for i = 1:length(r)
        val += r[i] * g[i]
    end
    return T(0.5) * val
end

function getvalue(term::L2TargetTerm{T}) where {T}
    @unpack x, αₓ, α = term
    val = zero(T)
    @inbounds @simd for i = 1:length(x)
        val += αₓ[i] * x[i] * x[i]
    end
    return α * val
end

function calc_well_index!(targ::TargetFunction)
    calc_well_index!(targ.terms.Pbhp)
    calc_well_index!(targ.prob, targ.wviews.Gw)
    calc_well_index!(targ.terms.Pinj)
    calc_well_index!(targ.prob, targ.wviews.Jinj)
end

function calc_well_index!(term::BHPTargetTerm)    
    @unpack B, J⁻¹, ΔP, ΔPlim, ΔPmin, ΔPmax = term
    @inbounds @simd for i = 1:length(ΔP)
        ΔPlim[i] = ΔP[i] < ΔPmin[i] ? ΔPmin[i] : ΔP[i]
        ΔPlim[i] = ΔP[i] > ΔPmax[i] ? ΔPmax[i] : ΔPlim[i]
    end
    mul!(J⁻¹, B, ΔPlim)    
end

function calc_well_index!(prob::NonlinearProblem, wviews::WellIndexViews{:Gw})
    @unpack xviews, pviews = wviews
    @unpack Pcalc, Pbhp, Jp, M, Gw, Qliq_h = prob.params
    @inbounds @simd for i = 1:length(xviews)
        fill!(pviews[i], inv(xviews[i]))
    end
    @inbounds @simd for i = 1:length(Pcalc)
        Jp[i] = Gw[i] * M[i]
        Pbhp[i] = Pcalc[i] - Qliq_h[i] / Jp[i]
    end
end

function calc_well_index!(prob::NonlinearProblem, wviews::WellIndexViews{:Jinj})
    @unpack xviews, pviews = wviews
    @unpack Pcalc, Pinj, Jinj, Qinj_h, λ = prob.params
    @inbounds @simd for i = 1:length(xviews)
        fill!(pviews[i], inv(xviews[i]))
    end
    @inbounds @simd for i = 1:length(Pcalc)        
        Pinj[i] = Pcalc[i] + Qinj_h[i] / Jinj[i]
        Jinj[i] *= λ[i]
    end
end