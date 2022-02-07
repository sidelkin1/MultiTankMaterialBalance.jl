Base.@kwdef struct PresTargetTerm{S, T} <: AbstractTargetTerm{T}
    Pobs::Vector{T}
    Wobs::Vector{T}
    Wnan::BitVector
    ΔP::Vector{T}
    g::Matrix{T}
    gviews::Vector{ColumnSlice{T}}
    α::T
end

Base.@kwdef struct PresBoundTerm{S, T} <: AbstractTargetTerm{T}
    Pobs::Vector{T}
    Wnan::BitVector
    ΔP::Vector{T}
    g::Matrix{T}
    gviews::Vector{ColumnSlice{T}}
    α::T
end

Base.@kwdef struct BHPTargetTerm{S, T} <: AbstractTargetTerm{T}
    Pobs::Vector{T}
    Wnan::BitVector
    ΔP::Vector{T}
    B::SparseMatrixCSC{T, Int}
    W::SparseMatrixCSC{T, Int}
    g::Matrix{T}
    gviews::Vector{ColumnSlice{T}}
    α::T
end

Base.@kwdef struct BHPBoundTerm{S, T} <: AbstractTargetTerm{T}
    Wnan::BitVector 
    J⁻¹::Vector{T}
    J⁻¹min::Vector{T}
    J⁻¹max::Vector{T}
    ΔJ⁻¹min::Vector{T}
    ΔJ⁻¹max::Vector{T}
    ΔJ⁻¹sum::Vector{T}    
    g::Matrix{T}
    gviews::Vector{ColumnSlice{T}}
    α::T
end

Base.@kwdef struct L2TargetTerm{T} <: AbstractTargetTerm{T}
    x::Vector{T}
    αₓ::Vector{T}
    α::T
end

Base.@kwdef struct FracInjectionTerm{T}
    λviews::CartesianView{T}
    J⁻¹min::Vector{T}
    J⁻¹max::Vector{T}
    gbuf::Vector{T}
    gviews::VectorView{T}
    idx::Vector{Int}
end

Base.@kwdef struct WellIndexViews{S, T<:AbstractFloat}
    xviews::VectorView{T}
    pviews::Vector{RowRange{T}}
    yviews::CartesianView{T}
end

Base.@kwdef struct TargetFunction{T<:AbstractFloat}
    terms::@NamedTuple begin
        Pres::PresTargetTerm{:Pres, T}        
        Pmin::PresBoundTerm{:Pmin, T}
        Pmax::PresBoundTerm{:Pmax, T}
        Pbhp::BHPTargetTerm{:Pbhp, T}
        Jp::BHPBoundTerm{:Jp, T}
        Pinj::BHPTargetTerm{:Pinj, T}
        Jinj::BHPBoundTerm{:Jinj, T}
        L2::L2TargetTerm{T}
    end
    wviews::@NamedTuple begin
        Jp::WellIndexViews{:Jp, T}
        Jinj::WellIndexViews{:Jinj, T}
        Gw::WellIndexViews{:Gw, T}
    end
    finj::FracInjectionTerm{T}
    prob::NonlinearProblem{T}
end

function build_bhp_matrices(df::AbstractDataFrame, q, w, N, M)

    # Разбивка дебитов на итервалы закрепления, представленных
    # в виде исходного вектора ('src') и в виде матрицы ('dst')
    src, dst = @with df begin
        rng = UnitRange.(:Jstart, :Jstop)        
        getindex.(Ref(LinearIndices((M, N))), rng, :Istart),
        getindex.(Ref(LinearIndices((N, M))), :Istart, rng)
    end    

    # Заполнение системной матрицы дебитов
    Q = spzeros(eltype(q), length(q), nrow(df))    
    @_ map(_1[_3] .= q[_2], eachcol(Q), src, dst)
    # Переупорядочивание вектора весов
    Dw = similar(w, nonmissingtype(eltype(w)))    
    w_ = Missings.replace(w, zero(eltype(Dw)))
    # TODO: Используется 'getindex', т.к. 'Missings.replace' 
    # работает только для скалярных индексов
    @_ map(Dw[_2] .= getindex.(Ref(w_), _1), src, dst)
    
    # Матрица для расчет Кпрод по взвешенному МНК (J = B * ΔP)
    # TODO: Используется QR-разложение с учетом весов,
    # почему-то не работает с 'spdiagm', 'Diagonal'  в правой части
    B = sparse((Diagonal(.√Dw) * Q) \ diagm(.√Dw))
    # Матрица для расчета взвешенной суммы квадратов остатков
    W = dropzeros((I - Q * B)' * Diagonal(Dw) * (I - Q * B))
    
    return B, W
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

function FracInjectionTerm{T}(df::AbstractDataFrame, λ) where {T}

    # Выборка нужных параметров
    df_inj = @view df[df.Parameter .=== :Jinj, :]
    df_λ = @view df[df.Parameter .=== :λ, :]

    # Ссылки на коэффициент λ
    yviews = @with(df_λ, view(λ, CartesianIndex.(:Istart, :Jstart)))    
    λviews = @with(df_inj, view(yviews, :Link))

    # Интервалы допустимых значений обратных значений Кпрод
    J⁻¹min, J⁻¹max = @with(df_inj, (inv.(:Max_value), inv.(:Min_value)))

    # Градиент относительно λ
    gbuf = Array{T}(undef, nrow(df_inj))
    lnk = @with(df_inj, .!df_λ.Const[:Link])
    gviews = view(gbuf, axes(df_inj, 1)[lnk])

    # Привязка λ к индексам глобального градиента
    df_view = @view df[(df.Parameter .∉ Ref((:Gw, :Jinj, :Jp))) .& .!df.Const, :]
    gλ = axes(df_view, 1)[df_view.Parameter .=== :λ]

    # Индексы для обновления глобального градиента
    idx =  @with(df_inj, :Link[lnk])
    nums = collect(pairs(unique(idx)))
    replace!(idx, reverse.(nums)...)    
    idx = isempty(gλ) ? gλ : gλ[idx]

    return FracInjectionTerm{T}(; λviews, J⁻¹min, J⁻¹max, gbuf, gviews, idx)
end

function PresTargetTerm{S, T}(Pobs, Wobs, α, N, M) where {S, T}
    idx = vec(LinearIndices((M, N))')
    params = (
        Pobs = replace(Pobs[idx], missing => T(NaN)),
        Wobs = replace(Wobs[idx], missing => zero(T)),
        Wnan = .!ismissing.(Pobs[idx]),
        ΔP = Array{T}(undef, length(idx)),
        g = Array{T}(undef, N, M),
    )
    gviews = collect(eachcol(params.g))
    PresTargetTerm{S, T}(; gviews, α, params...)
end

function PresBoundTerm{S, T}(Pobs, α, N, M) where {S, T}
    idx = vec(LinearIndices((M, N))')
    params = (
        Pobs = replace(Pobs[idx], missing => T(NaN)),
        Wnan = .!ismissing.(Pobs[idx]),
        ΔP = Array{T}(undef, length(idx)),
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

    # Параметры для расчета Кпрод
    params = (
        Pobs = replace(Pobs[idx], missing => T(NaN)),
        Wnan = .!ismissing.(Pobs[idx]),
        ΔP = Array{T}(undef, length(idx)),        
        g = Array{T}(undef, N, M),
    )
    gviews = collect(eachcol(params.g))
    B, W = build_bhp_matrices(df_view, Q, Wobs, N, M)
    
    BHPTargetTerm{S, T}(; B, W, gviews, α, params...)
end

function BHPBoundTerm{S, T}(df::AbstractDataFrame, α, N, M) where {S, T}

    # Список всех значений Кпрод
    name = S === :Jp ? :Gw : :Jinj
    df_view = @view df[df.Parameter .=== name, :]

    # Параметры для расчета Кпрод
    params = (      
        Wnan = .!df_view.Ignore,  
        J⁻¹ = Array{T}(undef, nrow(df_view)),
        ΔJ⁻¹min = Array{T}(undef, nrow(df_view)),
        ΔJ⁻¹max = Array{T}(undef, nrow(df_view)),
        ΔJ⁻¹sum = Array{T}(undef, nrow(df_view)),
        g = Array{T}(undef, N, M),
    )
    gviews = collect(eachcol(params.g))
    J⁻¹min, J⁻¹max = @with(df_view, (inv.(:Max_value), inv.(:Min_value)))

    BHPBoundTerm{S, T}(; J⁻¹min, J⁻¹max, gviews, α, params...)
end

function L2TargetTerm{T}(df::AbstractDataFrame, x, α) where {T}
    df_view = @view df[(df.Parameter .∉ Ref((:Gw, :Jinj, :Jp))) .& .!df.Const, :]
    αₓ = replace(df_view.alpha, missing => zero(T))
    return L2TargetTerm{T}(; x, αₓ, α)
end

function TargetFunction{T}(df_rates::AbstractDataFrame, df_params::AbstractDataFrame, prob::NonlinearProblem{T}, fset::FittingSet{T}, α) where {T}
    
    Nt = size(prob.C, 2)
    Nd = length(prob.pviews)

    terms = @with df_rates begin (
        Pres = PresTargetTerm{^(:Pres), T}(:Pres, :Wres, α["alpha_resp"], Nt, Nd),
        Pmin = PresBoundTerm{^(:Pmin), T}(:Pres_min, α["alpha_lb"], Nt, Nd),
        Pmax = PresBoundTerm{^(:Pmax), T}(:Pres_max, α["alpha_ub"], Nt, Nd),
        Pbhp = BHPTargetTerm{^(:Pbhp), T}(df_params, :Pbhp_prod, :Wbhp_prod, :Qliq ./ :Total_mobility, α["alpha_bhp"], Nt, Nd),
        Jp = BHPBoundTerm{^(:Jp), T}(df_params, α["alpha_bhp_bound"], Nt, Nd),
        Pinj = BHPTargetTerm{^(:Pinj), T}(df_params, :Pbhp_inj, :Wbhp_inj, .-:Qinj, α["alpha_inj"], Nt, Nd),
        Jinj = BHPBoundTerm{^(:Jinj), T}(df_params, α["alpha_inj_bound"], Nt, Nd),
        L2 = L2TargetTerm{T}(df_params, fset.cache.ybuf, α["alpha_l2"]),
    ) end

    wviews = (
        Jp = WellIndexViews{:Jp, T}(df_params, prob, terms.Jp.J⁻¹),
        Jinj = WellIndexViews{:Jinj, T}(df_params, prob, terms.Jinj.J⁻¹),
        Gw = WellIndexViews{:Gw, T}(df_params, prob, terms.Jp.J⁻¹)
    )
    
    finj = FracInjectionTerm{T}(df_params, prob.params.λ)

    TargetFunction{T}(; terms, wviews, finj, prob)
end

function update_targ!(targ::TargetFunction)
    update_term!(targ.finj, targ)
    terms = @inbounds values(targ.terms)[1:end-1]
    # FIXED: Использование 'map' вместо 'for' сохраняет 'type-stability'
    map(term -> update_term!(term, targ), terms)
    return targ
end

function update_term!(term::FracInjectionTerm, targ::TargetFunction)
    @unpack λviews, J⁻¹min, J⁻¹max  = term
    @unpack "ₐ", J⁻¹min, J⁻¹max = targ.terms.Jinj
    @inbounds @simd for i = 1:length(λviews)
        J⁻¹minₐ[i] = λviews[i] * J⁻¹min[i]
        J⁻¹maxₐ[i] = λviews[i] * J⁻¹max[i]
    end 
end

function update_term!(term::PresTargetTerm{S, T}, targ::TargetFunction) where {S, T}
    @unpack Pcalc = targ.prob.params
    @unpack Wobs, Pobs, Wnan, ΔP, g, α = term
    @inbounds @simd for i = 1:length(Pcalc)
        ΔP[i] = Wnan[i] ? Pcalc[i] - Pobs[i] : zero(T)
        g[i] = Wnan[i] ? T(2) * α * Wobs[i] * ΔP[i] : zero(T)
    end
    return term
end

compare(Pcalc, Pobs, ::Val{:Pmin}) = Pcalc < Pobs
compare(Pcalc, Pobs, ::Val{:Pmax}) = Pcalc > Pobs

function update_term!(term::PresBoundTerm{S, T}, targ::TargetFunction) where {S, T}
    @unpack Pcalc = targ.prob.params
    @unpack Pobs, Wnan, ΔP, g, α = term
    @inbounds @simd for i = 1:length(Pcalc)
        Wobs = Wnan[i] & compare(Pcalc[i], Pobs[i], Val(S))
        ΔP[i] = Wobs ? Pcalc[i] - Pobs[i] : zero(T)
        g[i] = Wobs ? T(2) * α * ΔP[i] : zero(T)
    end
    return term
end

function update_term!(term::BHPTargetTerm{S, T}, targ::TargetFunction) where {S, T}
    @unpack Pcalc = targ.prob.params
    @unpack Pobs, Wnan, ΔP, W, g, α = term    
    @inbounds @simd for i = 1:length(Pcalc)
        ΔP[i] = Wnan[i] ? Pcalc[i] - Pobs[i] : zero(T)
    end
    mul!(vec(g), W, ΔP, T(2) * α, zero(T))
    return term
end

unpackbhp(targ::TargetFunction, ::Val{:Jp}) = targ.terms.Pbhp.B, targ.terms.Pbhp.ΔP
unpackbhp(targ::TargetFunction, ::Val{:Jinj}) = targ.terms.Pinj.B, targ.terms.Pinj.ΔP

function update_term!(term::BHPBoundTerm{S, T}, targ::TargetFunction) where {S, T}
    @unpack Wnan, J⁻¹, J⁻¹min, J⁻¹max = term 
    @unpack ΔJ⁻¹min, ΔJ⁻¹max, ΔJ⁻¹sum, g, α = term   
    # Оценка Кпрод
    B, ΔP = unpackbhp(targ, Val(S))
    mul!(J⁻¹, B, ΔP)
    # Проверка мин.\макс. ограничений
    @inbounds @simd for i = 1:length(J⁻¹)
        Wobs = Wnan[i] & !(J⁻¹min[i] ≤ J⁻¹[i] ≤ J⁻¹max[i])
        ΔJ⁻¹min[i] = Wobs ? J⁻¹[i] - J⁻¹min[i] : zero(T)
        ΔJ⁻¹max[i] = Wobs ? J⁻¹[i] - J⁻¹max[i] : zero(T)
        ΔJ⁻¹sum[i] = ΔJ⁻¹min[i] + ΔJ⁻¹max[i]
    end    
    mul!(vec(g), B', ΔJ⁻¹sum, α, zero(T))    
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
    @unpack ΔP, g = term
    val = zero(T)
    @inbounds @simd for i = 1:length(ΔP)
        val += ΔP[i] * g[i]
    end
    return T(0.5) * val
end

function getvalue(term::BHPBoundTerm{S, T}) where {S, T}
    @unpack ΔJ⁻¹min, ΔJ⁻¹max, α = term
    val = zero(T)
    @inbounds @simd for i = 1:length(ΔJ⁻¹min)
        val += ΔJ⁻¹min[i] * ΔJ⁻¹max[i]
    end
    return α * val
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
    calc_well_index!(targ.prob, targ.wviews.Gw)
    calc_well_index!(targ.prob, targ.wviews.Jinj)
end

function calc_well_index!(prob::NonlinearProblem{T}, wviews::WellIndexViews{:Gw, T}) where {T}
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

function calc_well_index!(prob::NonlinearProblem{T}, wviews::WellIndexViews{:Jinj, T}) where {T}
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