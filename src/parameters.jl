Base.@kwdef struct ParameterValidity{A<:AbstractArray{<:Real}}
    V::A
end

Base.@kwdef struct FittingParameter{S, T} <: AbstractFittingParameter{T} 
    xviews::VectorRange{T}                              # ссылки на частичный диапазон глобального вектора параметров
    pviews::Vector{RowRange{T}}                         # ссылки на внутренний массив параметров модели (на весь периода закрепеления)
    yviews::CartesianView{T}                            # ссылки на внутренний массив параметров модели (на начало периода закрепеления)
    valids::ParameterValidity{BitMatrix}                # массив периодов закрепления параметра
    vviews::Vector{ParameterValidity{ColumnSliceBool}}  # ссылки на столбцы массива периода действия параметра
    gviews::VectorRange{T}                              # ссылки на частичный диапазон глобального вектора градиента
    bviews::VectorView{T}                               # ссылки на расчетный буфер 1
    bviews2::VectorView{T}                              # ссылки на расчетный буфер 2
end

Base.@kwdef struct FittingCache{T<:AbstractFloat}
    xbuf::Vector{T}
    ybuf::Vector{T}
    gbuf::Vector{T}
    tbuf::Vector{T}
    tbuf2::Vector{T}
    cbuf::Vector{T}
    cbuf2::Vector{T}
end

Base.@kwdef struct FittingSet{T<:AbstractFloat, PS<:AbstractParametersScaling{T}, TP<:Tuple{Vararg{<:AbstractFittingParameter{T}}}}
    params::TP
    scale::PS
    α::Vector{T}
    cache::FittingCache{T}
end

function build_fitting_views(df::AbstractDataFrame, prob::NonlinearProblem, ::Val{S}) where {S}
    P = getfield(prob.params, S)
    @with df begin (
        pviews = view.(Ref(P), :Istart, UnitRange.(:Jstart, :Jstop)),
        yviews = view(P, CartesianIndex.(:Istart, :Jstart)),
    ) end
end

function build_validity_matrix(df::AbstractDataFrame, N, M)
    V = falses(N, M)
    for (v, dfr) ∈ zip(eachrow.((V, df))...)
        v[UnitRange(dfr.Jstart, dfr.Jstop)] .= true
    end
    params_and_views(ParameterValidity, (V = V,))
end

function FittingCache{T}(Nt, Nc, Nx) where {T}
    kwargs = (
        xbuf = Array{T}(undef, Nx),
        ybuf = Array{T}(undef, Nx),
        gbuf = Array{T}(undef, Nx),
        tbuf = Array{T}(undef, Nt),
        tbuf2 = Array{T}(undef, Nt),
        cbuf = Array{T}(undef, Nc),
        cbuf2 = Array{T}(undef, Nc),
    )
    FittingCache{T}(kwargs...)
end

function FittingParameter{S, T}(df::AbstractDataFrame, prob::NonlinearProblem{T}, cache::FittingCache{T}, rng) where {S, T}
    
    # Кол-во временных шагов
    Nd = length(prob.pviews)

    # Расчетные буферы
    idx = df[:, :Istart]
    if S === :Tconn 
        bviews = view(cache.cbuf, idx)
        bviews2 = view(cache.cbuf2, idx)
    else
        bviews = view(cache.tbuf, idx)
        bviews2 = view(cache.tbuf2, idx)
    end

    # Ссылки на глобальные векторы
    xviews = view(cache.ybuf, rng)
    gviews = view(cache.gbuf, rng)        

    # Ссылки на параметры внутри модели
    pviews, yviews = build_fitting_views(df, prob, Val(S))
    # Периоды закрепления параметров
    valids, vviews = build_validity_matrix(df, nrow(df), Nd)

    FittingParameter{S, T}(; pviews, yviews, valids, vviews, bviews, bviews2, xviews, gviews)
end

function FittingSet{T}(df_params::AbstractDataFrame, prob::NonlinearProblem{T}, scale::AbstractParametersScaling{T}) where {T}

    # Число соединений и блоков
    Nc, Nt = size(prob.C)

    # Фильтруем параметры, требующие настройки (Skip == false)
    df = @view df_params[df_params.Skip .=== false, :]
    cache = FittingCache{T}(Nt, Nc, nrow(df))
    α = df[:, :alpha]
    
    stop = 0
    # FIXED: Сам по себе 'map' для 'GroupedDataFrame' 
    # невозможен (reserved), но можно по 'pairs(..)'
    gd = groupby(df, :Parameter)
    params = map(pairs(gd)) do (key, df)
        sym = key.Parameter
        start = stop + 1
        stop += nrow(df)
        FittingParameter{sym, T}(df, prob, cache, start:stop)        
    end |> Tuple

    FittingSet{T, typeof(scale), typeof(params)}(; params, scale, α, cache)
end

function getparams!(fset::FittingSet)
    @unpack params, scale = fset
    @unpack xbuf, ybuf = fset.cache

    # FIXED: Использование 'map' вместо 'for' сохраняет 'type-stability'
    map(params) do param
        @unpack xviews, yviews = param    
        copyto!(xviews, yviews)        
    end
    scalex!(xbuf, ybuf, scale)

    return xbuf
end

function getparams!!(x, fset::FittingSet)
    copyto!(x, getparams!(fset))    
end

function setparams!(fset::FittingSet, xnew)    
    @unpack params, scale = fset
    @unpack xbuf, ybuf = fset.cache

    copyto!(xbuf, xnew)
    unscalex!(ybuf, xbuf, scale)

    # FIXED: Использование 'map' вместо 'for' сохраняет 'type-stability'
    map(params) do param
        @unpack xviews, pviews = param
        # FIXED: Быстрее, чем 'fill!.(pviews, xviews)'
        @inbounds @simd for i = 1:length(xviews)
            fill!(pviews[i], xviews[i])
        end
    end

    return ybuf
end