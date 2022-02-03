Base.@kwdef struct FittingParameter{T, S} <: AbstractFittingParameter{T}
    xviews::VectorView{T}
    pviews::Vector{RowRange{T}}
    yviews::CartesianView{T}
end

Base.@kwdef struct FittingSet{T<:AbstractFloat, P<:Vector{<:AbstractFittingParameter{T}}}
    params::P
    xmodel::Vector{T}
    xoptim::Vector{T}
    xrange::Vector{T}
    xmin::Vector{T}
    α::Vector{T}
end

function build_fitting_views(df::AbstractDataFrame, prob::NonlinearProblem, sym::Symbol)
    P = getfield(prob.params, sym)
    @with df begin (
        pviews = view.(Ref(P), :Istart, UnitRange.(:Jstart, :Jstop)),
        yviews = view(P, CartesianIndex.(:Istart, :Jstart)),
    ) end
end

function FittingSet{T}(df_params::AbstractDataFrame, prob::NonlinearProblem{T}) where {T}
    
    # Вектор параметров для настройки
    N = sum(.!df_params.Skip::Vector{Bool})
    xmodel = Array{T}(undef, N)
    xoptim = Array{T}(undef, N)

    # Масштабные множители для параметров
    df = @view df_params[df_params.Skip .=== false, :]
    xrange = @with(df, :Max_value - :Min_value)
    xmin = @with(df, @. -:Min_value / xrange)
    α = df[:, :alpha]

    # Разбивка по параметрам, требующих настройки (Skip == false)
    gd = @chain df_params begin
        @subset(:Skip .=== false)
        @select(:Parameter, :Istart, :Jstart, :Jstop)            
        groupby(_, :Parameter)
    end

    # Ссылки (views) для настраиваемых параметров
    stop = 0
    params = map(pairs(gd)) do (key, df)
        # Название параметра
        sym = key.Parameter

        # Срез глобального вектора параметров
        start = stop + 1
        stop = start + nrow(df) - 1        
        xviews = view(xmodel, start:stop)

        # Ссылки на параметры внутри модели
        pviews, yviews = build_fitting_views(df, prob, sym)
        FittingParameter{T, sym}(; xviews, pviews, yviews)
    end

    FittingSet{T, typeof(params)}(; params, xmodel, xoptim, xrange, xmin, α)
end

function getparams(fset::FittingSet)
    @unpack xmodel, xoptim, xrange, xmin = fset
    for params ∈ fset.params
        @unpack xviews, yviews = params        
        copyto!(xviews, yviews)        
    end
    @. xoptim = xmin + xmodel / xrange
    return xoptim
end

function getparams!(x, fset::FittingSet)
    copyto!(x, getparams(fset))    
end

function setparams!(fset::FittingSet, xnew)
    @unpack xmodel, xoptim, xrange, xmin = fset    
    copyto!(fset.xoptim, xnew)
    @. xmodel = (xoptim - xmin) * xrange
    for params ∈ fset.params
        @unpack xviews, pviews = params        
        fill!.(pviews, xviews)
    end
    return xnew
end