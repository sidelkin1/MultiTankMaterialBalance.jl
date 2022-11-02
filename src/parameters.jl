Base.@kwdef struct FittingParameter{S, T} <: AbstractFittingParameter{T}    
    xviews::VectorRange{T}                              # references to a partial range of the global parameter vector
    pviews::Vector{RowRange{T}}                         # references to the internal array of model parameters (for the entire period of fixing)
    yviews::CartesianView{T}                            # references to the internal array of model parameters (at the beginning of the period of fixing)
    gviews::VectorRange{T}                              # references to a partial range of the global gradient vector
    bviews::VectorView{T}                               # references to auxiliary buffer 1
    bviews2::VectorView{T}                              # references to auxiliary buffer 1
    V::BitMatrix                                        # array of parameter fixing periods
    vviews::Vector{ColumnSliceBool}                     # references to columns of the array of the period of validity of the parameter
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
    cache::FittingCache{T}
end

function build_fitting_views(df::AbstractDataFrame, prob::NonlinearProblem, name::Symbol)
    P = getfield(prob.params, name)
    pviews, yviews = @with df begin
        view.(Ref(P), :Istart, UnitRange.(:Jstart, :Jstop)),
        view(P, CartesianIndex.(:Istart, :Jstart))
    end
    return pviews, yviews
end

function build_validity_matrix(df::AbstractDataFrame, N, M)
    V = falses(N, M)
    for (v, dfr) ∈ zip(eachrow.((V, df))...)
        v[UnitRange(dfr.Jstart, dfr.Jstop)] .= true
    end
    return V, collect(eachcol(V))
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
    
    # Number of time steps
    Nd = length(prob.pviews)

    # Auxiliary buffers
    idx = df[:, :Istart]
    if S === :Tconn 
        bviews = view(cache.cbuf, idx)
        bviews2 = view(cache.cbuf2, idx)
    else
        bviews = view(cache.tbuf, idx)
        bviews2 = view(cache.tbuf2, idx)
    end

    # References to global vectors
    xviews = view(cache.ybuf, rng)
    gviews = view(cache.gbuf, rng)        

    # References to internal parameters of model
    pviews, yviews = build_fitting_views(df, prob, S)
    # Parameter fixing periods
    V, vviews = build_validity_matrix(df, nrow(df), Nd)

    FittingParameter{S, T}(; xviews, pviews, yviews, gviews, bviews, bviews2, V, vviews)
end

function FittingSet{T}(df::AbstractDataFrame, prob::NonlinearProblem{T}, scale::AbstractParametersScaling{T}) where {T}

    # Number of connections and tanks
    Nc, Nt = size(prob.C)
    
    # Allocate memory for buffers
    df_view = getparams(df, Val(:tanks), Val(:var))
    cache = FittingCache{T}(Nt, Nc, nrow(df_view))
    
    stop = 0
    # FIXED: By itself 'map' for 'GroupedDataFrame'
    # is not possible (reserved), but possible by 'pairs(..)'
    gd = groupby(df_view, :Parameter)
    params = map(pairs(gd)) do (key, df)
        sym = key.Parameter
        start = stop + 1
        stop += nrow(df)
        FittingParameter{sym, T}(df, prob, cache, start:stop)        
    end |> Tuple

    FittingSet{T, typeof(scale), typeof(params)}(; params, scale, cache)
end

function getparams!(fset::FittingSet)
    @unpack params, scale = fset
    @unpack xbuf, ybuf = fset.cache

    # FIXED: Using 'map' instead of 'for' preserves 'type-stability'
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

    # FIXED: Using 'map' instead of 'for' preserves 'type-stability'
    map(params) do param
        @unpack xviews, pviews = param
        # FIXED: Faster than 'fill!.(pviews, xviews)'
        @inbounds @simd for i ∈ eachindex(xviews)
            fill!(pviews[i], xviews[i])
        end
    end

    return ybuf
end