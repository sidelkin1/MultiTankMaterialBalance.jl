Base.@kwdef struct LinearScaling{T} <: AbstractParametersScaling{T}
    xrange::Vector{T}
    xmin::Vector{T}
end

Base.@kwdef struct SigmoidScaling{T} <: AbstractParametersScaling{T}
    xrange::Vector{T}
    xmin::Vector{T}
    xsigm::Vector{T}
end

function LinearScaling{T}(df_params::AbstractDataFrame) where {T}
    df = @view df_params[df_params.Skip .=== false, :]
    xrange = @with(df, :Max_value .- :Min_value)
    xmin = @with(df, @. -:Min_value / xrange)
    LinearScaling{T}(; xrange, xmin)
end

function SigmoidScaling{T}(df_params::AbstractDataFrame) where {T}
    df = @view df_params[df_params.Skip .=== false, :]
    xrange = @with(df, :Max_value .- :Min_value)
    xmin = @with(df, @. -:Min_value / xrange)
    xsigm = Array{T}(undef, length(xrange))
    SigmoidScaling{T}(; xrange, xmin, xsigm)
end

function scalex!(x, y, scale::LinearScaling{T}) where {T}
    @unpack xrange, xmin = scale
    @inbounds @simd for i = 1:length(x)
        x[i] = clamp(xmin[i] + y[i] / xrange[i], zero(T), one(T))
    end
end

function scalex!(x, y, scale::SigmoidScaling{T}) where {T}
    @unpack xrange, xmin = scale    
    @inbounds @simd for i = 1:length(x)    		
        zmin = T(0.0001) / xrange[i]
        z = clamp(xmin[i] + y[i] / xrange[i], zmin, one(T) - zmin)
        x[i] = log(z / (one(T) - z))
    end
end

function unscalex!(y, x, scale::LinearScaling)
    @unpack xrange, xmin = scale
    @inbounds @simd for i = 1:length(y)
        y[i] = (x[i] - xmin[i]) * xrange[i]
    end
end

function unscalex!(y, x, scale::SigmoidScaling{T}) where {T}
    @unpack xrange, xmin, xsigm = scale
    @inbounds @simd for i = 1:length(y)
        xsigm[i] = one(T) / (one(T) + exp(-x[i]))
        y[i] = xrange[i] * (xsigm[i] - xmin[i])
    end
end

function unscaleg!(g, scale::LinearScaling)
    @unpack xrange = scale
    @inbounds @simd for i = 1:length(g)
        g[i] *= xrange[i]
    end
end

function unscaleg!(g, scale::SigmoidScaling{T}) where {T}
    @unpack xrange, xsigm = scale
    @inbounds @simd for i = 1:length(g)
        g[i] *= xrange[i] * xsigm[i] * (one(T) - xsigm[i])
    end
end