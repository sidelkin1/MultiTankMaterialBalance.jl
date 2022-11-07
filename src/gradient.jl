function grad!(g, fset::FittingSet{T}, prob::NonlinearProblem, targ::TargetFunction{T}, μ, n) where {T}
    @unpack cache = fset
    @unpack gbuf = cache

    # Calculation of the gradient for each group of parameters
    fill!(gbuf, zero(T))
    # FIXED: Using 'map' instead of 'for' preserves 'type-stability'
    map(fset.params) do param
        grad!(cache, param, prob, targ, μ, n)
    end
    @turbo for i ∈ eachindex(g)
        g[i] += gbuf[i]
    end

    return g
end

function grad!(cache::FittingCache, param::FittingParameter{:Tconn}, prob::NonlinearProblem, targ::TargetFunction, μ, n)
    @unpack C = prob
    @unpack gviews, bviews = param    
    @unpack Pcalc = @inbounds prob.pviews[n]
    @unpack cbuf, cbuf2 = cache
    V = @inbounds param.vviews[n]
    
    mul!(cbuf, C, Pcalc)
    mul!(cbuf2, C, μ)
    @turbo for i ∈ eachindex(cbuf)
        cbuf[i] *= cbuf2[i]
    end
    @inbounds @simd for i ∈ eachindex(bviews)
        gviews[i] = bviews[i] * V[i]
    end

    return cache
end

function grad!(cache::FittingCache, param::FittingParameter{:Tconst}, prob::NonlinearProblem, targ::TargetFunction, μ, n)

    @unpack gviews, bviews = param    
    @unpack Pcalc, Pi = @inbounds prob.pviews[n]
    @unpack tbuf = cache
    V = @inbounds param.vviews[n]

    @turbo for i ∈ eachindex(tbuf)
        tbuf[i] = (Pcalc[i] - Pi[i]) * μ[i]
    end
    @inbounds @simd for i ∈ eachindex(bviews)
        gviews[i] = bviews[i] * V[i]
    end

    return cache
end

function grad!(cache::FittingCache{T}, param::FittingParameter{:Vpi, T}, prob::NonlinearProblem{T}, targ::TargetFunction{T}, μ, n) where {T}    
    
    @unpack tbuf, tbuf2 = cache
    @unpack "ⁿ", Pcalc, Pi, Swi, Bwi, Boi, cw, co, cf = @inbounds prob.pviews[n]
    @unpack "ⁿ⁻¹", Pcalc, Pi, Swi, Bwi, Boi, cw, co, cf, Δt = @inbounds prob.pviews[n-1]
    @unpack gviews, bviews, bviews2 = param
    Vⁿ = @inbounds param.vviews[n]
    Vⁿ⁻¹ = @inbounds param.vviews[n-1]

    @turbo for i ∈ eachindex(tbuf)
        ΔP = Pcalcⁿ[i] - Piⁿ[i]
        tbuf[i] = (
            Swiⁿ[i] / Bwiⁿ[i] * exp((cwⁿ[i] + cfⁿ[i]) * ΔP)
            + (one(T) - Swiⁿ[i]) / Boiⁿ[i] * exp((coⁿ[i] + cfⁿ[i]) * ΔP)
        ) * μ[i] / Δtⁿ⁻¹[]        
        ΔP2 = Pcalcⁿ⁻¹[i] - Piⁿ⁻¹[i]
        tbuf2[i] = (
            Swiⁿ⁻¹[i] / Bwiⁿ⁻¹[i] * exp((cwⁿ⁻¹[i] + cfⁿ⁻¹[i]) * ΔP2)
            + (one(T) - Swiⁿ⁻¹[i]) / Boiⁿ⁻¹[i] * exp((coⁿ⁻¹[i] + cfⁿ⁻¹[i]) * ΔP2)
        ) * μ[i] / Δtⁿ⁻¹[]        
    end
    @inbounds @simd for i ∈ eachindex(bviews)
        gviews[i] = bviews[i] * Vⁿ[i] - bviews2[i] * Vⁿ⁻¹[i]
    end

    return cache
end

function grad!(cache::FittingCache{T}, param::FittingParameter{:cf, T}, prob::NonlinearProblem{T}, targ::TargetFunction{T}, μ, n) where {T}    
    
    @unpack tbuf, tbuf2 = cache
    @unpack "ⁿ", Pcalc, Vpi, Pi, Swi, Bwi, Boi, cw, co, cf = @inbounds prob.pviews[n]    
    @unpack "ⁿ⁻¹", Pcalc, Vpi, Pi, Swi, Bwi, Boi, cw, co, cf, Δt = @inbounds prob.pviews[n-1]    
    @unpack gviews, bviews, bviews2 = param
    Vⁿ = @inbounds param.vviews[n]
    Vⁿ⁻¹ = @inbounds param.vviews[n-1]

    @turbo for i ∈ eachindex(tbuf)
        ΔP = Pcalcⁿ[i] - Piⁿ[i]
        tbuf[i] = (
            Swiⁿ[i] * Vpiⁿ[i] / Bwiⁿ[i] * exp((cwⁿ[i] + cfⁿ[i]) * ΔP) * ΔP
            + (one(T) - Swiⁿ[i]) * Vpiⁿ[i] / Boiⁿ[i] * exp((coⁿ[i] + cfⁿ[i]) * ΔP) * ΔP
        ) * μ[i] / Δtⁿ⁻¹[]        
        ΔP2 = Pcalcⁿ⁻¹[i] - Piⁿ⁻¹[i]
        tbuf2[i] = (
            Swiⁿ⁻¹[i] * Vpiⁿ⁻¹[i] / Bwiⁿ⁻¹[i] * exp((cwⁿ⁻¹[i] + cfⁿ⁻¹[i]) * ΔP2) * ΔP2
            + (one(T) - Swiⁿ⁻¹[i]) * Vpiⁿ⁻¹[i] / Boiⁿ⁻¹[i] * exp((coⁿ⁻¹[i] + cfⁿ⁻¹[i]) * ΔP2) * ΔP2
        ) * μ[i] / Δtⁿ⁻¹[]        
    end
    @inbounds @simd for i ∈ eachindex(bviews)
        gviews[i] = bviews[i] * Vⁿ[i] - bviews2[i] * Vⁿ⁻¹[i]
    end

    return cache
end

function grad!(cache::FittingCache, param::FittingParameter{:λ}, prob::NonlinearProblem, targ::TargetFunction, μ, n)
    
    @unpack gviews, bviews = param    
    @unpack Qinj = @inbounds prob.pviews[n]    
    @unpack tbuf = cache
    V = @inbounds param.vviews[n]
    gλ = @inbounds targ.terms.Pinj.gλviews[n]
    
    @turbo for i ∈ eachindex(tbuf)
        tbuf[i] = -Qinj[i] * μ[i] + gλ[i]
    end
    @inbounds @simd for i ∈ eachindex(bviews)
        gviews[i] = bviews[i] * V[i]
    end

    return cache
end

function grad!(g, term::L2TargetTerm)
    @turbo for i ∈ eachindex(g)
        g[i] += term.g[i]
    end
    return g
end