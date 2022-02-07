function grad!(g, fset::FittingSet{T}, prob::NonlinearProblem, μ, n) where {T}
    @unpack cache = fset
    @unpack gbuf = cache

    # Расчет градиента по каждой группе параметров
    fill!(gbuf, zero(T))
    # FIXED: Использование 'map' вместо 'for' сохраняет 'type-stability'
    map(fset.params) do param
        grad!(cache, param, prob, μ, n)
    end
    @simd for i = 1:length(g)
        g[i] += gbuf[i]
    end

    return g
end

function grad!(cache::FittingCache, param::FittingParameter{:Tconn}, prob::NonlinearProblem, μ, n)
    @unpack C = prob
    @unpack gviews, bviews = param    
    @unpack Pcalc = @inbounds prob.pviews[n]
    @unpack cbuf, cbuf2 = cache
    V = @inbounds param.vviews[n]
    
    mul!(cbuf, C, Pcalc)
    mul!(cbuf2, C, μ)
    @inbounds @simd for i = 1:length(cbuf)
        cbuf[i] *= cbuf2[i]
    end
    @inbounds @simd for i = 1:length(bviews)
        gviews[i] = bviews[i] * V[i]
    end

    return cache
end

function grad!(cache::FittingCache, param::FittingParameter{:Tconst}, prob::NonlinearProblem, μ, n)

    @unpack gviews, bviews = param    
    @unpack Pcalc, Pi = @inbounds prob.pviews[n]
    @unpack tbuf = cache
    V = @inbounds param.vviews[n]

    @inbounds @simd for i = 1:length(tbuf)
        tbuf[i] = (Pcalc[i] - Pi[i]) * μ[i]
    end
    @inbounds @simd for i = 1:length(bviews)
        gviews[i] = bviews[i] * V[i]
    end

    return cache
end

function grad!(cache::FittingCache, param::FittingParameter{:Vpi, T}, prob::NonlinearProblem, μ, n) where {T}    
    
    @unpack tbuf, tbuf2 = cache
    @unpack "ⁿ", Pcalc, Pi, Swi, Bwi, Boi, cw, co, cf = @inbounds prob.pviews[n]
    @unpack "ⁿ⁻¹", Pcalc, Pi, Swi, Bwi, Boi, cw, co, cf, Δt = @inbounds prob.pviews[n-1]
    @unpack gviews, bviews, bviews2 = param
    Vⁿ = @inbounds param.vviews[n]
    Vⁿ⁻¹ = @inbounds param.vviews[n-1]

    @inbounds @simd for i = 1:length(tbuf)
        ΔP = Pcalcⁿ[i] - Piⁿ[i]
        tbuf[i] = Swiⁿ[i] / Bwiⁿ[i] * exp((cwⁿ[i] + cfⁿ[i]) * ΔP)
        tbuf[i] += (one(T) - Swiⁿ[i]) / Boiⁿ[i] * exp((coⁿ[i] + cfⁿ[i]) * ΔP)
        tbuf[i] *= μ[i] / Δtⁿ⁻¹[]
        ΔP = Pcalcⁿ⁻¹[i] - Piⁿ⁻¹[i]
        tbuf2[i] = Swiⁿ⁻¹[i] / Bwiⁿ⁻¹[i] * exp((cwⁿ⁻¹[i] + cfⁿ⁻¹[i]) * ΔP)
        tbuf2[i] += (one(T) - Swiⁿ⁻¹[i]) / Boiⁿ⁻¹[i] * exp((coⁿ⁻¹[i] + cfⁿ⁻¹[i]) * ΔP)
        tbuf2[i] *= μ[i] / Δtⁿ⁻¹[]          
    end
    @inbounds @simd for i = 1:length(bviews)
        gviews[i] = bviews[i] * Vⁿ[i] - bviews2[i] * Vⁿ⁻¹[i]
    end

    return cache
end

function grad!(cache::FittingCache, param::FittingParameter{:cf, T}, prob::NonlinearProblem, μ, n) where {T}    
    
    @unpack tbuf, tbuf2 = cache
    @unpack "ⁿ", Pcalc, Vpi, Pi, Swi, Bwi, Boi, cw, co, cf = @inbounds prob.pviews[n]    
    @unpack "ⁿ⁻¹", Pcalc, Vpi, Pi, Swi, Bwi, Boi, cw, co, cf, Δt = @inbounds prob.pviews[n-1]    
    @unpack gviews, bviews, bviews2 = param
    Vⁿ = @inbounds param.vviews[n]
    Vⁿ⁻¹ = @inbounds param.vviews[n-1]

    @inbounds @simd for i = 1:length(tbuf)
        ΔP = Pcalcⁿ[i] - Piⁿ[i]
        tbuf[i] = Swiⁿ[i] * Vpiⁿ[i] / Bwiⁿ[i] * exp((cwⁿ[i] + cfⁿ[i]) * ΔP) * ΔP
        tbuf[i] += (one(T) - Swiⁿ[i]) * Vpiⁿ[i] / Boiⁿ[i] * exp((coⁿ[i] + cfⁿ[i]) * ΔP) * ΔP
        tbuf[i] *= μ[i] / Δtⁿ⁻¹[]
        ΔP = Pcalcⁿ⁻¹[i] - Piⁿ⁻¹[i]
        tbuf2[i] = Swiⁿ⁻¹[i] * Vpiⁿ⁻¹[i] / Bwiⁿ⁻¹[i] * exp((cwⁿ⁻¹[i] + cfⁿ⁻¹[i]) * ΔP) * ΔP
        tbuf2[i] += (one(T) - Swiⁿ⁻¹[i]) * Vpiⁿ⁻¹[i] / Boiⁿ⁻¹[i] * exp((coⁿ⁻¹[i] + cfⁿ⁻¹[i]) * ΔP) * ΔP
        tbuf2[i] *= μ[i] / Δtⁿ⁻¹[]        
    end
    @inbounds @simd for i = 1:length(bviews)
        gviews[i] = bviews[i] * Vⁿ[i] - bviews2[i] * Vⁿ⁻¹[i]
    end

    return cache
end

function grad!(cache::FittingCache, param::FittingParameter{:λ, T}, prob::NonlinearProblem, μ, n) where {T}    
    
    @unpack gviews, bviews = param    
    @unpack Qinj = @inbounds prob.pviews[n]
    @unpack tbuf = cache
    V = @inbounds param.vviews[n]

    @inbounds @simd for i = 1:length(tbuf)
        tbuf[i] = -Qinj[i] * μ[i]
    end
    @inbounds @simd for i = 1:length(bviews)
        gviews[i] = bviews[i] * V[i]
    end

    return cache
end

function grad!(g, targ::TargetFunction)
    grad!(g, targ.finj, targ)
    grad!(g, targ.terms.L2)
end

function grad!(g, term::FracInjectionTerm, targ::TargetFunction)
    @unpack gbuf, gviews, idx = term
    @unpack J⁻¹min, J⁻¹max, ΔJ⁻¹min, ΔJ⁻¹max, α = targ.terms.Jinj
    @inbounds @simd for i = 1:length(gbuf)
        gbuf[i] = -α * (J⁻¹max[i] * ΔJ⁻¹min[i] + J⁻¹min[i] * ΔJ⁻¹max[i])
    end
    @inbounds @simd for i = 1:length(idx)
        g[idx[i]] += gviews[i]
    end
end

function grad!(g, term::L2TargetTerm{T}) where {T}
    @unpack α, x, αₓ = term
    @inbounds @simd for i = 1:length(g)
        g[i] += α * T(2) * αₓ[i] * x[i]
    end
    return g
end