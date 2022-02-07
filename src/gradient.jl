function grad!(g, fset::FittingSet{T}, prob::NonlinearProblem, targ::TargetFunction, μ, n) where {T}
    @unpack cache = fset
    @unpack gbuf = cache

    # Расчет градиента по каждой группе параметров
    fill!(gbuf, zero(T))
    # FIXED: Использование 'map' вместо 'for' сохраняет 'type-stability'
    map(fset.params) do param
        grad!(cache, param, prob, targ, μ, n)
    end
    @simd for i = 1:length(g)
        g[i] += gbuf[i]
    end

    return g
end

grad!(param::FittingParameter{S}, prob::NonlinearProblem, targ::TargetFunction, cache::FittingCache, μ, n) where {S} = nothing

function grad!(cache::FittingCache, param::FittingParameter{:Tconn}, prob::NonlinearProblem, targ::TargetFunction, μ, n)
    @unpack C = prob
    @unpack gviews, bviews = param    
    @unpack V = @inbounds param.vviews[n]
    @unpack Pcalc = @inbounds prob.pviews[n]
    @unpack cbuf, cbuf2 = cache
    
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

function grad!(cache::FittingCache, param::FittingParameter{:Tconst}, prob::NonlinearProblem, targ::TargetFunction, μ, n)

    @unpack gviews, bviews = param
    @unpack V = @inbounds param.vviews[n]
    @unpack Pcalc, Pi = @inbounds prob.pviews[n]
    @unpack tbuf = cache

    @inbounds @simd for i = 1:length(tbuf)
        tbuf[i] = (Pcalc[i] - Pi[i]) * μ[i]
    end
    @inbounds @simd for i = 1:length(bviews)
        gviews[i] = bviews[i] * V[i]
    end

    return cache
end

function grad!(cache::FittingCache, param::FittingParameter{:Vpi, T}, prob::NonlinearProblem, targ::TargetFunction, μ, n) where {T}    
    
    @unpack tbuf, tbuf2 = cache
    @unpack "ⁿ", Pcalc, Pi, Swi, Bwi, Boi, cw, co, cf = @inbounds prob.pviews[n]
    @unpack "ⁿ", V = @inbounds param.vviews[n]
    @unpack "ⁿ⁻¹", Pcalc, Pi, Swi, Bwi, Boi, cw, co, cf, Δt = @inbounds prob.pviews[n-1]
    @unpack "ⁿ⁻¹", V = @inbounds param.vviews[n-1]
    @unpack gviews, bviews, bviews2 = param

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

function grad!(cache::FittingCache, param::FittingParameter{:cf, T}, prob::NonlinearProblem, targ::TargetFunction, μ, n) where {T}    
    
    @unpack tbuf, tbuf2 = cache
    @unpack "ⁿ", Pcalc, Vpi, Pi, Swi, Bwi, Boi, cw, co, cf = @inbounds prob.pviews[n]
    @unpack "ⁿ", V = @inbounds param.vviews[n]
    @unpack "ⁿ⁻¹", Pcalc, Vpi, Pi, Swi, Bwi, Boi, cw, co, cf, Δt = @inbounds prob.pviews[n-1]
    @unpack "ⁿ⁻¹", V = @inbounds param.vviews[n-1]
    @unpack gviews, bviews, bviews2 = param

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

function grad!(cache::FittingCache, param::FittingParameter{:λ, T}, prob::NonlinearProblem, targ::TargetFunction, μ, n) where {T}    
    
    @unpack gviews, bviews = param
    @unpack V = @inbounds param.vviews[n]
    @unpack Qinj, Pinj, Jinj = @inbounds prob.pviews[n]
    @unpack Wnan, Wobs, Pobs = @inbounds targ.terms.Pinj.pviews[n]
    @unpack α = targ.terms.Pinj
    @unpack tbuf = cache

    @inbounds @simd for i = 1:length(tbuf)
        tbuf[i] = -Qinj[i] * μ[i] + Wnan[i] * (T(2) * α * Qinj[i] * Wobs[i] * (Pinj[i] - Pobs[i])) / Jinj[i]
    end
    @inbounds @simd for i = 1:length(bviews)
        gviews[i] = bviews[i] * V[i]
    end

    return cache
end

function grad!(cache::FittingCache, param::FittingParameter{:Jp, T}, prob::NonlinearProblem, targ::TargetFunction, μ, n) where {T}    
    
    @unpack gviews, bviews = param    
    @unpack V = @inbounds param.vviews[n]
    @unpack Qliq, Pbhp, Jp = @inbounds prob.pviews[n]
    @unpack Wnan, Wobs, Pobs = @inbounds targ.terms.Pbhp.pviews[n]
    @unpack α = targ.terms.Pbhp
    @unpack tbuf = cache

    @inbounds @simd for i = 1:length(tbuf)
        tbuf[i] = Wnan[i] * (T(2) * α * Qliq[i] * Wobs[i] * (Pbhp[i] - Pobs[i])) / Jp[i]^2
    end
    @inbounds @simd for i = 1:length(bviews)
        gviews[i] = bviews[i] * V[i]
    end

    return cache
end

function grad!(cache::FittingCache, param::FittingParameter{:Jinj, T}, prob::NonlinearProblem, targ::TargetFunction, μ, n) where {T}    
    
    @unpack gviews, bviews = param    
    @unpack V = @inbounds param.vviews[n]
    @unpack Qinj, Pinj, Jinj, λ = @inbounds prob.pviews[n]
    @unpack Wnan, Wobs, Pobs = @inbounds targ.terms.Pinj.pviews[n]
    @unpack α = targ.terms.Pinj
    @unpack tbuf = cache

    @inbounds @simd for i = 1:length(tbuf)
        tbuf[i] = Wnan[i] * (-T(2) * α * λ[i] * Qinj[i] * Wobs[i] * (Pinj[i] - Pobs[i])) / Jinj[i]^2
    end
    @inbounds @simd for i = 1:length(bviews)
        gviews[i] = bviews[i] * V[i]
    end

    return cache
end

function grad!(g, term::L2TargetTerm{T}) where {T}    
    @unpack α, x, αₓ = term
    for i = 1:length(g)
        g[i] += α * T(2) * αₓ[i] * x[i]
    end
    return g
end