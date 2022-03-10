function solve!(solver::AbstractNonlinearSolver; verbose=false)

    @unpack P, prob, linalg = solver    
    init_values!(P, prob)

    for n = 2:length(prob.pviews)
        @unpack Δt = @inbounds prob.pviews[n-1]
        prepare_step!(prob, n, P) 
        perform_step!(solver, Δt[], n)
       
        if verbose
            @printf "n: %d, " n
            @printf "iter: %d, " solver.iter
            @printf "norm_r: %.5e, " solver.norm_r
            @printf "dP_rel: %.5e\n" solver.dP_rel
        end
        
        accept_step!(prob, n, P)
        solver.success || break      
    end
    solver.success || error("Newton solver did not converge")

    return solver
end

function init_values!(P, prob::NonlinearProblem)

    @unpack Pi = @inbounds prob.pviews[1]
    @unpack Vwprev, Voprev, Vwi, Voi = prob.cache    
    
    # Initialization at zero time step
    update_cache!(prob, 1)
    copyto!(P, Pi)
    copyto!(Vwprev, Vwi)
    copyto!(Voprev, Voi)

    return P, prob
end

function prepare_step!(prob::NonlinearProblem, n, P)

    params = @inbounds prob.pviews[n]
    @unpack Pmin, Pmax, Qliq, Qinj, Qliq_h, Qinj_h, λ = params    
    @unpack Qsum = prob.cache

    update_cache!(prob, n)
    @turbo for i = 1:length(P)   
        # Correction of rates taking into account the upper and lower bounds of pressure
        Qliq[i] = (P[i] ≥ Pmin[i]) * Qliq_h[i]
        Qinj[i] = (P[i] ≤ Pmax[i]) * Qinj_h[i]
        Qsum[i] = Qliq[i] - λ[i] * Qinj[i]    
    end
    
    return prob
end

function accept_step!(prob::NonlinearProblem, n, P)

    params = @inbounds prob.pviews[n]
    @unpack jac_next, Pcalc, Δt = params
    @unpack Vwprev, Voprev, Vw, Vo, cwf, cof = prob.cache

    # Saving data at the current time step
    copyto!(Pcalc, P)
    copyto!(Vwprev, Vw)
    copyto!(Voprev, Vo)

    # The diagonal of the jacobian with respect to reservoir pressure on the previous time step
    @turbo for i = 1:length(P)        
        jac_next[i] = -(Vw[i] * cwf[i] + Vo[i] * cof[i]) / Δt[]
    end

    return prob
end

function update_cache!(prob::NonlinearProblem{T}, n) where {T}

    params = @inbounds prob.pviews[n]

    # Initial porous volumes of fluids
    if params.Vupd[]
        @unpack Vpi, Swi, Bwi, Boi = params
        @unpack Vwi, Voi = prob.cache
        @turbo for i = 1:length(Vpi)
            Vwi[i] = Swi[i] * Vpi[i] / Bwi[i]
            Voi[i] = (one(T) - Swi[i]) * Vpi[i] / Boi[i]
        end
    end

    # Elements of the jacobian that do not depend on reservoir pressure
    if params.Tupd[]
        @unpack Tconn = params
        @unpack CTC, Cbuf = prob.cache
        # copyto!(Cbuf, prob.C)
        # lmul!(Diagonal(Tconn), Cbuf)
        # mul!(CTC, prob.C', Cbuf)
        # FIXED: For a sparse matrix 'C', the construction below is faster
        CTC .= prob.C' * Diagonal(Tconn) * prob.C
    end

    # Total compressibility of the reservoir-fluid system
    if params.cupd[]
        @unpack cw, co, cf = params
        @unpack cwf, cof = prob.cache        
        @turbo for i = 1:length(cf)
            cwf[i] = cw[i] + cf[i]
            cof[i] = co[i] + cf[i]
        end
    end

    return prob
end