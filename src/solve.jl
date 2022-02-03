function solve!(prob::NonlinearProblem, alg::AbstractSolveAlgorithm, args...; kwargs...)
    solver = init(prob, alg, args...; kwargs...)
    solve!(solver)
end

function solve!(solver::AbstractNonlinearSolver)

    @unpack P = solver
    params = solver.prob.pviews[1]
    copyto!(P, params.Pi)
    init_cache!(solver.cache, params)

    for (n, params) ∈ enumerate(solver.itr)

        update_cache!(solver.cache, params, solver.prob.C, P) 
        perform_step!(solver, solver.alg, params)
       
        @printf "n: %d, " n
        @printf "iter: %d, " solver.iter
        @printf "norm_r: %.5e, " solver.norm_r
        @printf "dP_rel: %.5e\n" solver.dP_rel 
        
        update_params!(params, solver.cache, P)
        @inbounds solver.prob.jacs[n] = copy(solver.F)

        solver.success || break      
    end
    solver.success || error("Newton solver did not converge")

    return solver
end

function init_cache!(cache::ProblemCache, params::ModelParameters)

    @unpack Vpi, Swi, Bwi, Boi = params
    @unpack Vwprev, Voprev = cache

    # Поровые объемы на нулевом временном шаге
    @fastmath @. Vwprev = Swi * Vpi / Bwi
    @fastmath @. Voprev = (1 - Swi) * Vpi / Boi

    return cache
end

function update_cache!(cache::ProblemCache, params::ModelParameters, C, P)

    @unpack cw, co, cf, Vpi, Swi, Bwi, Boi, Pmin, Pmax, Tconst = params
    @unpack Tconn, Qliq, Qinj, Qwat_h, Qoil_h, Qinj_h, λ = params
    @unpack Vwi, Voi, diagCTC, CTC, idx, cwf, cof, Qsum = cache

    # Нач. поровые объемов флюидов в пов. усл.
    @fastmath @. Vwi = Swi * Vpi / Bwi
    @fastmath @. Voi = (1 - Swi) * Vpi / Boi
   
    # Коррекция отборов с учетом верх. и ниж. границ Рпл
    @fastmath @. Qliq = (P ≥ Pmin) * (Qwat_h + Qoil_h)
    @fastmath @. Qinj = (P ≤ Pmax) * Qinj_h
    @fastmath @. Qsum = Qliq - λ * Qinj     

    # Суммарные сжимаемости системы пласт-флюид
    @fastmath @. cwf = cw + cf
    @fastmath @. cof = co + cf
    
    # Элементы якобиана на тек. временном шаге, не зависящие от Рпл 
    CTC .= C' * Diagonal(Tconn) * C
    copyto!(diagCTC, view(CTC, idx))
    @fastmath @. diagCTC += Tconst

    return cache, params
end

function update_params!(params::ModelParameters, cache::ProblemCache, P)

    @unpack jac_next, Pcalc, Qliq, Jp, Pbhp, λ, Qinj, Jinj, Pinj = params
    @unpack Vwprev, Voprev, Vw, Vo, cwf, cof = cache

    # Сохраняем данные на текущем временном шаге
    copyto!(Pcalc, P)
    copyto!(Vwprev, Vw)
    copyto!(Voprev, Vo)

    # Якобиан относительно Рпл на пред. временном шаге
    @fastmath @. jac_next = -(Vw * cwf + Vo * cof)

    # Забойные давления
    @fastmath @. Pbhp = P - Qliq / Jp
    @fastmath @. Pinj = P + λ * Qinj / Jinj

    return params, cache
end

function Base.getproperty(obj::AbstractNonlinearSolver, sym::Symbol)
    if sym === :itr
        return obj.prob.pviews
    else # fallback to getfield
        return getfield(obj, sym)
    end
end