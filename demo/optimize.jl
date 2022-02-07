const ScaleType{T} = Union{LinearScaling{T}, SigmoidScaling{T}} where {T}

function fun(fset::FittingSet, solver::NewtonSolver, targ::TargetFunction, adjoint::AdjointSolver, ::Val{:NLopt})
    function f(x::Vector, grad::Vector, fset, solver, targ, adjoint)
        setparams!(fset, x)
        solve!(solver)
        update_targ!(targ)
        if length(grad) > 0
            solve!(adjoint)
            copyto!(grad, adjoint.g)
        end
        return getvalue(targ)
    end
    return (x, grad) -> f(x, grad, fset, solver, targ, adjoint)
end

function fun(fset::FittingSet, solver::NewtonSolver, targ::TargetFunction, adjoint::AdjointSolver, ::Val{:SciPy})
    function f(x::Vector, fset, solver, targ, adjoint)    
        setparams!(fset, x)
        solve!(solver)
        update_targ!(targ)
        solve!(adjoint)
        return getvalue(targ), adjoint.g
    end
    return x -> f(x, fset, solver, targ, adjoint)
end

function optimize(fun::Function, x0, maxiters, opts, scale::ScaleType{T}, ::Val{:NLopt}) where {T}
    minx = copy(x0)
    for (n, method) in enumerate(Iterators.cycle(opts["active"]))
        n > maxiters && break
        opt = Opt(Symbol(method), length(x0))
        method_opts = get(opts["methods"], method, nothing)

        if !isnothing(method_opts)
            for (key, val) ∈ method_opts
                setproperty!(opt, Symbol(key), val)
            end
        end

        if isa(scale, LinearScaling)
            opt.lower_bounds = zero(T)
            opt.upper_bounds = one(T)
        else isa(scale, SigmoidScaling)
            opt.lower_bounds = T(-Inf)
            opt.upper_bounds = T(Inf)
        end
        
        opt.min_objective = fun
        minf, minx, ret = optimize!(opt, minx)
        numevals = opt.numevals
        println("n: $n, method: $method, minf: $minf, numevals: $numevals, ret: $ret")
    end
    return minx
end

function optimize(fun::Function, x0, maxiters, opts, scale::ScaleType{T}, ::Val{:SciPy}) where {T}
    res = SciPy.optimize.OptimizeResult(x=copy(x0))
    bounds = fill([zero(T), one(T)], length(x0))
    for (n, method) in enumerate(Iterators.cycle(opts["active"]))
        n > maxiters && break
        method_opts = get(opts["methods"], method, nothing)

        if isa(scale, LinearScaling)
            res = SciPy.optimize.minimize(fun, res["x"]; method, jac=true, bounds, options=method_opts)
        else isa(scale, SigmoidScaling)
            res = SciPy.optimize.minimize(fun, res["x"]; method, jac=true, options=method_opts)
        end

        println("n: $n, method: $method, minf: $(res["fun"]), numevals: $(res["nfev"]), ret: $(res["message"])")
    end
    return res
end

function print_result(minx, initial_x, fun::Function, targ::TargetFunction, ::Val{:NLopt})
    println("Before optimization:")
    fun(initial_x, [])
    println(getvalues(targ))
    calc_well_index!(targ)    
    println("After optimization:")
    fun(minx, [])
    println(getvalues(targ))
    calc_well_index!(targ)    
end

function print_result(res, initial_x, fun::Function, targ::TargetFunction, ::Val{:SciPy})
    println("Before optimization:")
    fun(initial_x)
    println(getvalues(targ))
    calc_well_index!(targ)    
    println("After optimization:")
    fun(res["x"])
    println(getvalues(targ))
    calc_well_index!(targ)    
end