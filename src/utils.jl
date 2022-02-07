import UnPack

opencsv(path) = open(read, path, CSV_ENC)

# FIXED: Переделанная версия '@unpack' с возможностью добавления суффикса к названию переменных
macro unpack(args)
    args.head != :(=) && error("Expression needs to be of form `a, b = c`")
    items, suitecase = args.args    
    items = isa(items, Symbol) ? [items] : items.args
    suffix, items... = isa(first(items), AbstractString) ? items : ("", items...)
    suitecase_instance = gensym()
    kd = map(items) do key
        var = Symbol(string(key) * suffix)
        :( $var = $UnPack.unpack($suitecase_instance, Val{$(Expr(:quote, key))}()) )
    end
    kdblock = Expr(:block, kd...)
    expr = quote
        local $suitecase_instance = $suitecase # handles if suitecase is not a variable but an expression
        $kdblock
        $suitecase_instance # return RHS of `=` as standard in Julia
    end
    esc(expr)
end

x_abschange(x, x_previous) = maxdiff(x, x_previous)
maxdiff(x, y) = mapreduce((a, b) -> abs(a - b), max, x, y)

function assess_convergence(x, x_prev, f_x, x_tol, f_tol)
    x_converged, f_converged = false, false
    if x_abschange(x, x_prev) ≤ x_tol * maximum(abs, x)
        x_converged = true
    end
    if maximum(abs, f_x) ≤ f_tol
        f_converged = true
    end
    return x_converged, f_converged
end

getparams(df::AbstractDataFrame, ::Val{:tanks}) = @view df[df.Parameter .∉ Ref((:Gw, :Jinj, :Jp)), :]
getparams(df::AbstractDataFrame, ::Val{:wells}) = @view df[df.Parameter .∈ Ref((:Gw, :Jinj, :Jp)), :]
getparams(df::AbstractDataFrame, ::Val{:tanks}, ::Val{:var}) = @view df[(df.Parameter .∉ Ref((:Gw, :Jinj, :Jp))) .& .!df.Const, :]
getparams(df::AbstractDataFrame, ::Val{:tanks}, ::Val{:const}) = @view df[(df.Parameter .∉ Ref((:Gw, :Jinj, :Jp))) .& df.Const, :]
getparams(df::AbstractDataFrame, ::Val{:wells}, ::Val{:var}) = @view df[(df.Parameter .∈ Ref((:Gw, :Jinj, :Jp))) .& .!df.Ignore, :]
getparams(df::AbstractDataFrame, ::Val{:wells}, ::Val{:const}) = @view df[(df.Parameter .∈ Ref((:Gw, :Jinj, :Jp))) .& df.Ignore, :]
getparams(df::AbstractDataFrame, ::Val{S}) where {S} = @view df[df.Parameter .=== S, :]
getparams(df::AbstractDataFrame, ::Val{S}, ::Val{:var}) where {S} = @view df[(df.Parameter .=== S) .& .!df.Const, :]
getparams(df::AbstractDataFrame, ::Val{S}, ::Val{:const}) where {S} = @view df[(df.Parameter .=== S) .& df.Const, :]
getparams(df::AbstractDataFrame, ::Val{:Gw}, ::Val{:var}) = @view df[(df.Parameter .=== :Gw) .& .!df.Ignore, :]
getparams(df::AbstractDataFrame, ::Val{:Jinj}, ::Val{:var}) = @view df[(df.Parameter .=== :Jinj) .& .!df.Ignore, :]
getparams(df::AbstractDataFrame, ::Val{:Jp}, ::Val{:var}) = @view df[(df.Parameter .=== :Jp) .& .!df.Ignore, :]
getparams(df::AbstractDataFrame, ::Val{:Gw}, ::Val{:const}) = @view df[(df.Parameter .=== :Gw) .& df.Ignore, :]
getparams(df::AbstractDataFrame, ::Val{:Jinj}, ::Val{:const}) = @view df[(df.Parameter .=== :Jinj) .& df.Ignore, :]
getparams(df::AbstractDataFrame, ::Val{:Jp}, ::Val{:const}) = @view df[(df.Parameter .=== :Jp) .& df.Ignore, :]