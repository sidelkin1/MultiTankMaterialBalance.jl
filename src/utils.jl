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

function build_rate_matrix(::Type{T}, data, N, M) where {T}
    @_ data |> 
        replace(__, missing =>  NaN) |>
        convert(Vector{T}, __)::Vector{T} |> 
        reshape(__, M, N) |> 
        permutedims(__)
end

function params_and_views(::Type{T}, data) where {T}
    params = T(; data...)
    pviews = map(zip(eachcol.(values(data))...)) do x
        T(; (keys(data) .=> x)...)
    end
    return params, pviews
end