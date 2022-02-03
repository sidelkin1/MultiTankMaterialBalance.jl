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