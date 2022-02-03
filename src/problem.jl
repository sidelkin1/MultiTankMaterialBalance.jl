Base.@kwdef struct ModelParameters{A<:AbstractArray{<:AbstractFloat}}
    Tconn::A
    Pi::A
    Bwi::A
    Boi::A
    cw::A
    co::A
    cf::A
    Swi::A
    Vpi::A    
    Tconst::A
    Jp::A
    Jinj::A
    λ::A
    Pmin::A
    Pmax::A
    Qoil_h::A
    Qwat_h::A
    Qinj_h::A
    Pcalc::A
    Qliq::A
    Qinj::A
    jac_next::A
    Pbhp::A
    Pinj::A
end

Base.@kwdef struct NonlinearProblem{T<:AbstractFloat}    
    params::ModelParameters{Matrix{T}}
    C::SparseMatrixCSC{T, Int}
    pviews::Vector{ModelParameters{ColumnSlice{T}}}
    jacs::Vector{SuiteSparse.CHOLMOD.Factor}
end

function build_incidence_matrix(::Type{T}, df_params) where {T}
    
    # Нумерация блоков
    dict = @with df_params begin 
        @_ :Tank::Vector{String} |> 
            unique(__) |> 
            Dict(__ .=> 1:length(__))
    end
    
    # Номера соседних блоков
    df = @view df_params[df_params.Parameter .=== :Tconn, :]
    N = @with df begin
        @_ [:Tank :Neighb]::Matrix{String} |> 
            unique(__, dims=1) |> 
            getindex.(Ref(dict), __)
    end

    # Матрица смежности блоков
    C = spzeros(T, size(N, 1), length(dict))
    C[CartesianIndex.(axes(C, 1), N)] .= [-1 1]

    return C
end

function build_parameter_matrix(::Type{T}, df_params, name, N, M) where {T}
    A = Array{T}(undef, N, M)
    df = @view df_params[df_params.Parameter .=== name, :]
    @with df @byrow begin
        cols = UnitRange(:Jstart::Int, :Jstop::Int)
        A[:Istart::Int, cols] .= convert(T, :Init_value)::T
    end
    return A
end

function NonlinearProblem{T}(df_rates::AbstractDataFrame, df_params::AbstractDataFrame) where {T}
    
    # Формируем матрицу смежности
    C = build_incidence_matrix(T, df_params)    
    # Число соединений и блоков
    (Nc, Nt) = size(C)
    # Число временных шагов
    Nd = (length∘unique)(df_rates.Date::Vector{Date})
    
    # Формируем матрицы параметров и отборов
    kwargs = (
        # Фиксированные параметры модели
        Tconn = build_parameter_matrix(T, df_params, :Tconn, Nc, Nd),
        Pi = build_parameter_matrix(T, df_params, :Pi, Nt, Nd),
        Bwi = build_parameter_matrix(T, df_params, :Bwi, Nt, Nd),
        Boi = build_parameter_matrix(T, df_params, :Boi, Nt, Nd),
        cw = build_parameter_matrix(T, df_params, :cw, Nt, Nd),
        co = build_parameter_matrix(T, df_params, :co, Nt, Nd),
        cf = build_parameter_matrix(T, df_params, :cf, Nt, Nd),
        Swi = build_parameter_matrix(T, df_params, :Swi, Nt, Nd),
        Vpi = build_parameter_matrix(T, df_params, :Vpi, Nt, Nd),
        Tconst = build_parameter_matrix(T, df_params, :Tconst, Nt, Nd),
        Jp = build_parameter_matrix(T, df_params, :Jp, Nt, Nd),
        Jinj = build_parameter_matrix(T, df_params, :Jinj, Nt, Nd),
        λ = build_parameter_matrix(T, df_params, :λ, Nt, Nd),
        Pmin = build_parameter_matrix(T, df_params, :Pmin, Nt, Nd),
        Pmax = build_parameter_matrix(T, df_params, :Pmax, Nt, Nd),
        Qoil_h = build_rate_matrix(T, df_rates.Qoil, Nt, Nd),
        Qwat_h = build_rate_matrix(T, df_rates.Qwat, Nt, Nd),
        Qinj_h = build_rate_matrix(T, df_rates.Qinj, Nt, Nd),
        # Вычисляемые параметры модели
        Pcalc = Array{T}(undef, Nt, Nd),
        Qliq = Array{T}(undef, Nt, Nd),
        Qinj = Array{T}(undef, Nt, Nd),
        jac_next = Array{T}(undef, Nt, Nd),
        Pbhp = Array{T}(undef, Nt, Nd),
        Pinj = Array{T}(undef, Nt, Nd),
    )

    params, pviews = params_and_views(ModelParameters, kwargs)
    jacs = map(_ -> cholesky(spdiagm(ones(T, Nt))), 1:Nd)
    NonlinearProblem{T}(; params, C, pviews, jacs)
end

Base.@kwdef struct ProblemCache{T<:AbstractFloat}
    Vw::Vector{T}
    Vo::Vector{T}
    Vwprev::Vector{T}
    Voprev::Vector{T}
    Vwi::Vector{T}
    Voi::Vector{T}
    cwf::Vector{T}
    cof::Vector{T}
    Qsum::Vector{T}
    CTC::SparseMatrixCSC{T, Int}
    diagCTC::Vector{T}
    idx::Vector{Int}
end

function ProblemCache{T}(prob::NonlinearProblem{T}) where {T}
    Nt = size(prob.params.Pcalc, 1)
    kwargs = (
        Vw = Array{T}(undef, Nt),
        Vo = Array{T}(undef, Nt),
        Vwprev = Array{T}(undef, Nt),
        Voprev = Array{T}(undef, Nt),
        Vwi = Array{T}(undef, Nt),
        Voi = Array{T}(undef, Nt),
        cwf = Array{T}(undef, Nt),
        cof = Array{T}(undef, Nt),
        Qsum = Array{T}(undef, Nt),
        CTC = spzeros(T, Nt, Nt),
        diagCTC = Array{T}(undef, Nt),    
    )
    idx = diagind(kwargs.CTC)
    ProblemCache{T}(; kwargs..., idx)
end

function val_and_jac!(r, J, P, cache::ProblemCache, params::ModelParameters)

    @unpack Vwprev, Voprev, Vw, Vo = cache
    @unpack CTC, diagCTC, idx = cache
    @unpack Vwi, Voi, cwf, cof, Qsum = cache
    @unpack Pi, Tconst = params

    # Поровые объемы воды и нефти в пов. усл.
    @fastmath @. Vw = Vwi * exp(cwf * (P - Pi))
    @fastmath @. Vo = Voi * exp(cof * (P - Pi))
    
    # Обновляем вектор невязки до выполнения условия ∥r∥ ≈ 0
    mul!(r, CTC, P)
    @fastmath @. r += Vw - Vwprev + Vo - Voprev + Tconst * (P - Pi) + Qsum
    
    # В целом, якобиан статичен за исключение диагональных элементов
    J .= CTC
    @fastmath @inbounds @. J[idx] = diagCTC + Vw * cwf + Vo * cof

    return r, J
end