Base.@kwdef struct ModelParameters{A<:AbstractArray{<:AbstractFloat}, B<:AbstractArray{<:Bool}}
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
    Tupd::B
    Vupd::B
    cupd::B
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
    Cbuf::SparseMatrixCSC{T, Int}
end

Base.@kwdef struct NonlinearProblem{T<:AbstractFloat}
    params::ModelParameters{Matrix{T}, BitMatrix}    
    pviews::Vector{ModelParameters{ColumnSlice{T}, ColumnSliceBool}}
    C::SparseMatrixCSC{T, Int}
    cache::ProblemCache{T}
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

function build_update_matrix(::Type{T}, df_params, names, M) where {T}
    A = falses(1, M)
    for name ∈ names
        df = @view df_params[df_params.Parameter .=== name, :]
        A[df.Jstart] .= true
    end
    return A
end

function NonlinearProblem{T}(df_rates::AbstractDataFrame, df_params::AbstractDataFrame) where {T}
    
    # Формируем матрицу смежности
    C = build_incidence_matrix(T, df_params)    
    # Число соединений и блоков
    Nc, Nt = size(C)
    # Число временных шагов
    Nd = (length∘unique)(df_rates.Date::Vector{Date})
    
    # Формируем матрицы параметров и отборов
    kwargs = (
        # Исходные параметры модели
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

        # Флаги обновления буферов
        Tupd = build_update_matrix(T, df_params, (:Tconn,), Nd),
        Vupd = build_update_matrix(T, df_params, (:Vpi, :Bwi, :Boi, :Swi), Nd),
        cupd = build_update_matrix(T, df_params, (:cw, :co, :cf), Nd),
        
        # Вычисляемые параметры модели
        Pcalc = Array{T}(undef, Nt, Nd),
        Qliq = Array{T}(undef, Nt, Nd),
        Qinj = Array{T}(undef, Nt, Nd),
        jac_next = Array{T}(undef, Nt, Nd),
        Pbhp = Array{T}(undef, Nt, Nd),
        Pinj = Array{T}(undef, Nt, Nd),
    )

    params, pviews = params_and_views(ModelParameters, kwargs)    
    cache = ProblemCache{T}(Nt, Nc)
    NonlinearProblem{T}(; params, C, pviews, cache)
end

function ProblemCache{T}(Nt, Nc) where {T}
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
        Cbuf = spzeros(T, Nc, Nt),        
    )
    ProblemCache{T}(; kwargs...)
end

function val_and_jac!(r, J, P, prob::NonlinearProblem, n)

    params = @inbounds prob.pviews[n]
    @unpack Vwprev, Voprev, Vw, Vo, CTC = prob.cache    
    @unpack Vwi, Voi, cwf, cof, Qsum = prob.cache
    @unpack Pi, Tconst = params

    # Инициализация невязки и якобиана
    mul!(r, CTC, P)
    copyto!(J, CTC)

    @inbounds @simd for i = 1:length(P)
        # Поровые объемы воды и нефти в пов. усл.
        Vw[i] = Vwi[i] * exp(cwf[i] * (P[i] - Pi[i]))
        Vo[i] = Voi[i] * exp(cof[i] * (P[i] - Pi[i]))

        # Обновляем вектор невязки до выполнения условия ∥r∥ ≈ 0
        r[i] += Vw[i] - Vwprev[i] 
        r[i] += Vo[i] - Voprev[i] 
        r[i] += Tconst[i] * (P[i] - Pi[i]) + Qsum[i]

        # В целом, якобиан статичен за исключением диагональных элементов
        J[i, i] += Tconst[i] + Vw[i] * cwf[i] + Vo[i] * cof[i]
    end    

    return r, J
end