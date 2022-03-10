Base.@kwdef struct ModelParameters{A<:AbstractArray{<:AbstractFloat}, B<:AbstractArray{<:Bool}}
    Δt::A
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
    Gw::A
    Jp::A
    M::A
    Jinj::A
    λ::A
    Pmin::A
    Pmax::A
    Qliq_h::A
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
    diagJ::Vector{T}
end

Base.@kwdef struct NonlinearProblem{T<:AbstractFloat}
    params::ModelParameters{Matrix{T}, BitMatrix}    
    pviews::Vector{ModelParameters{ColumnSlice{T}, ColumnSliceBool}}
    C::SparseMatrixCSC{T, Int}
    cache::ProblemCache{T}
end

function build_incidence_matrix(::Type{T}, df) where {T}
    
    # Tanks numbering
    dict = @with df begin 
        @_ :Tank::Vector{String} |> 
            unique(__) |> 
            Dict(__ .=> 1:length(__))
    end
    
    # Numbers of neighbouring tanks
    df_view = getparams(df, Val(:Tconn))
    N = @with df_view begin
        @_ [:Tank :Neighb]::Matrix{String} |> 
            unique(__, dims=1) |> 
            getindex.(Ref(dict), __)
    end

    # Tank adjacency matrix
    C = spzeros(T, size(N, 1), length(dict))
    C[CartesianIndex.(axes(C, 1), N)] .= [-1 1]

    return C
end

function build_parameter_matrix(::Type{T}, df, name::Symbol, N, M) where {T}
    A = Array{T}(undef, N, M)
    df_view = getparams(df, Val(name))
    @with df_view @byrow begin
        cols = UnitRange(:Jstart::Int, :Jstop::Int)
        A[:Istart::Int, cols] .= convert(T, :Init_value)::T
    end
    return A
end

function build_update_matrix(::Type{T}, df, names::NTuple{N, Symbol}, M) where {T, N}
    A = falses(1, M)
    for name ∈ names
        df_view = getparams(df, Val(name))
        A[df_view.Jstart] .= true
    end
    return A
end

function build_rate_matrix(::Type{T}, data, N, M) where {T}
    @_ data |> 
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

function NonlinearProblem{T}(df_rates::AbstractDataFrame, df_params::AbstractDataFrame) where {T}
    
    # Adjacency matrix building
    C = build_incidence_matrix(T, df_params)    
    # Number of connections and tanks
    Nc, Nt = size(C)
    # Number of time steps
    dates = unique(df_rates.Date::Vector{Date})
    Nd = length(dates)
    
    # Form matrices of parameters and history of production/injection
    kwargs = (
        # Input variables of model
        Δt = build_rate_matrix(T, daysinmonth.(dates), 1, Nd),    
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
        λ = build_parameter_matrix(T, df_params, :λ, Nt, Nd),
        Pmin = build_parameter_matrix(T, df_params, :Pmin, Nt, Nd),
        Pmax = build_parameter_matrix(T, df_params, :Pmax, Nt, Nd), 
        Qliq_h = build_rate_matrix(T, df_rates.Qliq, Nt, Nd),
        Qinj_h = build_rate_matrix(T, df_rates.Qinj, Nt, Nd),
        M = build_rate_matrix(T, df_rates.Total_mobility, Nt, Nd),

        # Auxiliary buffer update flags
        Tupd = build_update_matrix(T, df_params, (:Tconn,), Nd),
        Vupd = build_update_matrix(T, df_params, (:Vpi, :Bwi, :Boi, :Swi,), Nd),
        cupd = build_update_matrix(T, df_params, (:cw, :co, :cf,), Nd),
        
        # Output variables of model
        Pcalc = build_parameter_matrix(T, df_params, :Pi, Nt, Nd),
        Qliq = build_rate_matrix(T, df_rates.Qliq, Nt, Nd),
        Qinj = build_rate_matrix(T, df_rates.Qinj, Nt, Nd),
        Gw = build_parameter_matrix(T, df_params, :Gw, Nt, Nd),
        Jp = build_parameter_matrix(T, df_params, :Jp, Nt, Nd),
        Jinj = build_parameter_matrix(T, df_params, :Jinj, Nt, Nd),
        Pbhp = build_parameter_matrix(T, df_params, :Pi, Nt, Nd),
        Pinj = build_parameter_matrix(T, df_params, :Pi, Nt, Nd),
        jac_next = Array{T}(undef, Nt, Nd),
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
        diagJ = Array{T}(undef, Nt),
    )
    ProblemCache{T}(; kwargs...)
end

function val_and_jac!(r, J, P, Δt, prob::NonlinearProblem, n)

    @unpack Vwprev, Voprev, Vw, Vo, CTC = prob.cache    
    @unpack Vwi, Voi, cwf, cof, Qsum, diagJ = prob.cache
    @unpack Pi, Tconst = @inbounds prob.pviews[n]

    # Residual and Jacobian initialization
    mul!(r, CTC, P)
    copyto!(J, CTC)

    @turbo for i = 1:length(P)
        # Pore volumes of water and oil in surface conditions
        Vw[i] = Vwi[i] * exp(cwf[i] * (P[i] - Pi[i]))
        Vo[i] = Voi[i] * exp(cof[i] * (P[i] - Pi[i]))

        # Update the residual vector until the condition is met ∥r∥ ≈ 0
        r[i] += (Vw[i] - Vwprev[i] + Vo[i] - Voprev[i]) / Δt
        r[i] += Tconst[i] * (P[i] - Pi[i]) + Qsum[i]

        # In general, the Jacobian is static except for the diagonal elements
        diagJ[i] = Tconst[i] + (Vw[i] * cwf[i] + Vo[i] * cof[i]) / Δt
    end

    # Update the main diagonal of the Jacobian
    # TODO: We took it out from previous loop, because macro '@turbo' does not work with 'Sparse Arrays'
    @inbounds @simd for i = 1:length(diagJ)        
        J[i, i] += diagJ[i]
    end    

    return r, J
end