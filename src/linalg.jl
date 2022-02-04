mutable struct DenseLinearSolver{T<:AbstractFloat} <: AbstractLinearSolver{T}
    Abuf::Vector{Matrix{T}}
    Fbuf::Vector{CholeskyPivoted{T, Matrix{T}}}
end

struct RecursiveLinearSolver{T<:AbstractFloat} <: AbstractLinearSolver{T}
    Abuf::Vector{Matrix{T}}
    Fbuf::Vector{LU{T, Matrix{T}}}
    threshold::Int
end

struct SparseLinearSolver{T<:AbstractFloat} <: AbstractLinearSolver{T}
    A::SparseMatrixCSC{T, Int}
    Fsym::SuiteSparse.CHOLMOD.Factor{T}
    Fbuf::Vector{SuiteSparse.CHOLMOD.Factor{T}}
    perm::Union{Nothing, Vector{Int}}
end

getjac(solver::AbstractLinearSolver, n) = nothing
getjac(solver::DenseLinearSolver, n) = @inbounds solver.Abuf[n]
getjac(solver::RecursiveLinearSolver, n) = @inbounds solver.Abuf[n]
getjac(solver::SparseLinearSolver, n) = solver.A

setjac!(solver::AbstractLinearSolver, n) = nothing
setjac!(solver::SparseLinearSolver, n) = !isnothing(solver.perm) && @inbounds solver.Fbuf[n] = copy(solver.Fsym)

getperm(prob, ::Val{S}) where {S} = nothing
getperm(prob, ::Val{:none}) = 1:size(prob.C, 2)
getperm(prob, ::Val{:amd}) = AMD.symamd(prob.C' * prob.C)
getperm(prob, ::Val{:metis}) = Int.(Metis.permutation(prob.C' * prob.C)[1])

function DenseLinearSolver{T}(prob::NonlinearProblem{T}) where {T}
    Nt = size(prob.C, 2)
    Nd = length(prob.pviews)
    Abuf = map(_ -> diagm(ones(T, Nt)), 1:Nd)
    Fbuf = map(A -> cholesky!(A, Val(true)), Abuf)
    DenseLinearSolver{T}(Abuf, Fbuf)
end

function RecursiveLinearSolver{T}(prob::NonlinearProblem{T}; threshold=100) where {T}
    Nt = size(prob.C, 2)
    Nd = length(prob.pviews)
    Abuf = map(_ -> diagm(ones(T, Nt)), 1:Nd)
    Fbuf = map(A -> RecursiveFactorization.lu!(A, Val(true); threshold), Abuf)
    RecursiveLinearSolver{T}(Abuf, Fbuf, threshold)
end

function SparseLinearSolver{T}(prob::NonlinearProblem{T}; reorder::Symbol=:none) where {T}
    Nt = size(prob.C, 2)
    Nd = length(prob.pviews)
    A = spdiagm(ones(T, Nt))
    perm = getperm(prob, Val(reorder)) # способ переупорядочивания матрицы
    Fsym = cholesky(A; perm) # символьное разложение
    Fbuf = fill(Fsym, Nd)
    SparseLinearSolver{T}(A, Fsym, Fbuf, perm)
end

function solve!(x, b, n, solver::DenseLinearSolver)
    @unpack Abuf, Fbuf = solver
    @inbounds Fbuf[n] = cholesky!(Abuf[n], Val(true))
    # TODO: Почему-то быстрее, чем 'ldiv!'
    x .= @inbounds Fbuf[n] \ b
    return x
end

function solve!(x, b, n, solver::RecursiveLinearSolver)
    @unpack Abuf, Fbuf, threshold = solver
    @inbounds Fbuf[n] = RecursiveFactorization.lu!(Abuf[n], Val(true); threshold)
    # TODO: Почему-то быстрее, чем 'ldiv!'
    x .= @inbounds Fbuf[n] \ b    
    return x
end

function solve!(x, b, n, solver::SparseLinearSolver)
    @unpack A, Fsym, Fbuf, perm = solver
    if isnothing(perm)
        @inbounds Fbuf[n] = cholesky(A)
        x .= @inbounds Fbuf[n] \ b
    else
        # FIXED: Если был задан непустой 'perm', то способ переупорядочивания уже содержится в 'Fsym'
        cholesky!(Fsym, A)
        x .= Fsym \ b
    end    
    return x
end