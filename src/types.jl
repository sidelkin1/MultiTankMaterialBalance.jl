abstract type AbstractSolveAlgorithm end
struct BFGSAlgorithm{M<:AbstractMatrix} <: AbstractSolveAlgorithm end
struct NewtonAlgorithm{M<:AbstractMatrix} <: AbstractSolveAlgorithm end
abstract type AbstractNonlinearSolver{T<:AbstractFloat, M<:AbstractMatrix} end

const DenseNewtonAlgorithm{T<:AbstractFloat} = NewtonAlgorithm{Matrix{T}}
const SparseNewtonAlgorithm{T<:AbstractFloat} = NewtonAlgorithm{SparseMatrixCSC{T, Int}}

abstract type AbstractTargetTerm{T<:AbstractFloat} end
abstract type AbstractFittingParameter{T<:AbstractFloat} end

const ColumnSlice{T} = SubArray{T, 1, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}
const ColumnSliceBool = SubArray{Bool, 1, BitMatrix, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}
const RowRange{T} = SubArray{T, 1, Matrix{T}, Tuple{Int, UnitRange{Int}}, true}
const CartesianView{T} = SubArray{T, 1, Matrix{T}, Tuple{Vector{CartesianIndex{2}}}, false}
const VectorView{T} = SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}