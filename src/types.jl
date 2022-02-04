abstract type AbstractLinearSolver{T<:AbstractFloat} end
abstract type AbstractNonlinearSolver{T<:AbstractFloat} end

abstract type AbstractTargetTerm{S, T<:AbstractFloat} end
abstract type AbstractFittingParameter{T<:AbstractFloat} end
abstract type AbstractParametersScaling{T<:AbstractFloat} end

const ColumnSlice{T} = SubArray{T, 1, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}
const ColumnSliceBool = SubArray{Bool, 1, BitMatrix, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}
const RowRange{T} = SubArray{T, 1, Matrix{T}, Tuple{Int, UnitRange{Int}}, true}
const CartesianView{T} = SubArray{T, 1, Matrix{T}, Tuple{Vector{CartesianIndex{2}}}, false}
const VectorRange{T} = SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
const VectorView{T} = SubArray{T, 1, Vector{T}, Tuple{Vector{Int}}, false}