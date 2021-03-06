module MultiTankMaterialBalance

export read_rates, read_params, process_params!, save_rates!, save_params!
export getvalue, getvalues, getparams!, getparams!!, setparams!, solve!, update_targ!, calc_well_index!
export NonlinearProblem, LinearScaling, SigmoidScaling
export DenseLinearSolver, RecursiveLinearSolver, SparseLinearSolver
export NewtonSolver, FittingSet, TargetFunction, AdjointSolver

using CSV
using DataFrames
using DataFramesMeta
using StringEncodings
using Dates
using Underscores
using LinearAlgebra
using SparseArrays
using UnPack
using Printf
using SuiteSparse
using RecursiveFactorization
using AMD
using Metis
using LoopVectorization

include("types.jl")
include("utils.jl")
include("problem.jl")
include("linalg.jl")
include("solve.jl")
include("newton.jl")
include("scale.jl")
include("parameters.jl")
include("target.jl")
include("gradient.jl")
include("adjoint.jl")
include("data.jl")

end