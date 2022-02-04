module MultiTankMaterialBalance

export psyms, Float, read_rates, read_params, process_params!, init, solve!
export SparseNewtonAlgorithm, DenseNewtonAlgorithm, NonlinearProblem, FittingSet, TargetFunction, AdjointSolver

using CSV
using DataFrames
using DataFramesMeta
using StringEncodings
using Dates
using Underscores
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using UnPack
using Printf
using SuiteSparse
using RecursiveFactorization

const Float = Float64

const psyms = (
    :Tconn => :Tconn, :Pi => :Pi, :Bwi => :Bwi, :Boi => :Boi, 
    :cw => :cw, :co => :co, :cf => :cf, :Swi => :Swi, :Vpi => :Vpi, 
    :Tconst => :Tconst, :Prod_index => :Jp, :Inj_index => :Jinj, 
    :Frac_inj => :λ, :Min_Pres => :Pmin, :Max_Pres => :Pmax,
)

include("types.jl")
include("utils.jl")
include("data.jl")
include("problem.jl")
include("solve.jl")
include("newton.jl")
include("parameters.jl")
include("target.jl")
include("adjoint.jl")

end
