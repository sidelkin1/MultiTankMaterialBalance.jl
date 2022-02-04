module MultiTankMaterialBalance

export read_rates, read_params, process_params!, save_rates!, save_params!
export getvalue, getvalues, getparams!, getparams!!, setparams!, solve!, update_targ!
export NonlinearProblem, LinearScaling, SigmoidScaling
export DenseLinearSolver, RecursiveLinearSolver, SparseLinearSolver
export NewtonSolver, FittingSet, TargetFunction, AdjointSolver

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
using AMD
using Metis

const PSYMS = (
    :Tconn => :Tconn, :Pi => :Pi, :Bwi => :Bwi, :Boi => :Boi, 
    :cw => :cw, :co => :co, :cf => :cf, :Swi => :Swi, :Vpi => :Vpi, 
    :Tconst => :Tconst, :Prod_index => :Jp, :Inj_index => :Jinj, 
    :Frac_inj => :λ, :Min_Pres => :Pmin, :Max_Pres => :Pmax,
)

const CSV_ENC = enc"WINDOWS-1251"

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