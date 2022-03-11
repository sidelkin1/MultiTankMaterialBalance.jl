using MultiTankMaterialBalance
using FiniteDiff
using Test

const Float = Float64

# Function to calculate the value and gradient of the objective function
function fun(fset::FittingSet, solver::NewtonSolver, targ::TargetFunction, adjoint::AdjointSolver)
    return (x, grad) -> begin
        setparams!(fset, x)
        solve!(solver)
        update_targ!(targ)
        if length(grad) > 0
            solve!(adjoint)
            copyto!(grad, adjoint.g)
        end
        return getvalue(targ)
    end    
end

@testset "MultiTankMaterialBalance.jl" begin
    # Path to csv files with source data
    workdir = "./data"    

    # Initial data for calculation
    opts_csv = Dict("dateformat" => "dd.mm.yyyy", "delim" => ";")
    df_rates = read_rates(joinpath(workdir, "tank_prod.csv"), opts_csv)
    df_params = read_params(joinpath(workdir, "tank_params.csv"), opts_csv)

    # Additional parameter preprocessing
    process_params!(df_params, df_rates)

    # Description of the forward problem
    prob = NonlinearProblem{Float}(df_rates, df_params)

    # Linear equation solver
    linalg = DenseLinearSolver{Float}(prob)

    # Algorithm to solve forward problem
    opts_nsol = Dict("maxiters" => 10, "P_tol" => 1e-7, "r_tol" => 1e-5)
    solver = NewtonSolver{Float}(prob, linalg, opts_nsol)
    
    # Solve forward problem
    solve!(solver)
    @test solver.success

    # Parameter scaling method
    scale = SigmoidScaling{Float}(df_params)

    # Fitting parameter list
    fset = FittingSet{Float}(df_params, prob, scale)

    # Objective function
    opts_targ = Dict("alpha_resp" => 1, "alpha_bhp" => 0.01, "alpha_inj" => 0.01, "alpha_lb" => 10, "alpha_ub" => 10, "alpha_l2" => 1)
    targ = TargetFunction{Float}(df_rates, df_params, prob, fset, opts_targ)

    # Algorithm to calculate objective function gradient
    adjoint_ = AdjointSolver{Float}(prob, targ, linalg, fset)

    # Function to calculate the value and gradient of the objective function
    optim_fun = fun(fset, solver, targ, adjoint_)

    # Initial parameter values
    initial_x = copy(getparams!(fset))
    # Preallocate gradient buffer
    grad = similar(initial_x)

    # Computing the gradient by the finite difference method
    grad_fd = FiniteDiff.finite_difference_gradient(x -> optim_fun(x, Float[]), initial_x, Val(:central); relstep=1e-5)

    # Computing the gradient by the adjoint equation method
    _ = optim_fun(initial_x, grad)

    # Comparing gradients computed in different ways
    @test all(isapprox.(grad, grad_fd; rtol=0.0001))
end