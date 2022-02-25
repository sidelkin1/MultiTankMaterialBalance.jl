using MultiTankMaterialBalance
using FiniteDiff
using Test

const Float = Float64

# Функция для вычисления значения и градиента целевой функции
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
    # Путь к csv-файлам с исходными данными
    workdir = "./data"    

    # Исходные данные для расчета
    opts_csv = Dict("dateformat" => "dd.mm.yyyy", "delim" => ";")
    df_rates = read_rates(joinpath(workdir, "tank_prod.csv"), opts_csv)
    df_params = read_params(joinpath(workdir, "tank_params.csv"), opts_csv)

    # Дополнительная предобработка параметров
    process_params!(df_params, df_rates)

    # Описание прямой задачи
    prob = NonlinearProblem{Float}(df_rates, df_params)

    # Способ решения СЛАУ
    linalg = DenseLinearSolver{Float}(prob)

    # Алгоритм решения прямой задачи
    opts_nsol = Dict("maxiters" => 10, "P_tol" => 1e-7, "r_tol" => 1e-5)
    solver = NewtonSolver{Float}(prob, linalg, opts_nsol)
    
    solve!(solver)
    @test solver.success

    # Способ масштабирования параметров
    scale = SigmoidScaling{Float}(df_params)

    # Список оптимизируемых параметров
    fset = FittingSet{Float}(df_params, prob, scale)

    # Целевая функция
    opts_targ = Dict("alpha_resp" => 1, "alpha_bhp" => 0.01, "alpha_inj" => 0.01, "alpha_lb" => 10, "alpha_ub" => 10, "alpha_l2" => 1)
    targ = TargetFunction{Float}(df_rates, df_params, prob, fset, opts_targ)

    # Алгоритм расчета градиента целевой функции
    adjoint_ = AdjointSolver{Float}(prob, targ, linalg, fset)

    # Функция для вычисления значения и градиента целевой функции
    optim_fun = fun(fset, solver, targ, adjoint_)

    # Начальное значение параметров
    initial_x = copy(getparams!(fset))
    # Массив для хранения градиента
    grad = similar(initial_x)

    # Вычисление градиента методом конечных разностей
    grad_fd = FiniteDiff.finite_difference_gradient(x -> optim_fun(x, Float[]), initial_x, Val(:central))

    # Вычисление градиента методом сопряженных уравнений
    _ = optim_fun(initial_x, grad)

    # Сравниваем градиенты, вычисленные различными способами
    @test all(isapprox.(grad, grad_fd; rtol=0.0001))
end