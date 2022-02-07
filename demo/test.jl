using MultiTankMaterialBalance
using ArgParse
using JSON
using FiniteDiff
using BenchmarkTools

const dir = @__DIR__

push!(ARGS, "--options", joinpath(dir, "options.json"),
            "--tank_params",  joinpath(dir, "tank_params.csv"),
            "--tank_prod",  joinpath(dir, "tank_prod.csv"),
            "--result_params", joinpath(dir, "result_params.csv"), 
            "--result_prod",  joinpath(dir,"result_prod.csv"))

include("optimize.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--options"
            help = "Путь к файлу с параметрами расчета"
            arg_type = String
            required = true
        "--tank_params"
            help = "Путь к файлу с параметрами блоков"
            arg_type = String
            required = true
        "--tank_prod"
            help = "Путь к файлу с отборами внутри блоков"
            arg_type = String
            required = true
        "--result_params"
            help = "Путь к файлу с результатами адаптации параметров блоков"
            arg_type = String
            required = true
        "--result_prod"
            help = "Путь к файлу с расчетными давлениями внутри блоков"
            arg_type = String
            required = true
    end

    return parse_args(ARGS, s)
end

parsed_args = parse_commandline()

# Параметры для расчета
opts = JSON.parsefile(parsed_args["options"])

# Разрядность формата вещественных чисел
Float = eval(Meta.parse(opts["float"]))

# Исходные данные для расчета
df_rates = read_rates(parsed_args["tank_prod"], opts["csv"])
df_params = read_params(parsed_args["tank_params"], opts["csv"])
process_params!(df_params, df_rates)

# Описание прямой задачи
prob = NonlinearProblem{Float}(df_rates, df_params)

# Способ масштабирования параметров
if opts["optimizer"]["scaling"] == "linear"
    scale = LinearScaling{Float}(df_params)
elseif opts["optimizer"]["scaling"] == "sigmoid"
    scale = SigmoidScaling{Float}(df_params)
end

# Способ решениия СЛАУ
if opts["solver"]["linalg"] == "dense"
    linalg = DenseLinearSolver{Float}(prob)
elseif opts["solver"]["linalg"] == "recursive"
    linalg = RecursiveLinearSolver{Float}(prob)
elseif opts["solver"]["linalg"] == "sparse"
    reorder = Symbol(opts["solver"]["reorder"])
    linalg = SparseLinearSolver{Float}(prob; reorder)
end

# Алгоритм решения прямой задачи
solver = NewtonSolver{Float}(prob, linalg, opts["solver"])
# Список оптимизируемых параметров
fset = FittingSet{Float}(df_params, prob, scale)
# Целевая функция
targ = TargetFunction{Float}(df_rates, df_params, prob, fset, opts["target_fun"])
# Алгоритм расчета градиента целевой функции
adjoint = AdjointSolver{Float}(prob, targ, linalg, fset)

# Адаптация модели
optim_pkg = Symbol(opts["optimizer"]["package"])
maxiters = opts["optimizer"]["maxiters"]
optim_opts = opts["optimizer"][String(optim_pkg)]
optim_fun = fun(fset, solver, targ, adjoint, Val(optim_pkg))
initial_x = copy(getparams!(fset))

grad = similar(initial_x)
fd_g = FiniteDiff.finite_difference_gradient(x -> optim_fun(x, grad), initial_x, Val(:central); absstep=1e-12)

setparams!(fset, initial_x)
solve!(solver; verbose=true)
update_targ!(targ)
solve!(adjoint; verbose=true)
calc_well_index!(targ)

val = optim_fun(initial_x, grad)
calc_well_index!(targ)

println(val)
println(getvalues(targ))
println(grad)
println(fd_g)
# @btime solve!($solver)
# @btime solve!($adjoint)

# Сохраняем результаты
save_rates!(df_rates, prob, parsed_args["result_prod"], opts["csv"])
save_params!(df_params, fset, targ, parsed_args["result_params"], opts["csv"])

nothing