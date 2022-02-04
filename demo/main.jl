using MultiTankMaterialBalance, DataFramesMeta, StringEncodings, CSV

dateformat = "dd.mm.yyyy"
dir = @__DIR__
df_rates = read_rates(joinpath(dir, "tank_prod.csv"), dateformat)
df_params = read_params(joinpath(dir, "tank_params.csv"), dateformat, psyms)
process_params!(df_params, df_rates)

α = (Pres=1., Pbhp=0.01, Pinj=0.01, Pmin=10., Pmax=10., L2=0.01)
if Float === Float64
    alg = SparseNewtonAlgorithm{Float}()
    P_tol = Float(1e-7)
    r_tol = Float(1e-5)
else
    alg = DenseNewtonAlgorithm{Float}()
    P_tol = Float(1e-5)
    r_tol = Float(1e-3)
end
maxiters = 10

prob = NonlinearProblem{Float}(df_rates, df_params)
fset = FittingSet{Float}(df_params, prob)
targ = TargetFunction{Float}(df_rates, prob, fset, α)
solver = init(prob, alg; maxiters, P_tol, r_tol)
adjoint = AdjointSolver{Float}(prob, targ)

# function simrun(prob, maxiters, P_tol, r_tol)
#     prob2 = deepcopy(prob)
#     solve!(prob2, alg; maxiters, P_tol, r_tol)    
# end
# @btime simrun($prob, $maxiters, $P_tol, $r_tol)




solve!(solver)
@transform! df_rates begin
    :Pcalc = vec(prob.params.Pcalc')
    :Qliq_calc = vec(prob.params.Qliq')
    :Qinj_calc = vec(prob.params.Qinj')
    :Frac_inj = vec(prob.params.λ')
    :Pbhp = vec(prob.params.Pbhp')
    :Pinj = vec(prob.params.Pinj')
end

solve!(adjoint)

open(joinpath(dir, "results.csv"), enc"WINDOWS-1251", "w") do io
    CSV.write(io, df_rates; delim=";", dateformat)
end

println("Done!")