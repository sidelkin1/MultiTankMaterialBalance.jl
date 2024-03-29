function read_rates(path, opts)
    
    types = Dict(
        :Field => String,
        :Tank => String,
        :Reservoir => String,
        :Date => Date,
        :Qoil => Float64,
        :Qwat => Float64,
        :Qliq => Float64,
        :Qinj => Float64,
        :Pres => Float64,
        :Source_resp => String,
        :Pbhp_prod => Float64,
        :Pbhp_inj => Float64,
        :Source_bhp => String,
        :Wellwork => String,
        :Wcut => Float64,
        :Total_mobility => Float64,
        :Wres => Float64,
        :Wbhp_prod => Float64,
        :Wbhp_inj => Float64,
        :Pres_min => Float64,
        :Pres_max => Float64        
    )
    # TODO: Forced column conversion to 'PooledArray' removed (albeit memory inefficient)
    opts_csv = Dict(
        :dateformat => opts["dateformat"],
        :delim => opts["delim"],
        :types => types,
        :pool => false
    )
    df = CSV.File(opencsv(path); opts_csv...) |> DataFrame

    # Removing weights for missed pressure measurements
    @with df begin
        @. :Wres[ismissing(:Pres)] = missing
        @. :Wbhp_prod[ismissing(:Pbhp_prod)] = missing
        @. :Wbhp_inj[ismissing(:Pbhp_inj)] = missing
    end

    # Sort by dates
    sort!(df, [:Field, :Tank, :Date])

    return df
end

function read_params(path, opts)

    types = Dict(
        :Field => String,
        :Tank => String,
        :Neighb => String,
        :Reservoir => String,
        :Parameter => String,
        :Date => Date,
        :Init_value => Float64,
        :Min_value => Float64,
        :Max_value => Float64,
        :alpha => Float64
    )
    # TODO: Forced column conversion to 'PooledArray' removed (albeit memory inefficient)
    opts_csv = Dict(
        :dateformat => opts["dateformat"],
        :delim => opts["delim"],
        :types => types,
        :pool => false
    )
    df = CSV.File(opencsv(path); opts_csv...) |> DataFrame
    
    # Fill in the missing values
    crit = ismissing.(df.Neighb)
    df[crit, :Neighb] .= df[crit, :Tank]
    # TODO: To convert 'Union{Missing, String}' to 'String'
    df.Neighb = convert.(String, df.Neighb)    

    # Convert 'String' to 'Symbol'
    df.Parameter = Symbol.(df.Parameter)
    replace!(df.Parameter, PSYMS...)

    # Sort by dates
    @eachrow!(df, (:Tank, :Neighb) = sort!([:Tank, :Neighb]))
    sort!(df, [:Parameter, :Field, :Tank, :Neighb, :Date])

    return df
end

function mark_null_params!(df::AbstractDataFrame, name::Symbol, isnull)
    df_view = getparams(df, Val(name))
    df_view[!, :Ignore] .|= @with df_view @byrow begin
        rows = UnitRange(:Jstart::Int, :Jstop::Int)
        all(isnull[rows, :Istart::Int])
    end
    df_view[!, :Const] .|= df_view.Ignore
    return df
end

function set_null_weights!(weight, df::AbstractDataFrame, name::Symbol)    
    df_view = getparams(df, Val(name), Val(:const))
    @with df_view @byrow begin
        rows = UnitRange(:Jstart::Int, :Jstop::Int)
        # TODO: Multiplication selected for 'missing propagation',
        # i.e. (missing * number => missing)
        weight[rows, :Istart::Int] .*= zero(eltype(weight))
    end
    return weight
end

function process_params!(df_params::AbstractDataFrame, df_rates::AbstractDataFrame)

    # Numbering tank names and dates
    numberof(data) = @_ data |> 
                    unique(__) |> 
                    Dict(__ .=> 1:length(__))
    tanks = numberof(df_rates.Tank::Vector{String})
    dates = numberof(df_rates.Date::Vector{Date})
    N = length(dates)

    # Add the tank number and the beginning of the parameter period
    @transform! df_params begin
        :Istart = getindex.(Ref(tanks), :Tank::Vector{String})        
        :Jstart = getindex.(Ref(dates), :Date::Vector{Date})
    end

    # For connection parameters, set the connection number
    df = getparams(df_params, Val(:Tconn))
    @with df begin
        connections = numberof(eachrow([:Tank :Neighb]::Matrix{String}))
        :Istart .= getindex.(Ref(connections), eachrow([:Tank :Neighb]))
    end
    
    # Adding the end of the parameter period
    df = groupby(df_params, [:Parameter, :Field, :Tank, :Neighb])
    @transform!(df, :Jstop = [:Jstart[2:end] .- 1; N])
    
    # TODO: Additionally, we split (if required) the intervals of fixing 'Jinj',
    # so that within each interval the value of 'λ' would not change.
    # This makes it much easier to differentiate the objective function with respect to 'P'
    split_params!(df_params, :λ, :Jinj)

    # Marking parameters that do not require fitting
    @transform!(df_params, :Const = :Min_value .== :Max_value)
    @transform!(df_params, :Ignore = :Const .& (:Init_value .≠ :Min_value))
    
    # Some parameters can be immediately excluded from the fitting
    crits = @with df_rates begin (
        λ = (:Qinj .== 0)::BitVector,
        Jp = (@. (ismissing(:Pbhp_prod) 
                | ismissing(:Wbhp_prod) 
                | (:Wbhp_prod == 0) 
                | (:Qliq == 0)))::BitVector,
        Jinj = (@. (ismissing(:Pbhp_inj) 
                | ismissing(:Wbhp_inj) 
                | (:Wbhp_inj == 0) 
                | (:Qinj == 0)))::BitVector,
    ) end
    for (key, values) ∈ pairs(crits)
        mark_null_params!(df_params, key, reshape(values, N, :))
    end

    # Forcing weights of measurements for ignored parameters to zero (Ignore == true)
    crits = @with df_rates begin (
        Jp = :Wbhp_prod,
        Jinj = :Wbhp_inj,
    ) end
    for (key, values) ∈ pairs(crits)
        set_null_weights!(reshape(values, N, :), df_params, key)
    end

    # Instead of well's PI we add well's geometric factor to fit
    crits = @with df_rates begin (
        Jp = :Total_mobility,
    ) end
    for (key, values) ∈ pairs(crits)
        add_geom_factor!(df_params, key, reshape(values, N, :))
    end

    return df_params, df_rates
end

function save_rates!(df::AbstractDataFrame, prob::NonlinearProblem, path, opts)
    
    @transform! df begin
        :Pres_calc = vec(prob.params.Pcalc')
        :Pbhp_calc = vec(prob.params.Pbhp')
        :Pinj_calc = vec(prob.params.Pinj')
        :Geom_factor = vec(prob.params.Gw')
        :Prod_Index = vec(prob.params.Jp')
        :Inj_index = vec(prob.params.Jinj')
        :Frac_inj = vec(prob.params.λ')
        :Qliq_calc = vec(prob.params.Qliq')
        :Qinj_calc = vec((prob.params.Qinj .* prob.params.λ)')
    end

    open(path, CSV_ENC, "w") do io
        CSV.write(io, df; delim=opts["delim"], dateformat=opts["dateformat"])
    end

    return df
end

function save_params!(df::AbstractDataFrame, fset::FittingSet, targ::TargetFunction, path, opts)

    @transform!(df, :Calc_value = :Init_value)    

    # Fill in tank parameters
    df_view = getparams(df, Val(:tanks), Val(:var))
    copyto!(df_view.Calc_value, fset.cache.ybuf)

    # Fill in the parameters of the wells
    df_view = getparams(df, Val(:Gw), Val(:var))    
    copyto!(df_view.Calc_value, targ.wviews.Gw.yviews)
    df_view = getparams(df, Val(:Jinj), Val(:var))
    copyto!(df_view.Calc_value, targ.wviews.Jinj.yviews)
    df_view = getparams(df, Val(:Jp), Val(:var))
    copyto!(df_view.Calc_value, targ.wviews.Jp.yviews)

    # Rename and remove extra columns
    replace!(df.Parameter, reverse.(PSYMS)...)
    select!(df, Not([:Istart, :Jstart, :Jstop, :Link, :Const, :Ignore])) 

    open(path, CSV_ENC, "w") do io
        CSV.write(io, df; delim=opts["delim"], dateformat=opts["dateformat"])
    end

    return df
end

function split_params!(df::AbstractDataFrame, src::Symbol, dst::Symbol)

    df_src = getparams(df, Val(src))
    df_dst = getparams(df, Val(dst))

    df_new = map(eachrow(df_dst)) do dfr
        # Interval intersection criterion
        crit = @with df_src begin
            @. (dfr.Istart == :Istart) & 
                (
                    ((dfr.Jstart >= :Jstart) & 
                        (dfr.Jstart <= :Jstop))
                    |
                    ((dfr.Jstart <= :Jstart) & 
                        (dfr.Jstop >= :Jstart))
                )
        end        
        # Split based on intersection criterion
        @transform! df_src[crit, :] begin
            # Duplicate information partially
            :Parameter = dfr.Parameter
            :Init_value = dfr.Init_value
            :Min_value = dfr.Min_value
            :Max_value = dfr.Max_value
            :alpha = dfr.alpha
            # Correcting the beginning and end of the fixing period
            :Date = [dfr.Date; :Date[2:end]]
            :Jstart = [dfr.Jstart; :Jstart[2:end]]
            :Jstop = [:Jstop[1:end-1]; dfr.Jstop]
            # Link to source
            :Link = axes(df_src, 1)[crit]
        end
    end |> items -> vcat(items...)

    # Forming a new list of fixing intervals
    deleteat!(df, df.Parameter .=== dst)
    append!(df, df_new; cols=:union)

    return df
end

function add_geom_factor!(df::AbstractDataFrame, name::Symbol, mobt)
       
    # Copy PI to calculate the geometric factor
    df_new = df[df.Parameter .=== name, :]

    # Calculate well geometric factors
    @eachrow! df_new begin
        # Fluid mobility over time
        rows = UnitRange(:Jstart::Int, :Jstop::Int)
        mobt_ = @view mobt[rows, :Istart::Int]

        # We do mobility equals to one if constant PI is required
        :Const && (mobt_ .= one(eltype(mobt)))

        # Current, min and max mobilities
        Minit = first(mobt_)
        Mmin, Mmax = extrema(mobt_)        

        # Mapping PI => geometric factor
        :Init_value /= Minit
        :Min_value /= Mmin
        :Max_value /= Mmax
        
        # Handle the case when Gmin > Gmax
        if :Max_value < :Min_value
            # Correct min/max mobilities that is Gmin == Gmax
            Mmin *= √(:Min_value / :Max_value)
            Mmax *= √(:Max_value / :Min_value)            
            clamp!(mobt_, Mmin, Mmax)
            # Calculate new Gmin, Gmax
            :Min_value = :Max_value = √(:Min_value * :Max_value)            
        end

        # Putting values within admissible limits
        :Init_value = clamp(:Init_value, :Min_value, :Max_value)        
        :Const |= :Min_value .== :Max_value
    end

    # Removing old information (if presented)
    deleteat!(df, df.Parameter .=== :Gw)
    # Add well geometric factors
    df_new[!, :Parameter] .= :Gw
    append!(df, df_new)

    return df
end