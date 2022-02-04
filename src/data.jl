function read_rates(path, opts)
    
    types = Dict(
        :Field => String,
        :Tank => String,
        :Reservoir => String,
        :Date => Date,
        :Qoil => Float64,
        :Qwat => Float64,
        :Qinj => Float64,
        :Pres => Float64,
        :Source_resp => String,
        :Pbhp_prod => Float64,
        :Pbhp_inj => Float64,
        :Source_bhp => String,
        :Wellwork => String,
        :Wcut => Float64,
        :Wres => Float64,
        :Wbhp_prod => Float64,
        :Wbhp_inj => Float64,
        :Pres_min => Float64,
        :Pres_max => Float64        
    )
    # TODO: Принудительно убрано преобразование столбцов в 'PooledArray' (хотя неэффективно с точки зрения памяти)
    df = CSV.File(opencsv(path); dateformat=opts["dateformat"], types, pool=false) |> DataFrame

    # Убираем веса на пропущенные замеры давлений
    @with df begin
        @. :Wres[ismissing(:Pres)] = missing
        @. :Wbhp_prod[ismissing(:Pbhp_prod)] = missing
        @. :Wbhp_inj[ismissing(:Pbhp_inj)] = missing
    end

    # Сортируем по датам
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
    # TODO: Принудительно убрано преобразование столбцов в 'PooledArray' (хотя неэффективно с точки зрения памяти)
    df = CSV.File(opencsv(path); dateformat=opts["dateformat"], types, pool=false) |> DataFrame
    
    # Заполняем пропущенные значения
    crit = ismissing.(df.Neighb)
    df[crit, :Neighb] .= df[crit, :Tank]
    # TODO Для преобразования 'Union{Missing, String}' в 'String'
    df.Neighb = convert.(String, df.Neighb)

    # Преобразуем 'String' в 'Symbol'
    df.Parameter = Symbol.(df.Parameter)
    replace!(df.Parameter, PSYMS...)

    # Коэффициент регуляризации на параметры (по умолчанию 0, если не задан)
    df.alpha = replace(df.alpha, missing => zero(types[:alpha]))

    # Сортируем по датам
    @eachrow!(df, (:Tank, :Neighb) = sort!([:Tank, :Neighb]))
    sort!(df, [:Parameter, :Field, :Tank, :Neighb, :Date])

    return df
end

function mark_null_params!(df_params, name, isnull)
    df = @view df_params[df_params.Parameter .=== name, :]
    df.Skip .= @with df @byrow begin
        rows = UnitRange(:Jstart::Int, :Jstop::Int)
        :Skip | all(isnull[rows, :Istart::Int])
    end
    return df_params
end

function set_null_weights!(weight, df_params, name)
    df = @subset(df_params, :Parameter .=== name, :Skip .=== true)
    @with df @byrow begin
        rows = UnitRange(:Jstart::Int, :Jstop::Int)
        # TODO: Выбрано умножение для 'missing propagation',
        # т.е. (missing * число === missing)
        weight[rows, :Istart::Int] .*= zero(eltype(weight))
    end
end

function process_params!(df_params, df_rates)

    # Нумеруем названия блоков и даты
    numberof(data) = @_ data |> 
                    unique(__) |> 
                    Dict(__ .=> 1:length(__))
    tanks = numberof(df_rates.Tank::Vector{String})
    dates = numberof(df_rates.Date::Vector{Date})
    N = length(dates)

    # Добавляем номер блока и начало периода параметра
    @transform! df_params begin
        :Istart = getindex.(Ref(tanks), :Tank::Vector{String})        
        :Jstart = getindex.(Ref(dates), :Date::Vector{Date})
    end

    # Для параметров соединений ставим номер соединения
    df = @view df_params[df_params.Parameter .=== :Tconn, :]
    @with df begin
        connections = numberof(eachrow([:Tank :Neighb]::Matrix{String}))
        :Istart .= getindex.(Ref(connections), eachrow([:Tank :Neighb]))
    end
    
    # Добавляем конец периода параметра
    df = groupby(df_params, [:Parameter, :Field, :Tank, :Neighb])
    @transform! df begin
        :Jstop = [:Jstart[2:end] .- 1; N]
    end

    # TODO: Дополнительно дробим (если требуется) интервалы закрепления 'Jinj',
    # чтобы внутри каждого интервале значение 'λ' не менялось бы.
    # Это значительно упрощает дифференцирование целевой функции по 'P'
    split_params!(df_params, :λ, :Jinj)

    # Помечаем параметры, не требующие настройки
    @transform! df_params begin
        :Skip = :Min_value .== :Max_value
    end
    # Часть параметров можно сразу исключать из настройки
    crits = @with df_rates begin (
        λ = (:Qinj .== 0)::BitVector,
        Jp = (@. (ismissing(:Pbhp_prod) 
                | ismissing(:Wbhp_prod) 
                | (:Wbhp_prod == 0) 
                | ((:Qoil + :Qwat) == 0)))::BitVector,
        Jinj = (@. (ismissing(:Pbhp_inj) 
                | ismissing(:Wbhp_inj) 
                | (:Wbhp_inj == 0) 
                | (:Qinj == 0)))::BitVector,
    ) end
    for (key, values) ∈ pairs(crits)
        mark_null_params!(df_params, key, reshape(values, N, :))
    end

    # TODO: Принудительно зануляются веса замеров для неизменяемых 
    # параметров, а значит они не будут включаться в расчет целевой функции.
    # Такое поведение может оказаться не приемлемым, если нам точно известно 
    # значение параметра 'Jp' (т.е. он должен быть фиксированным), но при этом
    # требуется подгонка профиля 'P' под фактические профили 'Qliq' и 'Pbhp'
    crits = @with df_rates begin (
        Jp = :Wbhp_prod,
        Jinj = :Wbhp_inj,
    ) end
    for (key, values) ∈ pairs(crits)
        set_null_weights!(reshape(values, N, :), df_params, key)
    end

    return df_params, df_rates
end

function save_rates!(df::AbstractDataFrame, prob::NonlinearProblem, path, opts)

    @transform! df begin
        :Pres_calc = vec(prob.params.Pcalc')
        :Pbhp_calc = vec(prob.params.Pbhp')
        :Pinj_calc = vec(prob.params.Pinj')
        :Prod_index = vec(prob.params.Jp')
        :Inj_index = vec(prob.params.Jinj')
        :Frac_inj = vec(prob.params.λ')
        :Tot_mobility = missing
        :Prod_index_adj = missing
    end

    open(path, CSV_ENC, "w") do io
        CSV.write(io, df; delim=opts["delim"], dateformat=opts["dateformat"])
    end

    return df
end

function save_params!(df::AbstractDataFrame, fset::FittingSet, path, opts)

    @transform!(df, :Calc_value = :Init_value)
    df_view = @view df[df.Skip .=== false, :]
    copyto!(df_view.Calc_value, fset.cache.ybuf)    

    replace!(df.Parameter, reverse.(PSYMS)...)
    select!(df, Not([:Istart, :Jstart, :Jstop, :Skip])) 

    open(path, CSV_ENC, "w") do io
        CSV.write(io, df; delim=opts["delim"], dateformat=opts["dateformat"])
    end

    return df
end

function split_params!(df::AbstractDataFrame, src::Symbol, dst::Symbol)
    df_src = @view df[df.Parameter .=== src, :]
    df_dst = @view df[df.Parameter .=== dst, :]

    df_new = map(eachrow(df_dst)) do dfr
        # Критерий пересечения интервалов
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
        # Разбивка с учетом критерия пересечения
        @transform! df_src[crit, :] begin
            # Дублируем частично информацию 
            :Parameter = dfr.Parameter
            :Init_value = dfr.Init_value
            :Min_value = dfr.Min_value
            :Max_value = dfr.Max_value
            :alpha = dfr.alpha
            # Корректируем начало и конец периода закрепления
            :Date = [dfr.Date; :Date[2:end]]
            :Jstart = [dfr.Jstart; :Jstart[2:end]]
            :Jstop = [:Jstop[1:end-1]; dfr.Jstop]
        end
    end |> items -> vcat(items...)

    # Формируем новый список интервалов закрепления
    delete!(df, df.Parameter .=== dst)
    append!(df, df_new)
end