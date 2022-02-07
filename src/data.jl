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

function mark_null_params!(df::AbstractDataFrame, name::Symbol, isnull)
    df_view = @view df[df.Parameter .=== name, :]
    df_view.Ignore .|= @with df_view @byrow begin
        rows = UnitRange(:Jstart::Int, :Jstop::Int)
        all(isnull[rows, :Istart::Int])
    end
    return df
end

function set_null_weights!(weight, df::AbstractDataFrame, name::Symbol)    
    df_view = @view df[(df.Parameter .=== name) .& df.Ignore, :]
    @with df_view @byrow begin
        rows = UnitRange(:Jstart::Int, :Jstop::Int)
        # TODO: Выбрано умножение для 'missing propagation',
        # т.е. (missing * число === missing)
        weight[rows, :Istart::Int] .*= zero(eltype(weight))
    end
    return weight
end

function process_params!(df_params::AbstractDataFrame, df_rates::AbstractDataFrame)

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
    @transform!(df, :Jstop = [:Jstart[2:end] .- 1; N])
    
    # TODO: Дополнительно дробим (если требуется) интервалы закрепления 'Jinj',
    # чтобы внутри каждого интервале значение 'λ' не менялось бы.
    # Это значительно упрощает дифференцирование целевой функции по 'P'
    split_params!(df_params, :λ, :Jinj)

    # Помечаем параметры, не требующие настройки
    @transform!(df_params, :Const = :Min_value .== :Max_value)
    @transform!(df_params, :Ignore = :Const .& (:Init_value .≠ :Min_value))
    
    # Часть параметров можно сразу исключать из настройки
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

    # Принудительно зануляются веса замеров для игнорируемых параметров (Ignore == true)
    crits = @with df_rates begin (
        Jp = :Wbhp_prod,
        Jinj = :Wbhp_inj,
    ) end
    for (key, values) ∈ pairs(crits)
        set_null_weights!(reshape(values, N, :), df_params, key)
    end

    # Добавляем вместо Кпрод геом. факторы скважин для настройки
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
        :Prod_Index = vec((prob.params.Gw .* prob.params.M)')
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

function save_params!(df::AbstractDataFrame, fset::FittingSet, path, opts)

    @transform!(df, :Calc_value = :Init_value)    
    df_view = @view df[.!(df.Const .| df.Ignore), :]
    copyto!(df_view.Calc_value, fset.cache.ybuf)

    # Преобразуем геом. факторы => Кпрод
    df_geom = @view df[df.Parameter .=== :Gw, :]
    df_mobt = @view df[df.Parameter .=== :M, :]
    df_pi = @view df[df.Parameter .=== :Jp, :]
    @. df_pi.Calc_value = df_geom.Calc_value * df_mobt.Calc_value
    
    replace!(df.Parameter, reverse.(PSYMS)...)
    select!(df, Not([:Istart, :Jstart, :Jstop, :Const, :Ignore])) 

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

    return df
end

function add_geom_factor!(df::AbstractDataFrame, name::Symbol, mobt)
       
    # Копируем Кпрод для расчета геом. фактора
    df_new = df[df.Parameter .=== name, :]

    # Рассчитываем геом. факторы скважин
    @eachrow! df_new begin
        @newcol :Minit::Vector{eltype(mobt)}
        @newcol :Mmin::Vector{eltype(mobt)}
        @newcol :Mmax::Vector{eltype(mobt)}

        # Динамика изменения подвижности фдюида
        rows = UnitRange(:Jstart::Int, :Jstop::Int)
        mobt_ = @view mobt[rows, :Istart::Int]

        # Делаем единичную подвижность, если требуется постоянный Кпрод
        :Const && (mobt_ .= one(eltype(mobt)))

        # Текущее и мин./макс. значения подвижности        
        :Minit = first(mobt_)
        :Mmin, :Mmax = extrema(mobt_)        

        # Пересчет Кпрод => геом. фактор        
        :Init_value /= :Minit
        :Min_value /= :Mmin
        :Max_value /= :Mmax
        
        # Обрабатываем случай, когда Gmin >= Gmax
        if :Max_value < :Min_value
            # Корректируем мин./макс. подвижность, чтобы Gmin == Gmax
            :Mmin = :Mmin * sqrt(:Min_value / :Max_value)
            :Mmax = :Mmax * sqrt(:Max_value / :Min_value)            
            clamp!(mobt_, :Mmin, :Mmax)
            # Рассчитываем новые Gmin, Gmax
            :Min_value = :Max_value = sqrt(:Min_value * :Max_value)            
        end

        # Помещаем значения в допустимые пределы
        :Init_value = clamp(:Init_value, :Min_value, :Max_value)
        :Minit = clamp(:Minit, :Mmin, :Mmax)
        :Const = :Min_value .== :Max_value
    end

    # Исключаем Кпрод из настройки
    df[df.Parameter .=== name, :Ignore] .= true
    # Удаляем старую информацию
    delete!(df, df.Parameter .∈ Ref([:Gw, :M]))

    # Добавляем геом. факторы скважин к остальным параметрам
    df_add = df_new[:, Not([:Minit, :Mmin, :Mmax])]
    df_add.Parameter .= :Gw
    append!(df, df_add)
    # Добавляем подвижности к остальным параметрам
    df_add = df_new[:, Not([:Init_value, :Min_value, :Max_value])]
    rename!(df_add, :Minit => :Init_value, :Mmin => :Min_value, :Mmax => :Max_value)
    df_add.Parameter .= :M
    df_add.Ignore .= true
    append!(df, df_add)

    return df
end