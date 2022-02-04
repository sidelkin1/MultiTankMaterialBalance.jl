opencsv(path) = open(read, path, enc"WINDOWS-1251")

function read_rates(path, dateformat)
    
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
        :Wres => Float64,
        :Wbhp_prod => Float64,
        :Wbhp_inj => Float64,
        :Pres_min => Float64,
        :Pres_max => Float64        
    )
    df = CSV.File(opencsv(path); dateformat, types, pool=false) |> DataFrame

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

function read_params(path, dateformat, psyms)

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
    df = CSV.File(opencsv(path); dateformat, types, pool=false) |> DataFrame
    
    # Заполняем пропущенные значения
    crit = ismissing.(df.Neighb)
    df[crit, :Neighb] .= df[crit, :Tank]
    # TODO Для преобразования 'Union{Missing, String}' в 'String'
    df.Neighb = convert.(String, df.Neighb)

    # Преобразуем 'String' в 'Symbol'
    df.Parameter = Symbol.(df.Parameter)
    replace!(df.Parameter, psyms...)

    # Коэффициент регуляризации на параметры 
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

    return df_params
end