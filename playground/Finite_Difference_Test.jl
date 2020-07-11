using Plots

function finiteDiffFirst(values::Array{Float64}, step_size::Float64)::Array{Float64}
    deriv = copy(values)

    for i = 2:(length(deriv)-2)
        deriv[i] = (values[i+1] - values[i-1]) / (2*step_size)
    end

    return deriv
end

function finiteDiffSecond(values::Array{Float64}, step_size::Float64)::Array{Float64}
    deriv = copy(values)

    for i = 2:length(deriv)-2
        deriv[i] = (values[i+1] - 2values[i] + values[i-1]) / step_size^2
    end

    return deriv
end

#=function plot(x_val::Array{Float64}, y_val::Array{Float64})
    plot(x_val, y_val)
end
=#
function main()
    step_size = 0.01
    domain = 10 # go from 0 to domain
    len::Int64 = domain ÷ step_size # length of the array

    x_values = [i*step_size for i=0:len]
    y_values = sin.(x_values)

    println(len)

    plotly()
    plot(x_values, y_values)

    deriv = finiteDiffSecond(y_values, step_size)
    plot!(x_values[2:len-1], deriv[2:len-1])

    plot!(x_values, -sin.(x_values))

    #println(x_values)
    #println(y_values)
end

main()