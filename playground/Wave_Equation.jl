#= solve the wave equation of the form
 P_tt = c^2 P_xx
 using the method of lines

 First we note, that with T := P_t our equation becomes
 T_t = c^2 P_xx

 we can then solve the RHS of that equation with finite differencing and then solve for that timestep
 the resulting equaiton with RK4
 
 Solve with periodic boundary conditions

 We set c = 1
=#

using Plots

# finite Difference method for second derivative
# [du/dx]_i = (u_(i+1) - 2u_i + u_(i-1)) / h^2 + O(h^2)
# values: original Array
# return: Array of same size 
function finiteDiffSecond(values, step_size)
    deriv = copy(values)

    # deal with the ghost cells later
    for i = 2:length(deriv)-2
        deriv[i] = (values[i+1] - 2values[i] + values[i-1]) / step_size^2
    end

    return deriv
end

# Standard RK4 for du/dt = F(u) (no explicit time dependence)
# u_(n+1) = u_n + dt/6 * (K1 + 2*K2 + 2*K3 + K4)
# K1 = F(u_n)
# K2 = F(u_n + dt/2 * K1)
# K3 = F(u_n + dt/2 * K2)
# K4 = F(u_n + dt * K3)
function RK4(values, func, params, dt)
    dt_half = dt / 2.

    K1 = func.(values, params)
    K2 = func.(values + dt_half * K1, params)
    K3 = func.(values + dt_half * K2, params)
    K4 = func.(values + dt * K3, params)

    return values + dt/6. * (K1 + 2K2 + 2K3 + K4)
end

# periodic boundary conditions
function fillGhostCells!(values)
    values[1] = values[length(values) - 1]
    values[length(values)] = values[2]
end

# test RK4 with du/dt = u
function test_RK4()
    curr_step = [1.] # start with 1
    values = copy(curr_step)
    func(u) = return u

    dt = 0.005
    t = 0
    while t < 3
        curr_step = RK4(curr_step, func, dt)

        append!(values, curr_step)

        t += dt
    end

    plotly()

    plot(1:length(values), values)
    plot!(1:length(values), exp.(dt*(1:length(values))))
end

function main()
    init_foo(x) = sin(12*pi*x)

    points = 200
    x_data = range(0, stop=1, length=points)
    dx = x_data[2] - x_data[1] # should be equally spaced
    curr_step = init_foo.(x_data)

    dt = 0.01
    t = 0
    t_max = 1

    while t < t_max
        fillGhostCells!(curr_step)
        curr_step = RK4(curr_step, finiteDiffSecond, dx, dt)

        t += dt
    end

    plotly()

    plot(x_data, curr_step)
end

main()