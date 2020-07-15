#= solve the wave equation of the form
 psi_tt = c^2 psi_xx
 using the method of lines

 First we note, that with theta := psi_t our equation becomes
 theta_t = c^2 psi_xx

 So with u = (psi, theta), we have the differential equation
 du/dt = d/dt(psi, theta) = (u.theta, c^2 u.psi_xx)

 we can then solve the RHS of that equation with finite differencing and then solve for that timestep
 the resulting equaiton with RK4
 
 Solve with periodic boundary conditions

 We set c = 1
=#

using Plots

# define Wave type with some useful Base extensions
mutable struct Wave
    psi::Array{Float64} # see comment above for explanation
    theta::Array{Float64}
end

Base.:length(w::Wave)::Int = length(w.psi) # length psi == length theta should always be the case

function Base.:*(x::T, w::Wave)::Wave where T<:Real
    return Wave(x*w.psi, x*w.theta)
end

function Base.:*(w::Wave, x::T)::Wave where T<:Real
    return x*w
end

Base.:+(w1::Wave, w2::Wave)::Wave = Wave(w1.psi + w2.psi, w1.theta + w2.theta)

# finite Difference method for second derivative
# [du/dx]_i = (u_(i+1) - 2u_i + u_(i-1)) / h^2 + O(h^2)
# values: original Array
# return: Array of same size 
function finiteDiffSecond(values::Array{Float64}, step_size)::Array{Float64}
    deriv = copy(values)
    step_size_2 = step_size^2

    # deal with the ghost cells later
    for i = 2:length(deriv)-2
        deriv[i] = (values[i+1] - 2values[i] + values[i-1]) / step_size_2
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

    K1 = func(values, params)
    K2 = func(values + dt_half * K1, params)
    K3 = func(values + dt_half * K2, params)
    K4 = func(values + dt * K3, params)

    return values + dt/6. * (K1 + 2K2 + 2K3 + K4)
end

# periodic boundary conditions
function fillGhostCells!(values::Wave)
    # values.psi[1] = values.psi[length(values) - 1]
    # values.psi[length(values)] = values.psi[2]

    # values.theta[1] = values.theta[length(values) - 1]
    # values.theta[length(values)] = values.theta[2]

    values.psi[1] = values.psi[length(values) - 1]
    values.psi[length(values)] = values.psi[2]
end

function main()
    plotly()

    # solution: sin(2pi(x- t))
    init_psi(x) = sin(2*pi*x)
    init_theta(x) = 2*pi*cos(2*pi*x)

    points = 200
    x_data = range(0, stop=1, length=points)
    dx = x_data[2] - x_data[1] # should be equally spaced
    curr_step = Wave(init_psi.(x_data), init_theta.(x_data))

    dt = 0.0001
    t = 0
    t_max = 1.5

    # CFL condition
    if dt > 0.5*dx
        println("CFL condition not met!")
        return -1
    end

    # define our wave equation
    function WaveEq(w::Wave, step_size)::Wave
        return Wave(w.theta, finiteDiffSecond(w.psi, step_size))
        #return Wave(w.theta, -4*pi*pi*w.psi) # test functin psi(x,t) = sin(2pi x)
    end

    while t < t_max
        fillGhostCells!(curr_step)
        curr_step = RK4(curr_step, WaveEq, dx, dt)

        t += dt
    end

    plot(x_data, curr_step.psi)
end

main()