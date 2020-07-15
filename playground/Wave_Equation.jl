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
function finiteDiffSecond(vals::Array{Float64}, step_size)::Array{Float64}
    deriv = copy(vals)
    step_size_2 = step_size^2

    # deal with the ghost cells later
    for i = 2:length(deriv)-2
        deriv[i] = (vals[i+1] - 2vals[i] + vals[i-1]) / step_size_2
    end

    return deriv
end

# Standard RK4 for du/dt = F(u) (no explicit time dependence)
# u_(n+1) = u_n + dt/6 * (K1 + 2*K2 + 2*K3 + K4)
# K1 = F(u_n)
# K2 = F(u_n + dt/2 * K1)
# K3 = F(u_n + dt/2 * K2)
# K4 = F(u_n + dt * K3)
function RK4(vals, func, params, dt)
    dt_half = dt / 2.

    K1 = func(vals, params)
    K2 = func(vals + dt_half * K1, params)
    K3 = func(vals + dt_half * K2, params)
    K4 = func(vals + dt * K3, params)

    return vals + dt/6. * (K1 + 2K2 + 2K3 + K4)
end

# periodic boundary conditions
function fillGhostCells!(vals::Wave)
    vals.psi[1] = vals.psi[length(vals) - 1]
    vals.psi[length(vals)] = vals.psi[2]

    vals.theta[1] = vals.theta[length(vals) - 1]
    vals.theta[length(vals)] = vals.theta[2]

    # vals.psi[1] = vals.psi[length(vals) - 1]
    # vals.psi[length(vals)] = vals.psi[2]
end

# define our wave equation
function WaveEq(w::Wave, step_size)::Wave
    return Wave(w.theta, finiteDiffSecond(w.psi, step_size))
    #return Wave(w.theta, -w.psi) # test functin psi(x,t) = sin(x)
end

function WaveEqAnalytical(w::Wave, step_size)::Wave
    return Wave(w.theta, -w.psi) # test functin psi(x,t) = sin(x)
end

function main(x_max_arg)
    plotly()

    # solution: sin(2pi(x + t))
    # init_psi(x) = sin(2*pi*x)
    # init_theta(x) = 2*pi*cos(2*pi*x)

    init_psi(x) = sin(x)
    init_theta(x) = cos(x)

    points = 5000
    x_max = x_max_arg
    x_data = range(0, stop=x_max, length=points)
    dx = x_data[2] - x_data[1] # should be equally spaced
    curr_step = Wave(init_psi.(x_data), init_theta.(x_data))
    curr_step_ana = Wave(init_psi.(x_data), init_theta.(x_data))

    dt = 0.001
    t = 0
    t_max = 5

    # CFL condition
    if dt > 0.5*dx
        println("CFL condition not met!")
        return -1
    end

    cnt = 0

    while t < t_max
        #fillGhostCells!(curr_step)
        # fill ghost cells manually
        curr_step.psi[1] = sin(t)
        curr_step.psi[length(curr_step)] = sin(x_max + t)
        curr_step_ana.psi[1] = sin(t)
        curr_step_ana.psi[length(curr_step)] = sin(x_max + t)
        # curr_step.theta[1] = cos(t)
        # curr_step.theta[length(curr_step)] = cos(x_max + t)

        # if cnt % 1000000 == 0
        #     plot!(x_data, curr_step.psi, legend=false)
        # end

        curr_step = RK4(curr_step, WaveEq, dx, dt)
        curr_step_ana = RK4(curr_step_ana, WaveEqAnalytical, dx, dt)   

        t += dt
        cnt += 1
    end

    #plot(x_data, curr_step.psi, label="MOL")
    #plot!(x_data, curr_step_ana.psi, label="Ana")
    #plot(x_data, finiteDiffSecond(sin.(x_data .+ 1), dx))
    #plot!(x_data, -sin.(x_data .+ 1))
    #plot(x_data, sin.(2*pi*(x_data .+ 1.2)))

    println("done with " * string(x_max_arg))
    return maximum(abs.(curr_step.psi))
end

# test the second derivative finite difference
# works perfectly for the given function f_i
function testFiniteDiff()
    plotly()

    f_1(x) = sin(x)
    f_1_dd(x) = -sin(x)

    f_2(x) = exp(-x^2)
    f_2_dd(x) = 2*exp(-x^2)*(2x^2-1)

    f_3(x) = cos(2pi*x)
    f_3_dd(x) = -4pi^2*cos(2pi*x)

    x_max = 13
    h = 0.05
    x_data = range(0, stop=x_max, step=h)

    f       = f_3
    f_dd    = f_3_dd

    plot(x_data, finiteDiffSecond(f.(x_data), h), label="Numerical")
    plot!(x_data, f_dd.(x_data), label="Analytical")
end

# test RK4 with harmonic oscillator
# works perfectly for a wide range of paramters
function testRK4()
    plotly()

    coord = [1.0, 0.0]

    function harmonic_oscillator(c, params)
        return [c[2], -c[1]]
    end

    dt = 0.1
    t = 0
    t_max = 90
    cnt_max = 1000

    cnt = 1

    steps::Array{Float64, 1} = []

    while cnt <= cnt_max
        coord = RK4(coord, harmonic_oscillator, 0, dt)
        push!(steps, coord[1])

        t += dt
        cnt += 1
    end

    plot(range(0, step=dt, length=cnt_max), steps)
end

function plotStuff()
    # x=0 to 7pi with 5000 points
    # t=0 to x-axis with dt=0.001
    t_max = 1:10
    
    # y-axis: where it diverges from correct solution
    x_where_bad = [21.4, 20.4, 19.4, 18.4, 17.4, 16.4, 15.4, 14.4, 13.4, 12.4]

    # y-axis: |y| value of worst value (1 before hardcoded ghost cell)
    y_worst = [1.2, 3.6, 9.9, 27, 73, 200, 545, 1483, 4000, 10000]

    # y-axis: runtime
    runtime = [0.8, 1.6, 2.3, 3.0, 3.7, 4.4, 5.2, 5.9, 6.4, 7.4]

    # display(plot(t_max, x_where_bad, label="x where bad"))
    # display(plot(t_max, y_worst, label="y worst", yaxis=:log))
    # display(plot(t_max, runtime, label="runtime"))

    # reveals:
    # x wost linear
    # y worst exponential
    # runtime linear

    # x=0 to x-axis with 5000 points
    # t=0 to 5 with dt=0.001
    x_max = 11:0.05:16
    y_worst_xmax = [main(i) for i in x_max]

    plot(x_max, y_worst_xmax, label="y worst over xmax")
end

#main()

plotStuff()

# testFiniteDiff()