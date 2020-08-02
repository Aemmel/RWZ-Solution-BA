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
using Printf

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
    for i = 2:length(deriv)-1
        deriv[i] = (vals[i+1] - 2vals[i] + vals[i-1]) / step_size_2
    end

    return deriv
end

function finiteDiffPaper(vals::Array{Float64}, step_size)::Array{Float64}
    deriv = copy(vals)
    step_size_2 = 1. / 12. * step_size^2

    for i=3:length(deriv)-2
        deriv[i] = (-vals[i+2] - vals[i-2] + 16.0*(vals[i+1] + vals[i-1]) - 30.0 * vals[i]) / step_size_2
    end

    return deriv
end

function finiteDiffSecondFourthOrder(vals::Array{Float64}, step_size)::Array{Float64}
    deriv = copy(vals)
    step_size_2 = step_size^2

    fac_1 = - 1. / 12.
    fac_2 = 4. / 3.
    fac_3 = - 5. / 2.

    # deal with the ghost cells later
    for i = 3:length(deriv)-2
        deriv[i] = (fac_1*vals[i-2] + fac_2*vals[i-1] + fac_3*vals[i] + fac_2*vals[i+1] + fac_1*vals[i+2]) / step_size_2
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
end

# define our wave equation
function WaveEq(w::Wave, step_size)::Wave
    return Wave(w.theta, finiteDiffSecond(w.psi, step_size))
end

function WaveEqAnalytical(w::Wave, step_size)::Wave
    return Wave(w.theta, -w.psi) # test functin psi(x,t) = sin(x)
end

function main(;dt_arg=0.0001, x_points=401)
    plotly()

    # solution: sin(2pi(x + t))
    # init_psi(x) = sin(2*pi*x)
    # init_theta(x) = 2*pi*cos(2*pi*x)
    init_psi(x) = exp(-x^2)
    init_theta(x) = -2x*exp(-x^2)

    points = x_points
    x_max = 5
    x_data = range(-x_max, stop=x_max, length=points)
    dx = x_data[2] - x_data[1] # should be equally spaced
    curr_step = Wave(init_psi.(x_data), init_theta.(x_data))

    CFL_aplha = 1

    dt = CFL_aplha * dx
    t = 0
    t_max = 1

    # CFL condition
    if dt > dx
        println("CFL condition not met!")
        return -1
    end

    cnt = 0

    plot(x_data, curr_step.psi, legend=false)

    while t < t_max
        fillGhostCells!(curr_step)

        # Dirichlet for fourth order 
        # curr_step.psi[1] = sin(2pi*(t))
        # curr_step.psi[2] = sin(2pi*(dx+t))
        # curr_step.psi[length(curr_step)] = sin(2pi*(x_max + t))
        # curr_step.psi[length(curr_step) - 1] = sin(2pi*(x_max - dx + t))
        # curr_step.theta[1] = 2*pi*cos(2pi*(t))
        # curr_step.theta[2] = 2*pi*cos(2pi*(dx+t))
        # curr_step.theta[length(curr_step)] = 2*pi*cos(2pi*(x_max + t))
        # curr_step.theta[length(curr_step) - 1] = 2*pi*cos(2pi*(x_max - dx + t))

        curr_step = RK4(curr_step, WaveEq, dx, dt)

        t += dt
        cnt += 1
    end

    plot!(x_data, curr_step.psi, label="MOL")
    # @printf("dt: %.6f,  Points: %d,  min: %.6f\n", dt, points, minimum(curr_step.psi))
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

    x_max = 2
    h = 0.01
    x_data = range(0, stop=x_max, step=h)

    f       = f_2
    f_dd    = f_2_dd

    y = f.(x_data)

    for i=1:1
        y = finiteDiffSecondFourthOrder(y, h)

        # periodic boundary conditions
        y[1] = y[length(y) - 1]
        y[length(y)] = y[2]
    end

    plot(x_data, y, label="Numerical")
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
    display(plot(t_max, log.(y_worst), label="y worst"))
    # display(plot(t_max, runtime, label="runtime"))

    # reveals:
    # x wost linear
    # y worst exponential
    # runtime linear

    # x=0 to x-axis with 5000 points
    # t=0 to 5 with dt=0.001
    # x_max = 11:0.05:16
    # y_worst_xmax = [main(i) for i in x_max]

    # plot(x_max, y_worst_xmax, label="y worst over xmax")
end

main()

# for p=100:100:2000
#     for dt in 10. .^range(-3, stop=-5, step=-0.1)
#         main(dt, p)
#     end

#     println("-----------------------------------")
# end

# results
# dt: 0.001000,  Points: 100,  min: -1.065732
# dt: 0.000794,  Points: 100,  min: -1.065853
# dt: 0.000631,  Points: 100,  min: -1.065910
# dt: 0.000501,  Points: 100,  min: -1.066133
# dt: 0.000398,  Points: 100,  min: -1.065949
# dt: 0.000316,  Points: 100,  min: -1.066079
# dt: 0.000251,  Points: 100,  min: -1.066090
# dt: 0.000200,  Points: 100,  min: -1.065962
# dt: 0.000158,  Points: 100,  min: -1.065993
# dt: 0.000126,  Points: 100,  min: -1.066010
# dt: 0.000100,  Points: 100,  min: -1.066017
# dt: 0.000079,  Points: 100,  min: -1.065992
# dt: 0.000063,  Points: 100,  min: -1.065957
# dt: 0.000050,  Points: 100,  min: -1.065966
# dt: 0.000040,  Points: 100,  min: -1.065958
# dt: 0.000032,  Points: 100,  min: -1.065959
# dt: 0.000025,  Points: 100,  min: -1.065959
# dt: 0.000020,  Points: 100,  min: -1.065958
# dt: 0.000016,  Points: 100,  min: -1.065957
# dt: 0.000013,  Points: 100,  min: -1.065956
# dt: 0.000010,  Points: 100,  min: -1.065961
# -----------------------------------
# dt: 0.001000,  Points: 200,  min: -1.031224
# dt: 0.000794,  Points: 200,  min: -1.031460
# dt: 0.000631,  Points: 200,  min: -1.031580
# dt: 0.000501,  Points: 200,  min: -1.031857
# dt: 0.000398,  Points: 200,  min: -1.031675
# dt: 0.000316,  Points: 200,  min: -1.031830
# dt: 0.000251,  Points: 200,  min: -1.031849
# dt: 0.000200,  Points: 200,  min: -1.031713
# dt: 0.000158,  Points: 200,  min: -1.031750
# dt: 0.000126,  Points: 200,  min: -1.031770
# dt: 0.000100,  Points: 200,  min: -1.031780
# dt: 0.000079,  Points: 200,  min: -1.031752
# dt: 0.000063,  Points: 200,  min: -1.031714
# dt: 0.000050,  Points: 200,  min: -1.031725
# dt: 0.000040,  Points: 200,  min: -1.031716
# dt: 0.000032,  Points: 200,  min: -1.031717
# dt: 0.000025,  Points: 200,  min: -1.031717
# dt: 0.000020,  Points: 200,  min: -1.031716
# dt: 0.000016,  Points: 200,  min: -1.031715
# dt: 0.000013,  Points: 200,  min: -1.031714
# dt: 0.000010,  Points: 200,  min: -1.031720
# -----------------------------------
# dt: 0.001000,  Points: 300,  min: -1.020869
# dt: 0.000794,  Points: 300,  min: -1.021141
# dt: 0.000631,  Points: 300,  min: -1.021302
# dt: 0.000501,  Points: 300,  min: -1.021285
# dt: 0.000398,  Points: 300,  min: -1.021469
# dt: 0.000316,  Points: 300,  min: -1.021441
# dt: 0.000251,  Points: 300,  min: -1.021463
# dt: 0.000200,  Points: 300,  min: -1.021554
# dt: 0.000158,  Points: 300,  min: -1.021550
# dt: 0.000126,  Points: 300,  min: -1.021548
# dt: 0.000100,  Points: 300,  min: -1.021548
# dt: 0.000079,  Points: 300,  min: -1.021565
# dt: 0.000063,  Points: 300,  min: -1.021585
# dt: 0.000050,  Points: 300,  min: -1.021582
# dt: 0.000040,  Points: 300,  min: -1.021587
# dt: 0.000032,  Points: 300,  min: -1.021587
# dt: 0.000025,  Points: 300,  min: -1.021587
# dt: 0.000020,  Points: 300,  min: -1.021587
# dt: 0.000016,  Points: 300,  min: -1.021588
# dt: 0.000013,  Points: 300,  min: -1.021589
# dt: 0.000010,  Points: 300,  min: -1.021586
# -----------------------------------
# dt: 0.001000,  Points: 400,  min: -1.015147
# dt: 0.000794,  Points: 400,  min: -1.015487
# dt: 0.000631,  Points: 400,  min: -1.015699
# dt: 0.000501,  Points: 400,  min: -1.015817
# dt: 0.000398,  Points: 400,  min: -1.015950
# dt: 0.000316,  Points: 400,  min: -1.015981
# dt: 0.000251,  Points: 400,  min: -1.016013
# dt: 0.000200,  Points: 400,  min: -1.016061
# dt: 0.000158,  Points: 400,  min: -1.016069
# dt: 0.000126,  Points: 400,  min: -1.016074
# dt: 0.000100,  Points: 400,  min: -1.016078
# dt: 0.000079,  Points: 400,  min: -1.016086
# dt: 0.000063,  Points: 400,  min: -1.016094
# dt: 0.000050,  Points: 400,  min: -1.016094
# dt: 0.000040,  Points: 400,  min: -1.016096
# dt: 0.000032,  Points: 400,  min: -1.016097
# dt: 0.000025,  Points: 400,  min: -1.016097
# dt: 0.000020,  Points: 400,  min: -1.016097
# dt: 0.000016,  Points: 400,  min: -1.016098
# dt: 0.000013,  Points: 400,  min: -1.016098
# dt: 0.000010,  Points: 400,  min: -1.016097
# -----------------------------------
# dt: 0.001000,  Points: 500,  min: -1.011552
# dt: 0.000794,  Points: 500,  min: -1.012138
# dt: 0.000631,  Points: 500,  min: -1.012480
# dt: 0.000501,  Points: 500,  min: -1.012583
# dt: 0.000398,  Points: 500,  min: -1.012703
# dt: 0.000316,  Points: 500,  min: -1.012729
# dt: 0.000251,  Points: 500,  min: -1.012764
# dt: 0.000200,  Points: 500,  min: -1.012819
# dt: 0.000158,  Points: 500,  min: -1.012829
# dt: 0.000126,  Points: 500,  min: -1.012836
# dt: 0.000100,  Points: 500,  min: -1.012841
# dt: 0.000079,  Points: 500,  min: -1.012851
# dt: 0.000063,  Points: 500,  min: -1.012861
# dt: 0.000050,  Points: 500,  min: -1.012861
# dt: 0.000040,  Points: 500,  min: -1.012864
# dt: 0.000032,  Points: 500,  min: -1.012864
# dt: 0.000025,  Points: 500,  min: -1.012865
# dt: 0.000020,  Points: 500,  min: -1.012865
# dt: 0.000016,  Points: 500,  min: -1.012866
# dt: 0.000013,  Points: 500,  min: -1.012866
# dt: 0.000010,  Points: 500,  min: -1.012865
# -----------------------------------
# CFL condition not met!
# dt: 0.000794,  Points: 600,  min: -1.009643
# dt: 0.000631,  Points: 600,  min: -1.010070
# dt: 0.000501,  Points: 600,  min: -1.010354
# dt: 0.000398,  Points: 600,  min: -1.010485
# dt: 0.000316,  Points: 600,  min: -1.010537
# dt: 0.000251,  Points: 600,  min: -1.010578
# dt: 0.000200,  Points: 600,  min: -1.010627
# dt: 0.000158,  Points: 600,  min: -1.010642
# dt: 0.000126,  Points: 600,  min: -1.010652
# dt: 0.000100,  Points: 600,  min: -1.010659
# dt: 0.000079,  Points: 600,  min: -1.010668
# dt: 0.000063,  Points: 600,  min: -1.010675
# dt: 0.000050,  Points: 600,  min: -1.010676
# dt: 0.000040,  Points: 600,  min: -1.010679
# dt: 0.000032,  Points: 600,  min: -1.010679
# dt: 0.000025,  Points: 600,  min: -1.010680
# dt: 0.000020,  Points: 600,  min: -1.010680
# dt: 0.000016,  Points: 600,  min: -1.010681
# dt: 0.000013,  Points: 600,  min: -1.010681
# dt: 0.000010,  Points: 600,  min: -1.010680
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# dt: 0.000631,  Points: 700,  min: -1.008363
# dt: 0.000501,  Points: 700,  min: -1.008620
# dt: 0.000398,  Points: 700,  min: -1.008769
# dt: 0.000316,  Points: 700,  min: -1.008881
# dt: 0.000251,  Points: 700,  min: -1.008934
# dt: 0.000200,  Points: 700,  min: -1.008974
# dt: 0.000158,  Points: 700,  min: -1.008996
# dt: 0.000126,  Points: 700,  min: -1.009009
# dt: 0.000100,  Points: 700,  min: -1.009018
# dt: 0.000079,  Points: 700,  min: -1.009025
# dt: 0.000063,  Points: 700,  min: -1.009029
# dt: 0.000050,  Points: 700,  min: -1.009031
# dt: 0.000040,  Points: 700,  min: -1.009033
# dt: 0.000032,  Points: 700,  min: -1.009034
# dt: 0.000025,  Points: 700,  min: -1.009034
# dt: 0.000020,  Points: 700,  min: -1.009035
# dt: 0.000016,  Points: 700,  min: -1.009035
# dt: 0.000013,  Points: 700,  min: -1.009035
# dt: 0.000010,  Points: 700,  min: -1.009035
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000501,  Points: 800,  min: -1.007462
# dt: 0.000398,  Points: 800,  min: -1.007625
# dt: 0.000316,  Points: 800,  min: -1.007768
# dt: 0.000251,  Points: 800,  min: -1.007836
# dt: 0.000200,  Points: 800,  min: -1.007899
# dt: 0.000158,  Points: 800,  min: -1.007920
# dt: 0.000126,  Points: 800,  min: -1.007933
# dt: 0.000100,  Points: 800,  min: -1.007942
# dt: 0.000079,  Points: 800,  min: -1.007953
# dt: 0.000063,  Points: 800,  min: -1.007963
# dt: 0.000050,  Points: 800,  min: -1.007964
# dt: 0.000040,  Points: 800,  min: -1.007967
# dt: 0.000032,  Points: 800,  min: -1.007968
# dt: 0.000025,  Points: 800,  min: -1.007968
# dt: 0.000020,  Points: 800,  min: -1.007969
# dt: 0.000016,  Points: 800,  min: -1.007969
# dt: 0.000013,  Points: 800,  min: -1.007970
# dt: 0.000010,  Points: 800,  min: -1.007969
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000501,  Points: 900,  min: -1.006538
# dt: 0.000398,  Points: 900,  min: -1.006745
# dt: 0.000316,  Points: 900,  min: -1.006914
# dt: 0.000251,  Points: 900,  min: -1.007019
# dt: 0.000200,  Points: 900,  min: -1.007079
# dt: 0.000158,  Points: 900,  min: -1.007111
# dt: 0.000126,  Points: 900,  min: -1.007129
# dt: 0.000100,  Points: 900,  min: -1.007141
# dt: 0.000079,  Points: 900,  min: -1.007149
# dt: 0.000063,  Points: 900,  min: -1.007155
# dt: 0.000050,  Points: 900,  min: -1.007158
# dt: 0.000040,  Points: 900,  min: -1.007160
# dt: 0.000032,  Points: 900,  min: -1.007161
# dt: 0.000025,  Points: 900,  min: -1.007161
# dt: 0.000020,  Points: 900,  min: -1.007162
# dt: 0.000016,  Points: 900,  min: -1.007162
# dt: 0.000013,  Points: 900,  min: -1.007162
# dt: 0.000010,  Points: 900,  min: -1.007163
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000398,  Points: 1000,  min: -1.005990
# dt: 0.000316,  Points: 1000,  min: -1.006154
# dt: 0.000251,  Points: 1000,  min: -1.006278
# dt: 0.000200,  Points: 1000,  min: -1.006342
# dt: 0.000158,  Points: 1000,  min: -1.006387
# dt: 0.000126,  Points: 1000,  min: -1.006412
# dt: 0.000100,  Points: 1000,  min: -1.006426
# dt: 0.000079,  Points: 1000,  min: -1.006433
# dt: 0.000063,  Points: 1000,  min: -1.006435
# dt: 0.000050,  Points: 1000,  min: -1.006439
# dt: 0.000040,  Points: 1000,  min: -1.006440
# dt: 0.000032,  Points: 1000,  min: -1.006441
# dt: 0.000025,  Points: 1000,  min: -1.006442
# dt: 0.000020,  Points: 1000,  min: -1.006443
# dt: 0.000016,  Points: 1000,  min: -1.006443
# dt: 0.000013,  Points: 1000,  min: -1.006443
# dt: 0.000010,  Points: 1000,  min: -1.006444
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000398,  Points: 1100,  min: -1.005329
# dt: 0.000316,  Points: 1100,  min: -1.005522
# dt: 0.000251,  Points: 1100,  min: -1.005622
# dt: 0.000200,  Points: 1100,  min: -1.005687
# dt: 0.000158,  Points: 1100,  min: -1.005742
# dt: 0.000126,  Points: 1100,  min: -1.005773
# dt: 0.000100,  Points: 1100,  min: -1.005790
# dt: 0.000079,  Points: 1100,  min: -1.005796
# dt: 0.000063,  Points: 1100,  min: -1.005796
# dt: 0.000050,  Points: 1100,  min: -1.005801
# dt: 0.000040,  Points: 1100,  min: -1.005802
# dt: 0.000032,  Points: 1100,  min: -1.005803
# dt: 0.000025,  Points: 1100,  min: -1.005804
# dt: 0.000020,  Points: 1100,  min: -1.005804
# dt: 0.000016,  Points: 1100,  min: -1.005805
# dt: 0.000013,  Points: 1100,  min: -1.005805
# dt: 0.000010,  Points: 1100,  min: -1.005806
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000398,  Points: 1200,  min: -1.004830
# dt: 0.000316,  Points: 1200,  min: -1.004974
# dt: 0.000251,  Points: 1200,  min: -1.005111
# dt: 0.000200,  Points: 1200,  min: -1.005174
# dt: 0.000158,  Points: 1200,  min: -1.005220
# dt: 0.000126,  Points: 1200,  min: -1.005244
# dt: 0.000100,  Points: 1200,  min: -1.005258
# dt: 0.000079,  Points: 1200,  min: -1.005265
# dt: 0.000063,  Points: 1200,  min: -1.005268
# dt: 0.000050,  Points: 1200,  min: -1.005273
# dt: 0.000040,  Points: 1200,  min: -1.005274
# dt: 0.000032,  Points: 1200,  min: -1.005276
# dt: 0.000025,  Points: 1200,  min: -1.005277
# dt: 0.000020,  Points: 1200,  min: -1.005277
# dt: 0.000016,  Points: 1200,  min: -1.005278
# dt: 0.000013,  Points: 1200,  min: -1.005278
# dt: 0.000010,  Points: 1200,  min: -1.005278
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000316,  Points: 1300,  min: -1.004575
# dt: 0.000251,  Points: 1300,  min: -1.004685
# dt: 0.000200,  Points: 1300,  min: -1.004772
# dt: 0.000158,  Points: 1300,  min: -1.004788
# dt: 0.000126,  Points: 1300,  min: -1.004821
# dt: 0.000100,  Points: 1300,  min: -1.004843
# dt: 0.000079,  Points: 1300,  min: -1.004852
# dt: 0.000063,  Points: 1300,  min: -1.004854
# dt: 0.000050,  Points: 1300,  min: -1.004860
# dt: 0.000040,  Points: 1300,  min: -1.004861
# dt: 0.000032,  Points: 1300,  min: -1.004863
# dt: 0.000025,  Points: 1300,  min: -1.004864
# dt: 0.000020,  Points: 1300,  min: -1.004864
# dt: 0.000016,  Points: 1300,  min: -1.004865
# dt: 0.000013,  Points: 1300,  min: -1.004865
# dt: 0.000010,  Points: 1300,  min: -1.004866
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000316,  Points: 1400,  min: -1.004205
# dt: 0.000251,  Points: 1400,  min: -1.004330
# dt: 0.000200,  Points: 1400,  min: -1.004415
# dt: 0.000158,  Points: 1400,  min: -1.004448
# dt: 0.000126,  Points: 1400,  min: -1.004480
# dt: 0.000100,  Points: 1400,  min: -1.004502
# dt: 0.000079,  Points: 1400,  min: -1.004513
# dt: 0.000063,  Points: 1400,  min: -1.004516
# dt: 0.000050,  Points: 1400,  min: -1.004522
# dt: 0.000040,  Points: 1400,  min: -1.004523
# dt: 0.000032,  Points: 1400,  min: -1.004525
# dt: 0.000025,  Points: 1400,  min: -1.004526
# dt: 0.000020,  Points: 1400,  min: -1.004527
# dt: 0.000016,  Points: 1400,  min: -1.004527
# dt: 0.000013,  Points: 1400,  min: -1.004527
# dt: 0.000010,  Points: 1400,  min: -1.004528
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000316,  Points: 1500,  min: -1.003862
# dt: 0.000251,  Points: 1500,  min: -1.004015
# dt: 0.000200,  Points: 1500,  min: -1.004108
# dt: 0.000158,  Points: 1500,  min: -1.004143
# dt: 0.000126,  Points: 1500,  min: -1.004172
# dt: 0.000100,  Points: 1500,  min: -1.004197
# dt: 0.000079,  Points: 1500,  min: -1.004207
# dt: 0.000063,  Points: 1500,  min: -1.004210
# dt: 0.000050,  Points: 1500,  min: -1.004217
# dt: 0.000040,  Points: 1500,  min: -1.004218
# dt: 0.000032,  Points: 1500,  min: -1.004221
# dt: 0.000025,  Points: 1500,  min: -1.004222
# dt: 0.000020,  Points: 1500,  min: -1.004222
# dt: 0.000016,  Points: 1500,  min: -1.004223
# dt: 0.000013,  Points: 1500,  min: -1.004223
# dt: 0.000010,  Points: 1500,  min: -1.004224
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000251,  Points: 1600,  min: -1.003722
# dt: 0.000200,  Points: 1600,  min: -1.003822
# dt: 0.000158,  Points: 1600,  min: -1.003873
# dt: 0.000126,  Points: 1600,  min: -1.003909
# dt: 0.000100,  Points: 1600,  min: -1.003941
# dt: 0.000079,  Points: 1600,  min: -1.003952
# dt: 0.000063,  Points: 1600,  min: -1.003953
# dt: 0.000050,  Points: 1600,  min: -1.003961
# dt: 0.000040,  Points: 1600,  min: -1.003962
# dt: 0.000032,  Points: 1600,  min: -1.003964
# dt: 0.000025,  Points: 1600,  min: -1.003966
# dt: 0.000020,  Points: 1600,  min: -1.003966
# dt: 0.000016,  Points: 1600,  min: -1.003966
# dt: 0.000013,  Points: 1600,  min: -1.003966
# dt: 0.000010,  Points: 1600,  min: -1.003968
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000251,  Points: 1700,  min: -1.003460
# dt: 0.000200,  Points: 1700,  min: -1.003561
# dt: 0.000158,  Points: 1700,  min: -1.003621
# dt: 0.000126,  Points: 1700,  min: -1.003663
# dt: 0.000100,  Points: 1700,  min: -1.003692
# dt: 0.000079,  Points: 1700,  min: -1.003704
# dt: 0.000063,  Points: 1700,  min: -1.003718
# dt: 0.000050,  Points: 1700,  min: -1.003721
# dt: 0.000040,  Points: 1700,  min: -1.003726
# dt: 0.000032,  Points: 1700,  min: -1.003728
# dt: 0.000025,  Points: 1700,  min: -1.003729
# dt: 0.000020,  Points: 1700,  min: -1.003730
# dt: 0.000016,  Points: 1700,  min: -1.003731
# dt: 0.000013,  Points: 1700,  min: -1.003732
# dt: 0.000010,  Points: 1700,  min: -1.003731
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000251,  Points: 1800,  min: -1.003247
# dt: 0.000200,  Points: 1800,  min: -1.003360
# dt: 0.000158,  Points: 1800,  min: -1.003411
# dt: 0.000126,  Points: 1800,  min: -1.003457
# dt: 0.000100,  Points: 1800,  min: -1.003485
# dt: 0.000079,  Points: 1800,  min: -1.003505
# dt: 0.000063,  Points: 1800,  min: -1.003520
# dt: 0.000050,  Points: 1800,  min: -1.003524
# dt: 0.000040,  Points: 1800,  min: -1.003529
# dt: 0.000032,  Points: 1800,  min: -1.003531
# dt: 0.000025,  Points: 1800,  min: -1.003533
# dt: 0.000020,  Points: 1800,  min: -1.003534
# dt: 0.000016,  Points: 1800,  min: -1.003534
# dt: 0.000013,  Points: 1800,  min: -1.003535
# dt: 0.000010,  Points: 1800,  min: -1.003534
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000251,  Points: 1900,  min: -1.003044
# dt: 0.000200,  Points: 1900,  min: -1.003164
# dt: 0.000158,  Points: 1900,  min: -1.003232
# dt: 0.000126,  Points: 1900,  min: -1.003282
# dt: 0.000100,  Points: 1900,  min: -1.003296
# dt: 0.000079,  Points: 1900,  min: -1.003309
# dt: 0.000063,  Points: 1900,  min: -1.003322
# dt: 0.000050,  Points: 1900,  min: -1.003329
# dt: 0.000040,  Points: 1900,  min: -1.003333
# dt: 0.000032,  Points: 1900,  min: -1.003335
# dt: 0.000025,  Points: 1900,  min: -1.003337
# dt: 0.000020,  Points: 1900,  min: -1.003338
# dt: 0.000016,  Points: 1900,  min: -1.003339
# dt: 0.000013,  Points: 1900,  min: -1.003339
# dt: 0.000010,  Points: 1900,  min: -1.003339
# -----------------------------------
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# CFL condition not met!
# dt: 0.000200,  Points: 2000,  min: -1.002977
# dt: 0.000158,  Points: 2000,  min: -1.003057
# dt: 0.000126,  Points: 2000,  min: -1.003118
# dt: 0.000100,  Points: 2000,  min: -1.003142
# dt: 0.000079,  Points: 2000,  min: -1.003155
# dt: 0.000063,  Points: 2000,  min: -1.003167
# dt: 0.000050,  Points: 2000,  min: -1.003171
# dt: 0.000040,  Points: 2000,  min: -1.003176
# dt: 0.000032,  Points: 2000,  min: -1.003178
# dt: 0.000025,  Points: 2000,  min: -1.003179
# dt: 0.000020,  Points: 2000,  min: -1.003180
# dt: 0.000016,  Points: 2000,  min: -1.003181
# dt: 0.000013,  Points: 2000,  min: -1.003182
# dt: 0.000010,  Points: 2000,  min: -1.003181
# -----------------------------------