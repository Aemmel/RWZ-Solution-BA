#=
    Solve the RZW equation:
        Psi_tt - Psi_xx + V(r_s) Psi = S
    where:
        - Psi RWZ Master function (in accordance with Nagar)
        - _i partial derivative after i
        - x = tortoise coordinate
        - r_s r in schwarzschild coordinates
        - V RW/Z Potential
        - S Source term = 0 for now
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

# finite Differencing according to Calabrese & Gundlach arxiv:0509119 eq. 15
function finiteDiff(vals::Array{Float64}, step_size)::Array{Float64}
    deriv = copy(vals)
    step_size_2 = 12. * step_size^2

    for i=3:length(deriv)-2
        deriv[i] = (-vals[i+2] - vals[i-2] + 16.0*(vals[i+1] + vals[i-1]) - 30.0 * vals[i]) / step_size_2
    end

    #println(deriv[10:20])

    return deriv
end

# according to Calabrese & Gundlach arxiv:0509119 eq. 27-30
# they start at -2, we at 1. So to convert add 3 to the index position of C&G
function fillGhostCells!(vals::Wave, step_size)
    # reference for readibility
    psi = vals.psi
    theta = vals.theta

    #### psi
    i = 2
    psi[i] = -4. *step_size*theta[i+1] - 10. / 3. * psi[i+1] + 6. * psi[i+2] - 2. * psi[i+3] + 1. / 3. * psi[i+4]
    i = length(psi) - 1
    psi[i] = -4. *step_size*theta[i-1] - 10. / 3. * psi[i-1] + 6. * psi[i-2] - 2. * psi[i-3] + 1. / 3. * psi[i-4]

    i = 1
    psi[i] = -20.0*step_size*theta[i+1] - 80. / 3. * psi[i+1] + 40. * psi[i+2] - 15. * psi[i+3] + 8. / 3. * psi[i+4]
    i = length(psi)
    psi[i] = -20.0*step_size*theta[i-1] - 80. / 3. * psi[i-1] + 40. * psi[i-2] - 15. * psi[i-3] + 8. / 3. * psi[i-4]

    #### theta
    i = 2
    theta[i] = 4. * theta[i+1] - 6. * theta[i+2] + 4. * theta[i+3] - theta[i+4]
    i = length(theta) - 1
    theta[i] = 4. * theta[i-1] - 6. * theta[i-2] + 4. * theta[i-3] - theta[i-4]

    i = 1
    theta[i] = 4. * theta[i+1] - 6. * theta[i+2] + 4. * theta[i+3] - theta[i+4]
    i = length(theta)
    theta[i] = 4. * theta[i-1] - 6. * theta[i-2] + 4. * theta[i-3] - theta[i-4]
end

# RWZ Potential
# r in schwarzschild coordinate, not tortoise (either scalar or Array)
# l is mode and M is Mass (geometric units c=G=1)
# return either scalar at point r or Array at points r
@enum Potential even odd
function calcPotential(parity::Potential, r, l, M)
    # potentials has form:
    # (1-2M/r)*... = fac_1 * fac_2
    fac_1 = 1.0 .- 2.0M ./ r # the same for even and odd potential

    # useful terms
    lambda = l*(l + 1. )
    lambda_2 = lambda - 2.
    lambda_2_sq = lambda_2^2

    if parity == odd
        fac_2 = lambda ./ r.^2 - 6.0M ./ r.^3
    else 
        fac_2 = (lambda*lambda_2_sq .* r.^3 + 6.0lambda_2_sq*M .* r.^2 + 36.0lambda_2*M^2 .* r .+ 72.0*M^3) ./ (r.^3 .*(lambda_2 .* r .+ 6.0M).^2)
    end

    return fac_1 .* fac_2
end

# Standard RK4 for du/dt = F(u) (no explicit time dependence)
# u_(n+1) = u_n + dt/6 * (K1 + 2*K2 + 2*K3 + K4)
# K1 = F(u_n)
# K2 = F(u_n + dt/2 * K1)
# K3 = F(u_n + dt/2 * K2)
# K4 = F(u_n + dt * K3)
function timeStep(vals, func, params, dt)
    dt_half = dt / 2.

    K1 = func(vals, params)
    K2 = func(vals + dt_half * K1, params)
    K3 = func(vals + dt_half * K2, params)
    K4 = func(vals + dt * K3, params)

    return vals + dt/6. * (K1 + 2K2 + 2K3 + K4)
end

function RWZRightHandSide(w::Wave, step_size)::Wave
    return Wave(w.theta, finiteDiff(w.psi, step_size))
end

function main(;dt_arg=0.0001, x_points=2000)
    plotly()

    # solution: sin(2pi(x + t))
    init_psi(x) = sin(2*pi*x)
    init_theta(x) = 2*pi*cos(2*pi*x)

    points = x_points
    x_max = 2
    x_data = range(0, stop=x_max, length=points)
    dx = x_data[2] - x_data[1] # should be equally spaced
    curr_step = Wave(init_psi.(x_data), init_theta.(x_data))

    CFL_aplha = 1

    dt = CFL_aplha * dx
    t = 0
    t_max = 10.7

    # CFL condition
    if dt > dx
        println("CFL condition not met!")
        return -1
    end

    cnt = 0

    plot(x_data, curr_step.psi, legend=false)

    while t < t_max
        # fillGhostCells!(curr_step, dx)

        # Dirichlet for fourth order 
        curr_step.psi[1] = sin(2pi*(t))
        curr_step.psi[2] = sin(2pi*(dx+t))
        curr_step.psi[length(curr_step)] = sin(2pi*(x_max + t))
        curr_step.psi[length(curr_step) - 1] = sin(2pi*(x_max - dx + t))
        curr_step.theta[1] = 2*pi*cos(2pi*(t))
        curr_step.theta[2] = 2*pi*cos(2pi*(dx+t))
        curr_step.theta[length(curr_step)] = 2*pi*cos(2pi*(x_max + t))
        curr_step.theta[length(curr_step) - 1] = 2*pi*cos(2pi*(x_max - dx + t))

        curr_step = timeStep(curr_step, RWZRightHandSide, dx, dt)

        t += dt
        cnt += 1
    end

    plot(x_data, curr_step.psi, label="MOL")
    # @printf("dt: %.6f,  Points: %d,  min: %.6f\n", dt, points, minimum(curr_step.psi))
end

main()