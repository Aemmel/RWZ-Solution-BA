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

# is this the correct way of doing things? .. I don't know..
push!(LOAD_PATH, pwd());

using Plots
using Printf

import LambertW.lambertw

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

# tortoise to schwarzschild, needed for the potential
# mass is needed
function tortToSchwarz(tort_vals, M)
    return 2. * M .* (1 .+ lambertw.(exp.(tort_vals ./ (2. * M) .- 1), 0))
end

# finite Differencing according to Calabrese & Gundlach arxiv:0509119 eq. 15
function finiteDiff!(deriv::Array{Float64}, vals::Array{Float64}, step_size)
    step_size_2 = 12. * step_size^2

    for i=3:length(deriv)-2
        deriv[i] = (-vals[i+2] - vals[i-2] + 16.0*(vals[i+1] + vals[i-1]) - 30.0 * vals[i]) / step_size_2
    end
end

# according to Calabrese & Gundlach arxiv:0509119 eq. 27-30
# they start at -2, we at 1. So to convert, add 3 to the index position of C&G
# fill vals' ghost cells with rule for Psi (in C&G: phi)
function fillGhostPsi!(vals::Array{Float64}, orig::Wave, step_size)
    # reference for readibility
    psi = orig.psi
    theta = orig.theta

    #### psi
    i = 2
    vals[i] = -4. *step_size*theta[i+1] - 10. / 3. * psi[i+1] + 6. * psi[i+2] - 2. * psi[i+3] + 1. / 3. * psi[i+4]
    i = length(psi) - 1
    vals[i] = -4. *step_size*theta[i-1] - 10. / 3. * psi[i-1] + 6. * psi[i-2] - 2. * psi[i-3] + 1. / 3. * psi[i-4]

    i = 1
    vals[i] = -20.0*step_size*theta[i+1] - 80. / 3. * psi[i+1] + 40. * psi[i+2] - 15. * psi[i+3] + 8. / 3. * psi[i+4]
    # vals[i] = 5.0*theta[i+1] - 10. * psi[i+2] + 10. * psi[i+3] - 5. * psi[i+4] + psi[i+5]
    i = length(psi)
    vals[i] = -20.0*step_size*theta[i-1] - 80. / 3. * psi[i-1] + 40. * psi[i-2] - 15. * psi[i-3] + 8. / 3. * psi[i-4]
    # vals[i] = 5.0*theta[i-1] - 10. * psi[i-2] + 10. * psi[i-3] - 5. * psi[i-4] + psi[i-5]
end

# fill vals' ghost cells with rule for Theta (in C&G: Pi)
function fillGhostTheta!(vals::Array{Float64}, orig::Wave)
    # reference for readibility
    psi = orig.psi
    theta = orig.theta

    #### theta
    i = 2
    vals[i] = 4. * theta[i+1] - 6. * theta[i+2] + 4. * theta[i+3] - theta[i+4]
    i = length(theta) - 1
    vals[i] = 4. * theta[i-1] - 6. * theta[i-2] + 4. * theta[i-3] - theta[i-4]
    
    i = 1
    vals[i] = 4. * theta[i+1] - 6. * theta[i+2] + 4. * theta[i+3] - theta[i+4]
    i = length(theta)
    vals[i] = 4. * theta[i-1] - 6. * theta[i-2] + 4. * theta[i-3] - theta[i-4]
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

# RHS of the RWZ equation: Psi_xx - V(x) Psi
function RWZRightHandSide(w::Wave, (pot_vals, step_size)::Tuple{Array{Float64}, Float64})::Wave
    psi = w.psi
    theta = w.theta

    new_psi = copy(theta)
    new_theta = copy(psi)

    fillGhostPsi!(new_theta, w, step_size)
    #fillGhostTheta!(new_psi, w) # does weird stuff when activated..

    finiteDiff!(new_theta, psi, step_size)

    for i = 1:length(new_theta)
        new_theta[i] -= pot_vals[i] * psi[i]
    end

    return Wave(new_psi, new_theta)
end

# initial Value for gaussian packet centered around mu with variance sigma
# return (psi (gauss), theta (d/dx gauss))
function initialDataGauss(x_data, mu, sigma)::Wave
    sigma_2 = sigma*sigma
    fac = 1 / sigma / sqrt(2pi)
    dfac = - fac / sigma
    
    gauss(x) = fac * exp(-0.5*sigma_2*(x - mu)^2)
    dgauss(x) = dfac * (x - mu) * exp(-0.5*sigma_2*(x - mu)^2)

    return Wave(gauss.(x_data), dgauss.(x_data))
end

function main(; x_points=4000,      # how many x points
                x_min=-400,         # minimum r_tortoise
                x_max=+400,         # maximum r_tortoise
                t_max=150,          # max time (starts at 0)
                CFL_aplha=1,        # CFL for timestep
                mass=1,             # mass of the black hole
                ell=2,              # mode of the perturbation
                parity=odd,         # axial (odd) or polar (even) perturbation
                gauss_mu=150,       # mu (offset) for gaussian package
                gauss_sigma=1,      # sigma for gaussian package
)
    plotly()

    # init data
    x_data = range(x_min, stop=x_max, length=x_points)
    dx = x_data[2] - x_data[1]
    curr_step = initialDataGauss(x_data, gauss_mu, gauss_sigma)
    pot_vals = calcPotential(odd, tortToSchwarz(x_data, 1), ell, mass)

    dt = CFL_aplha * dx
    println("dt="*string(dt))
    println("dx="*string(dx))
    t = 0

    # CFL condition
    if dt > dx
        println("CFL condition not met!")
        return -1
    end

    # detector output
    detector_pos = 250
    detector_pos_index = searchsortedfirst(x_data, detector_pos)
    time_data = []      # x data for detector plot
    detector_data = []  # y data for detector plot

    cnt = 1
    plot_every = 10000000

    plot(x_data, curr_step.psi, label="starting")
    #plot!(x_data, calcPotential(odd, tortToSchwarz(x_data, 1), 5, 1))
    #yaxis!((-2, 2))

    while t < t_max
        curr_step = timeStep(curr_step, RWZRightHandSide, (pot_vals, dx), dt)
        
        if cnt % plot_every == 0
            plot!(x_data, curr_step.psi, label=string(t))
        end

        push!(time_data, t)
        push!(detector_data, curr_step.psi[detector_pos_index])

        t += dt
        cnt += 1
    end

    display(plot!(x_data, curr_step.psi, label="final"))

    display(plot(time_data, abs.(detector_data), label="|Psi|"))
    display(plot(time_data, log.(abs.(detector_data)), label="log|Psi|"))
    plot(time_data, detector_data, label="Psi")
    
    # @printf("dt: %.6f,  Points: %d,  min: %.6f\n", dt, points, minimum(curr_step.psi))
end

main(x_points=8000, x_max=600, ell=2, parity=odd, gauss_mu=150, gauss_sigma=0.8, t_max=480, CFL_aplha=0.5)