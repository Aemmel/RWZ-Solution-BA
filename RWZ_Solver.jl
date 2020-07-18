using Plots

# finite Differencing according to Calabrese & Gundlach arxiv:0509119
function finiteDiff(#=params=#)
    
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

function main(#=params=#)
    r = 2:0.01:10
    ell = 5
    mass = 1

    plotly()

    plot(r, calcPotential(even, r, ell, mass), label="even")
    plot!(r, calcPotential(odd, r, ell, mass), label="odd")
    #plot(r, r)
end

main()