# finite Differencing according to Calabrese & Gundlach arxiv:0509119
function finiteDiff(#=params=#)
    
end

# Standard RK4 for du/dt = F(u) (no explicit time dependence)
# u_(n+1) = u_n + dt/6 * (K1 + 2*K2 + 2*K3 + K4)
# K1 = F(u_n)
# K2 = F(u_n + dt/2 * K1)
# K3 = F(u_n + dt/2 * K2)
# K4 = F(u_n + dt * K3)
function time_step(vals, func, params, dt)
    dt_half = dt / 2.

    K1 = func(vals, params)
    K2 = func(vals + dt_half * K1, params)
    K3 = func(vals + dt_half * K2, params)
    K4 = func(vals + dt * K3, params)

    return vals + dt/6. * (K1 + 2K2 + 2K3 + K4)
end

function main(#=params=#)

end