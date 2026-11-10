function kinetic_energy(particle::Particle)
    return 0.5 * (particle.means["Rxx_mean"] + particle.means["Ryy_mean"] + particle.means["Rzz_mean"])
end

function pressure(particle::Particle)
    return particle.means["rho_mean"] * particle.means["e_mean"]
end

function pressure_gradient(particle::Particle)
    dp_dx = particle.means["rho_gradx"] * particle.means["e_mean"] + particle.means["rho_mean"] * particle.means["e_gradx"]
    dp_dy = particle.means["rho_grady"] * particle.means["e_mean"] + particle.means["rho_mean"] * particle.means["e_grady"]
    return dp_dx, dp_dy
end

function drift_ux(particle::Particle, kwparams::NamedTuple)
    p_mean = pressure(particle)
    dp_dx, _ = pressure_gradient(particle)
    dux = particle.ux - particle.means["ux_mean"]
    de = particle.e - particle.means["e_mean"]
    G_0x = -dp_dx / (kwparams.gamma1 * particle.means["rho_mean"]) - kwparams.gravity
    G_xx = - (0.5 + 0.75*kwparams.C0) * particle.omega
    G_ex = dp_dx / (p_mean * (kwparams.npoly - 1.0) * kwparams.gamma1)
    return G_0x + G_xx*dux + G_ex*de
end

function drift_uy(particle::Particle, kwparams::NamedTuple)
    p_mean = pressure(particle)
    _, dp_dy = pressure_gradient(particle)
    duy = particle.uy - particle.means["uy_mean"]
    de = particle.e - particle.means["e_mean"]
    G_0y = -dp_dy / (kwparams.gamma1 * particle.means["rho_mean"])
    G_yy = - (0.5 + 0.75*kwparams.C0) * particle.omega
    G_ey = dp_dy / (p_mean * (kwparams.npoly - 1.0) * kwparams.gamma1)
    return G_0y + G_yy*duy + G_ey*de
end

function drift_uz(particle::Particle, kwparams::NamedTuple)
    duz = particle.uz - particle.means["uz_mean"]
    G_zz = - (0.5 + 0.75*kwparams.C0) * particle.omega
    return G_zz*duz
end

function drift_energy(particle::Particle, kwparams::NamedTuple)
    k = kinetic_energy(particle)
    div_u = particle.means["ux_gradx"] + particle.means["uy_grady"]
    div_Fe = particle.means["Fex_gradx"] + particle.means["Fey_grady"]
    Fe_dot_grad_e = particle.means["Fex_mean"]*particle.means["e_gradx"] + particle.means["Fey_mean"]*particle.means["e_grady"]
    de = particle.e - particle.means["e_mean"]
    K_0 = (kwparams.gamma1 - 1.0)/(kwparams.npoly - 1.0) * (div_Fe - Fe_dot_grad_e / particle.means["e_mean"])
    K_0 -= (kwparams.gamma1 - 1.0) * particle.means["e_mean"] * div_u
    K_0 += kwparams.gamma1 * (kwparams.gamma1 - 1.0) * particle.omega * k
    K_e = - kwparams.gamma1 * (kwparams.gamma1 - 1.0) * particle.omega * k / (particle.means["e_mean"] * (kwparams.npoly - 1.0))
    K_e -= 0.5 * kwparams.C1 * particle.omega
    K_e -= (kwparams.gamma1 - 1.0) * div_u
    a_e = K_0 + K_e*de         
    if (kwparams.hx <= particle.x) && (particle.x <= kwparams.Lx - kwparams.hx)
        a_e += kwparams.kappa * particle.means["e_laplacian"]
    end
    return a_e
end

function drift_omega(particle::Particle, kwparams::NamedTuple)
    return 0.0
end

function diff_ux(particle::Particle, kwparams::NamedTuple)
    k = max(kinetic_energy(particle), 0.0)
    return sqrt(kwparams.C0 * particle.omega * k)
end

function diff_uy(particle::Particle, kwparams::NamedTuple)
    k = max(kinetic_energy(particle), 0.0)
    return sqrt(kwparams.C0 * particle.omega * k)
end

function diff_uz(particle::Particle, kwparams::NamedTuple)
    k = max(kinetic_energy(particle), 0.0)
    return sqrt(kwparams.C0 * particle.omega * k)
end

function diff_energy(particle::Particle, kwparams::NamedTuple)
    ke = max(particle.means["ke_mean"], 0.0)
    return sqrt(kwparams.C1 * particle.omega * ke)
end

function diff_omega(particle::Particle, kwparams::NamedTuple)
    return 0.0
end

function extra_diff_omega(particle::Particle, kwparams::NamedTuple)
    return 0.0
end