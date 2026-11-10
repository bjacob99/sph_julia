using LagrangianStochasticConvection
using ProgressBars

"""
Stochastic model coefficients
"""

function pressure(particle::Particle)
    return particle.means["rho_mean"] * particle.means["e_mean"]
end

function pressure_gradient(particle::Particle)
    dp_dx = particle.means["rho_gradx"] * particle.means["e_mean"] + particle.means["rho_mean"] * particle.means["e_gradx"]
    dp_dy = particle.means["rho_grady"] * particle.means["e_mean"] + particle.means["rho_mean"] * particle.means["e_grady"]
    return dp_dx, dp_dy
end

function shear_tensor_contraction(particle::Particle)
    shear = 4.0 * particle.means["ux_gradx"]^2
    shear += 2.0 * particle.means["ux_grady"]^2
    shear += 4.0 * particle.means["ux_grady"] * particle.means["uy_gradx"]
    shear += 2.0 * particle.means["uy_gradx"]^2
    shear += 4.0 * particle.means["uy_grady"]^2
    shear *= 0.25
    return shear
end

function Somega(particle::Particle, kwparams::NamedTuple)
    p_mean = pressure(particle)
    dp_dx, dp_dy = pressure_gradient(particle)
    P_over_epsilon = (dp_dx * particle.means["Fex_mean"] + dp_dy * particle.means["Fey_mean"])
    P_over_epsilon /= (p_mean * particle.means["omega_mean"] * particle.means["k_mean"] * kwparams.gamma1 * (kwparams.npoly - 1.0))
    P_over_epsilon += 0.5 * kwparams.Cmu * shear_tensor_contraction(particle) / particle.means["omega_mean"]^2
    return (kwparams.Ceps2 - 1.0) - (kwparams.Ceps1 - 1)*P_over_epsilon
end

function inverse_C(particle::Particle)
    k = particle.means["k_mean"]
    ke = particle.means["ke_mean"]
    invC = [1.0/ke   0.0        0.0        0.0  ;
            0.0      3.0/2.0/k  0.0        0.0  ;
            0.0      0.0        3.0/2.0/k  0.0  ;
            0.0      0.0        0.0        3.0/2.0/k ]
    return invC
end

function drift_ux(particle::Particle, kwparams::NamedTuple)
    p_mean = pressure(particle)
    dp_dx, _ = pressure_gradient(particle)
    k = particle.means["k_mean"]
    invC21 = 0.0
    invC22 = 3.0/2.0/k
    invC23 = 0.0
    invC24 = 0.0
    dux = particle.ux - particle.means["ux_mean"]
    duy = particle.uy - particle.means["uy_mean"]
    duz = particle.uz - particle.means["uz_mean"]
    de = particle.e - particle.means["e_mean"]
    domega = particle.omega - particle.means["omega_mean"]
    G_0x = -dp_dx / (kwparams.gamma1 * particle.means["rho_mean"]) - kwparams.gravity
    G_xx = - (0.5 + 0.75*kwparams.C0) * particle.means["omega_mean"]
    G_xx -= 0.5 * kwparams.C0 * domega * k * invC22
    G_xy = - 0.5 * kwparams.C0 * domega * k * invC23
    G_xz = - 0.5 * kwparams.C0 * domega * k * invC24
    G_ex = dp_dx / (p_mean * (kwparams.npoly - 1.0) * kwparams.gamma1)
    G_ex -= 0.5 * kwparams.C0 * domega * k * invC21 / (kwparams.gamma1*(kwparams.gamma1 - 1.0))
    return G_0x + G_xx*dux + G_xy*duy + G_xz*duz + G_ex*de
end

function drift_uy(particle::Particle, kwparams::NamedTuple)
    p_mean = pressure(particle)
    _, dp_dy = pressure_gradient(particle)
    k = particle.means["k_mean"]
    invC31 = 0.0
    invC32 = 0.0
    invC33 = 3.0/2.0/k
    invC34 = 0.0
    dux = particle.ux - particle.means["ux_mean"]
    duy = particle.uy - particle.means["uy_mean"]
    duz = particle.uz - particle.means["uz_mean"]
    de = particle.e - particle.means["e_mean"]
    domega = particle.omega - particle.means["omega_mean"]
    G_0y = -dp_dy / (kwparams.gamma1 * particle.means["rho_mean"])
    G_yx = - 0.5 * kwparams.C0 * domega * k * invC32
    G_yy = - (0.5 + 0.75*kwparams.C0) * particle.means["omega_mean"]
    G_yy -= 0.5 * kwparams.C0 * domega * k * invC33
    G_yz = - 0.5 * kwparams.C0 * domega * k * invC34
    G_ey = dp_dy / (p_mean * (kwparams.npoly - 1.0) * kwparams.gamma1)
    G_ey -= 0.5 * kwparams.C0 * domega * k * invC31 / (kwparams.gamma1*(kwparams.gamma1 - 1.0))
    return G_0y + G_yx*dux + G_yy*duy + G_yz*duz + G_ey*de
end

function drift_uz(particle::Particle, kwparams::NamedTuple)
    k = particle.means["k_mean"]
    invC41 = 0.0
    invC42 = 0.0
    invC43 = 0.0
    invC44 = 3.0/2.0/k
    dux = particle.ux - particle.means["ux_mean"]
    duy = particle.uy - particle.means["uy_mean"]
    duz = particle.uz - particle.means["uz_mean"]
    de = particle.e - particle.means["e_mean"]
    domega = particle.omega - particle.means["omega_mean"]
    G_zx = - 0.5 * kwparams.C0 * domega * k * invC42
    G_zy = - 0.5 * kwparams.C0 * domega * k * invC43
    G_zz = - (0.5 + 0.75*kwparams.C0) * particle.means["omega_mean"]
    G_zz -= 0.5 * kwparams.C0 * domega * k * invC44
    G_ez = - 0.5 * kwparams.C0 * domega * k * invC41 / (kwparams.gamma1*(kwparams.gamma1 - 1.0))
    return G_zx*dux + G_zy*duy + G_zz*duz + G_ez*de
end

function drift_energy(particle::Particle, kwparams::NamedTuple)
    k = particle.means["k_mean"]
    invC11 = 1.0/particle.means["ke_mean"]
    invC12 = 0.0
    invC13 = 0.0
    invC14 = 0.0
    div_u = particle.means["ux_gradx"] + particle.means["uy_grady"]
    div_Fe = particle.means["Fex_gradx"] + particle.means["Fey_grady"]
    Fe_dot_grad_e = particle.means["Fex_mean"]*particle.means["e_gradx"] + particle.means["Fey_mean"]*particle.means["e_grady"]
    dux = particle.ux - particle.means["ux_mean"]
    duy = particle.uy - particle.means["uy_mean"]
    duz = particle.uz - particle.means["uz_mean"]
    de = particle.e - particle.means["e_mean"]
    domega = particle.omega - particle.means["omega_mean"]
    K_0 = (kwparams.gamma1 - 1.0)/(kwparams.npoly - 1.0) * (div_Fe - Fe_dot_grad_e / particle.means["e_mean"])
    K_0 -= (kwparams.gamma1 - 1.0) * particle.means["e_mean"] * div_u
    K_0 += kwparams.gamma1 * (kwparams.gamma1 - 1.0) * particle.means["omega_mean"] * k
    K_x = - 0.5 * kwparams.C1 * domega * particle.means["ke_mean"] * invC12 * (kwparams.gamma1*(kwparams.gamma1 - 1.0))
    K_y = - 0.5 * kwparams.C1 * domega * particle.means["ke_mean"] * invC13 * (kwparams.gamma1*(kwparams.gamma1 - 1.0))
    K_z = - 0.5 * kwparams.C1 * domega * particle.means["ke_mean"] * invC14 * (kwparams.gamma1*(kwparams.gamma1 - 1.0))
    K_e = - kwparams.gamma1 * (kwparams.gamma1 - 1.0) * particle.means["omega_mean"] * k / (particle.means["e_mean"] * (kwparams.npoly - 1.0))
    K_e -= 0.5 * kwparams.C1 * particle.means["omega_mean"]
    K_e -= (kwparams.gamma1 - 1.0) * div_u
    K_e -= 0.5 * kwparams.C1 * domega * particle.means["ke_mean"] * invC11
    a_e = K_0 + K_x*dux + K_y*duy + K_z*duz + K_e*de
    # if (kwparams.hx <= particle.x) && (particle.x <= kwparams.Lx - kwparams.hx)
    #     a_e += kwparams.kappa * particle.means["e_laplacian"]
    # end
    if (particle.x <= kwparams.Lx - kwparams.hx)
        a_e += kwparams.kappa * particle.means["e_laplacian"]
    end
    return a_e
end

function drift_omega(particle::Particle, kwparams::NamedTuple)
    a_omega = -particle.means["omega_cond_mean"] * (particle.omega - particle.means["omega_mean"])
    a_omega -= particle.means["omega_mean"] * particle.omega * Somega(particle, kwparams)
    return a_omega
end

function diff_u(particle::Particle, kwparams::NamedTuple)
    k = max(particle.means["k_mean"], 0.0)
    return sqrt(kwparams.C0 * particle.omega * k)
end

function diff_energy(particle::Particle, kwparams::NamedTuple)
    ke = max(particle.means["ke_mean"], 0.0)
    return sqrt(kwparams.C1 * particle.omega * ke)
end

function diff_omega(particle::Particle, kwparams::NamedTuple)
    omega_mean = max(particle.means["omega_mean"], 0.0)
    omega_cond_mean = max(particle.means["omega_cond_mean"], 0.0)
    return sqrt(2.0 * kwparams.C2 * omega_mean * omega_cond_mean * particle.omega)
end

function extra_diff_omega(particle::Particle, kwparams::NamedTuple)
    omega_mean = max(particle.means["omega_mean"], 0.0)
    omega_cond_mean = max(particle.means["omega_cond_mean"], 0.0)
    return 2.0 * kwparams.C2 * omega_mean * omega_cond_mean
end

"""
Running simulation
"""

function main_complex(config_path::String)

    ## Initialisation of simulation

    particles, mean_computer, volume_computer, simu_params = init_simulation(config_path)

    rho_from_count = MeanVariable("rho_count", p -> p.mass, simu_params)
    rho = MeanVariable("rho", p -> p.mass, simu_params)
    ux = MeanVariable("ux", p -> p.mass * p.ux, simu_params)
    uy = MeanVariable("uy", p -> p.mass * p.uy, simu_params)
    uz = MeanVariable("uz", p -> p.mass * p.uz, simu_params)
    e = MeanVariable("e", p -> p.mass * p.e, simu_params)
    omega = MeanVariable("omega", p -> p.mass * p.omega, simu_params)
    k = MeanVariable("k", p -> p.mass * 0.5 * ((p.ux - p.means["ux_mean"])^2 + (p.uy - p.means["uy_mean"])^2 + 
            (p.uz - p.means["uz_mean"])^2), simu_params)
    Fex = MeanVariable("Fex", p -> p.mass * (p.ux - p.means["ux_mean"]) * (p.e - p.means["e_mean"]), simu_params)
    Fey = MeanVariable("Fey", p -> p.mass * (p.uy - p.means["uy_mean"]) * (p.e - p.means["e_mean"]), simu_params)
    ke = MeanVariable("ke", p -> p.mass * (p.e - p.means["e_mean"])^2, simu_params)
    Fkin = MeanVariable("fkin", p -> p.mass * 0.5 * (p.ux - p.means["ux_mean"]) * ((p.ux - p.means["ux_mean"])^2 + 
            (p.uy - p.means["uy_mean"])^2 + (p.uz - p.means["uz_mean"])^2), simu_params)
    omega_cond = MeanVariable("omega_cond", p -> ifelse(p.omega > p.means["omega_mean"], p.mass * p.omega, 0.0), simu_params)
    rho_cond = MeanVariable("rho_cond", p -> ifelse(p.omega > p.means["omega_mean"], p.mass, 0.0), simu_params)
    W, X, Y, XX, XY, YY = common_mean_variables(simu_params)
    
    display_simu_params(simu_params)
    bar = ProgressBar(simu_params.it_start:simu_params.ntime+simu_params.it_start-1)
    npart_to_save = min(10000, length(particles))

    ## Running simulation

    for it in bar
        ## Estimating particle volume
        particle_volumes!(particles, volume_computer, rho_from_count, simu_params)
        ## Estimating means on particle positions
        # First round
        set_particle_positions!(mean_computer, particles)
        standard_means!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, uz, e, omega)
        corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, uz, e, omega)
        divide_by_density!(rho, ux, uy, uz, e, omega)
        write_means_to_particles!(particles, rho, ux, uy, uz, e, omega)
        # Second round
        standard_means!(mean_computer, k, Fex, Fey, ke, omega_cond, rho_cond)
        corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, k, Fex, Fey, ke, omega_cond, rho_cond)
        divide_by_density!(rho, k, Fex, Fey, ke)
        divide_by_density!(rho_cond, omega_cond)
        write_means_to_particles!(particles, k, Fex, Fey, ke, omega_cond, rho_cond)
        ## Estimating means on fixed grid and saving
        if it % simu_params.freq_out == 0
            # Estimating: first round
            standard_means!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, e, omega, is_on_grid=true)
            corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, e, omega, is_on_grid=true)
            divide_by_density!(rho, ux, uy, e, omega, is_on_grid=true)
            # Estimating: second round
            standard_means!(mean_computer, k, Fex, ke, Fkin, is_on_grid=true)
            corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, k, Fex, ke, Fkin, is_on_grid=true)
            divide_by_density!(rho, k, Fex, ke, Fkin, is_on_grid=true)
            # Saving
            save_mean_output(it, simu_params.save_directory, "mean", mean_computer, rho, ux, uy, e, omega, k, Fex, ke, Fkin)
            save_mean_output(it, simu_params.save_directory, "gradx", mean_computer, e)
            save_particle_output(it, simu_params.save_directory, particles[1:npart_to_save])
            save_particle_output(-1, simu_params.save_directory, particles)
        end
        ## Advancing flow
        advance_flow!(particles, drift_ux, drift_uy, drift_uz, drift_energy, drift_omega,
                      diff_u, diff_u, diff_u, diff_energy, diff_omega, extra_diff_omega, simu_params)
    end
    save_particle_output(-1, simu_params.save_directory, particles)
end

main_complex(ARGS[1])