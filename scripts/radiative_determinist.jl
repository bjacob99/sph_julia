using LagrangianStochasticConvection
using ProgressBars

"""
Stochastic model coefficients
"""

function pressure(particle::Particle, kwparams::NamedTuple)
    return particle.means["rho_mean"] * particle.means["e_mean"]
end

function pressure_gradient(particle::Particle, kwparams::NamedTuple)
    dp_dx = particle.means["rho_gradx"] * particle.means["e_mean"] + particle.means["rho_mean"] * particle.means["e_gradx"]
    dp_dy = particle.means["rho_grady"] * particle.means["e_mean"] + particle.means["rho_mean"] * particle.means["e_grady"]
    return dp_dx, dp_dy
end

function drift_ux(particle::Particle, kwparams::NamedTuple)
    dp_dx, _ = pressure_gradient(particle, kwparams)
    G_0x = -dp_dx / (kwparams.gamma1 * particle.means["rho_mean"]) - kwparams.gravity
    return G_0x
end

function drift_uy(particle::Particle, kwparams::NamedTuple)
    _, dp_dy = pressure_gradient(particle, kwparams)
    G_0y = -dp_dy / (kwparams.gamma1 * particle.means["rho_mean"])
    return G_0y
end

function drift_energy(particle::Particle, kwparams::NamedTuple)
    k = particle.means["k_mean"]

    div_u = particle.means["ux_gradx"] + particle.means["uy_grady"]

    K_0 = -(kwparams.gamma1 - 1.0) * particle.means["e_mean"] * div_u

    a_e = K_0

    if (kwparams.hx <= particle.x) && (particle.x <= kwparams.Lx - kwparams.hx)
        a_e += kwparams.kappa * particle.means["e_laplacian"]
    end
    return a_e
end

function zero_coef(particle::Particle, kwparams::NamedTuple)
    return 0.0
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
    k = MeanVariable("k", p -> p.mass * 0.5 * ((p.ux - p.means["ux_mean"])^2 + (p.uy - p.means["uy_mean"])^2 + 
            (p.uz - p.means["uz_mean"])^2), simu_params)
    Rxx = MeanVariable("Rxx", p -> p.mass * (p.ux - p.means["ux_mean"])^2, simu_params)
    Ryy = MeanVariable("Ryy", p -> p.mass * (p.uy - p.means["uy_mean"])^2, simu_params)
    Rzz = MeanVariable("Rzz", p -> p.mass * (p.uz - p.means["uz_mean"])^2, simu_params)    
    Fex = MeanVariable("Fex", p -> p.mass * (p.ux - p.means["ux_mean"]) * (p.e - p.means["e_mean"]), simu_params)
    Fey = MeanVariable("Fey", p -> p.mass * (p.uy - p.means["uy_mean"]) * (p.e - p.means["e_mean"]), simu_params)
    ke = MeanVariable("ke", p -> p.mass * (p.e - p.means["e_mean"])^2, simu_params)
    Fkin = MeanVariable("fkin", p -> p.mass * 0.5 * (p.ux - p.means["ux_mean"]) * ((p.ux - p.means["ux_mean"])^2 + 
            (p.uy - p.means["uy_mean"])^2 + (p.uz - p.means["uz_mean"])^2), simu_params)
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
        standard_means!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, uz, e)
        corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, uz, e)
        divide_by_density!(rho, ux, uy, uz, e)
        write_means_to_particles!(particles, rho, ux, uy, uz, e)
        # Second round
        standard_means!(mean_computer, k, Fex, Fey, ke)
        corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, k, Fex, Fey, ke)
        divide_by_density!(rho, k, Fex, Fey, ke)
        write_means_to_particles!(particles, k, Fex, Fey, ke)
        ## Estimating means on fixed grid and saving
        if it % simu_params.freq_out == 0
            # Estimating: first round
            standard_means!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, e, is_on_grid=true)
            corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, e, is_on_grid=true)
            divide_by_density!(rho, ux, uy, e, is_on_grid=true)
            # Estimating: second round
            standard_means!(mean_computer, k, Rxx, Ryy, Rzz, Fex, ke, is_on_grid=true)
            corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, k, Rxx, Ryy, Rzz, Fex, ke, is_on_grid=true)
            divide_by_density!(rho, k, Rxx, Ryy, Rzz, Fex, ke, is_on_grid=true)
            # Saving
            save_mean_output(it, simu_params.save_directory, "mean", mean_computer, rho, ux, uy, e, k, Rxx, Ryy, Rzz, Fex, ke)
            save_mean_output(it, simu_params.save_directory, "gradx", mean_computer, e)
            save_particle_output(-1, simu_params.save_directory, particles)
        end
        ## Advancing flow
        advance_flow!(particles, drift_ux, drift_uy, zero_coef, drift_energy, zero_coef,
                      zero_coef, zero_coef, zero_coef, zero_coef, zero_coef, zero_coef, simu_params)
    end

end

main_complex(ARGS[1])