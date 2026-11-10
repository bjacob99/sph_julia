using LagrangianStochasticConvection
using ProgressBars

"""
Stochastic model coefficients
"""

function pressure(particle::Particle, kwparams::NamedTuple)
    return particle.means["rho_mean"]^kwparams.gamma1
end

function pressure_gradient(particle::Particle, kwparams::NamedTuple)
    dp_dx = kwparams.gamma1 * particle.means["rho_gradx"] * pressure(particle, kwparams) / particle.means["rho_mean"]
    dp_dy = kwparams.gamma1 * particle.means["rho_grady"] * pressure(particle, kwparams) / particle.means["rho_mean"]
    return dp_dx, dp_dy
end

function drift_ux(particle::Particle, kwparams::NamedTuple)
    dp_dx, _ = pressure_gradient(particle, kwparams)
    dux = particle.ux - particle.means["ux_mean"]
    G_0x = -dp_dx / (kwparams.gamma1 * particle.means["rho_mean"]) - kwparams.gravity
    G_xx = - (0.5 + 0.75*kwparams.C0) * particle.omega
    return G_0x + G_xx*dux
end

function drift_uy(particle::Particle, kwparams::NamedTuple)
    _, dp_dy = pressure_gradient(particle, kwparams)
    duy = particle.uy - particle.means["uy_mean"]
    G_0y = -dp_dy / (kwparams.gamma1 * particle.means["rho_mean"])
    G_yy = - (0.5 + 0.75*kwparams.C0) * particle.omega
    return G_0y + G_yy*duy
end

function drift_uz(particle::Particle, kwparams::NamedTuple)
    duz = particle.uz - particle.means["uz_mean"]
    G_zz = - (0.5 + 0.75*kwparams.C0) * particle.omega
    return G_zz*duz
end

function diff_u(particle::Particle, kwparams::NamedTuple)
    k = max(0.0, particle.means["k_mean"])
    return sqrt(kwparams.C0 * particle.omega * k)
end

function zero_coef(particle::Particle, kwparams::NamedTuple)
    return 0.0
end

"""
Running simulation
"""

function main_SLM(config_path::String)

    ## Initialisation of simulation

    particles, mean_computer, volume_computer, simu_params = init_simulation(config_path)

    rho_from_count = MeanVariable("rho_count", p -> p.mass, simu_params)
    rho = MeanVariable("rho", p -> p.mass, simu_params)
    ux = MeanVariable("ux", p -> p.mass * p.ux, simu_params)
    uy = MeanVariable("uy", p -> p.mass * p.uy, simu_params)
    uz = MeanVariable("uz", p -> p.mass * p.uz, simu_params)
    k = MeanVariable("k", p -> 0.5 * p.mass * ((p.ux - p.means["ux_mean"])^2 + (p.uy - p.means["uy_mean"])^2
            + (p.uz - p.means["uz_mean"])^2), simu_params)
    Rxx = MeanVariable("Rxx", p -> p.mass * (p.ux - p.means["ux_mean"])^2, simu_params)
    Ryy = MeanVariable("Ryy", p -> p.mass * (p.uy - p.means["uy_mean"])^2, simu_params)
    Rzz = MeanVariable("Rzz", p -> p.mass * (p.uz - p.means["uz_mean"])^2, simu_params)
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
        standard_means!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, uz)
        corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, uz)
        divide_by_density!(rho, ux, uy, uz)
        write_means_to_particles!(particles, rho, ux, uy, uz)
        # Second round
        standard_means!(mean_computer, k)
        corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, k)
        divide_by_density!(rho, k)
        write_means_to_particles!(particles, k)
        ## Estimating means on fixed grid and saving
        if it % simu_params.freq_out == 0
            # Estimating: first round
            standard_means!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, uz, is_on_grid=true)
            corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, uz, is_on_grid=true)
            divide_by_density!(rho, ux, uy, uz, is_on_grid=true)
            # Estimating: second round
            standard_means!(mean_computer, k, Rxx, Ryy, Rzz, is_on_grid=true)
            corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, k, Rxx, Ryy, Rzz, is_on_grid=true)
            divide_by_density!(rho, k, Rxx, Ryy, Rzz, is_on_grid=true)
            # Saving
            save_mean_output(it, simu_params.save_directory, "mean", mean_computer, rho, ux, uy, k, Rxx, Ryy, Rzz)
            save_mean_output(it, simu_params.save_directory, "gradx", mean_computer, rho)
            # save_particle_output(it, simu_params.save_directory, particles[1:npart_to_save])
            save_particle_output(-1, simu_params.save_directory, particles)
        end
        ## Advancing flow
        advance_flow!(particles, drift_ux, drift_uy, drift_uz, zero_coef, zero_coef,
                      diff_u, diff_u, diff_u, zero_coef, zero_coef, zero_coef, simu_params)
    end
    save_particle_output(-1, simu_params.save_directory, particles)
end

main_SLM(ARGS[1])
