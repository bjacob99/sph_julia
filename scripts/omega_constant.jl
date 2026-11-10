using LagrangianStochasticConvection
using ProgressBars

include("./omega_constant_stochastic_model.jl")


particles, mean_computer, volume_computer, simu_params = init_simulation(ARGS[1])

include("./omega_constant_mean_variables.jl")


display_simu_params(simu_params)
bar = ProgressBar(simu_params.it_start:simu_params.ntime+simu_params.it_start-1)
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
    standard_means!(mean_computer, Rxx, Ryy, Rzz, Fex, Fey, ke)
    corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, Rxx, Ryy, Rzz, Fex, Fey, ke)
    divide_by_density!(rho, Rxx, Ryy, Rzz, Fex, Fey, ke)
    write_means_to_particles!(particles, Rxx, Ryy, Rzz, Fex, Fey, ke)
    ## Estimating means on fixed grid and saving
    if it % simu_params.freq_out == 0
        # Estimating: first round
        standard_means!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, e, is_on_grid=true)
        corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, e, is_on_grid=true)
        divide_by_density!(rho, ux, uy, e, is_on_grid=true)
        # Estimating: second round
        standard_means!(mean_computer, Rxx, Ryy, Rzz, Fex, Fkin, ke, is_on_grid=true)
        corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, Rxx, Ryy, Rzz, Fex, Fkin, ke, is_on_grid=true)
        divide_by_density!(rho, Rxx, Ryy, Rzz, Fex, Fkin, ke, is_on_grid=true)
        # Saving
        save_mean_output(it, simu_params.save_directory, "mean", mean_computer, rho, ux, uy, e, Rxx, Ryy, Rzz, Fex, Fkin, ke)
        save_mean_output(it, simu_params.save_directory, "gradx", mean_computer, e)
    end
    ## Saving simulation state in case of crash
    if it % 1000 == 0
        save_particle_output(-1, simu_params.save_directory, particles)
    end
    ## Advancing flow
    advance_flow!(particles, drift_ux, drift_uy, drift_uz, drift_energy, drift_omega,
                  diff_ux, diff_uy, diff_uz, diff_energy, diff_omega, extra_diff_omega, simu_params)
end
save_particle_output(-1, simu_params.save_directory, particles)