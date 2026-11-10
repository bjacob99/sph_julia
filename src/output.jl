#=
This file defines all methods related to the outputs of the simulation
=#

"""

    display_simu_params(kwparams::NamedTuple)

Print information about the simulation being run, based on the contents of `kwparams`.
The named tuple must contain the following fields:
* `"Ly"` for the horizontal size of the domain
* `"ncells_x"` and `"ncells_y"` for the number of ``\\mathcal{R}``cells in the kernel-mean formalism
* `"ntime"` and `"dt"` for the number and duration of time steps
* `"nparticles"` for the number of particles
* `"npoly"`, `"gamma1"`, `"gravity"` and `"kappa"` for the polytropic index, adiabatic exponent, gravitational
  acceleration and radiative diffusivity
* `"ebot"` and `"etop"` for the imposed bottom energy and top energy
* `"initial_directory"` and `"save_directory"` for the directory where the initial state of the particles is read,
  and the directory where the output of the present simulation will be saved
"""
function display_simu_params(kwparams::NamedTuple)
    println("\n")
    println("------------------------------------------------------------------------------------------")
    println("RUN PARAMETERS")
    println("    Aspect ratio = $(kwparams.Ly) ($(kwparams.ncells_x) x $(kwparams.ncells_y) cells)")
    println("    $(kwparams.ntime) time steps, Delta t = $(kwparams.dt)")
    println("    Total duration: $(kwparams.ntime*kwparams.dt)")
    println("    Number of particles: $(kwparams.nparticles)")
    println("PHYSICS")
    println("    Polytropic index: n = $(kwparams.npoly)")
    println("    First adiabatic index: Gamma_1 = $(kwparams.gamma1)")
    println("    Gravity = $(kwparams.gravity)")
    println("    Radiative diffusivity = $(kwparams.kappa)")
    println("    Imposed energy: $(kwparams.ebot) in bottom layer, $(kwparams.etop) in top layer")
    println("OUTPUT")
    if kwparams.initial_directory == "None"
        println("    Starting from scratch")
    else
        println("    Initial state from directory: $(kwparams.initial_directory)")
    end
    println("    Saving in directory: $(kwparams.save_directory)")
    println("    Saving every $(kwparams.freq_out) time steps")
    println("------------------------------------------------------------------------------------------")
    println("\n")
end

"""

    save_mean_output(id::Int64, save_dir::String, type::String, mean_computer::MeanComputer, variables::MeanVariable...)

Write estimates of the means of each `variable` in `variables` at each node of a predetermined regular 2D grid
contained in `mean_computer` to a text file. The text file is saved under `XXX/raw_output/ZZZ_YYYYYY.dat`, where
`XXX = save_dir`, and `YYYYYY = id`, padded with zeros to form a 6-digit number, and `ZZZ == type` is the type of means being saved
(must be either "mean", "gradx", "grady", or "laplacian").
"""
function save_mean_output(id::Int64, save_dir::String, type::String, mean_computer::MeanComputer, variables::MeanVariable...)
    if type == "mean"
        all_mean_variables = hcat(mean_computer.eval_pos.x, mean_computer.eval_pos.y, (variable.mean_eval for variable in variables)...)
    elseif type == "gradx"
        all_mean_variables = hcat(mean_computer.eval_pos.x, mean_computer.eval_pos.y, (variable.gradx_eval for variable in variables)...)
    elseif type == "grady"
        all_mean_variables = hcat(mean_computer.eval_pos.x, mean_computer.eval_pos.y, (variable.grady_eval for variable in variables)...)
    elseif type == "laplacian"
        all_mean_variables = hcat(mean_computer.eval_pos.x, mean_computer.eval_pos.y, (variable.hessxx_eval .+ variable.hessyy_eval for variable in variables)...)
    else
        error("Trying to save wrong mean type ($(type))")
    end
    filename = type * "_" * lpad(id, 6, '0') * ".dat"
    path = joinpath(save_dir, "raw_output", filename)
    open(path, "w") do io
        column_names = ["x", "y", (var.name for var in variables)...]
        header = "# " * join(column_names, "    ") * "\n"
        write(io, header)
        writedlm(io, all_mean_variables)
    end
end

"""

    save_particle_output(id::Int64, save_dir::String, particles::Vector{Particle})

Write the properties of all particles in `particles` to a text file. The text file is saved under `XXX/raw_output/particles_YYYYYY.dat`,
where `XXX = save_dir`, and `YYYYYY = id`, padded with zeros to form a 6-digit number. If `id < 0`, the text file is saved
under `save_dir/final_state.dat` instead.
"""
function save_particle_output(id::Int64, save_dir::String, particles::Vector{Particle})
    x_particles = [particle.x for particle in particles]
    y_particles = [particle.y for particle in particles]
    ux_particles = [particle.ux for particle in particles]
    uy_particles = [particle.uy for particle in particles]
    uz_particles = [particle.uz for particle in particles]
    e_particles = [particle.e for particle in particles]
    omega_particles = [particle.omega for particle in particles]
    id < 0 ? filename = "final_state.dat" : filename = "particles_" * lpad(id, 6, '0') * ".dat"
    path = joinpath(save_dir, "raw_output", filename)
    open(path, "w") do io
        write(io, "# x    y    ux    uy    uz    e    omega\n")
        writedlm(io, [x_particles y_particles ux_particles uy_particles uz_particles e_particles omega_particles])
    end
end