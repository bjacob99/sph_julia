#=
This file defines all methods related to the initialisation of a simulation
=#

#=
Setting up parameters
=#

"""

    get_last_time_index(path::String)

Return the time index of the last saved time step in the simulation directory `path`.
"""
function get_last_time_index(path::String)
    output_path = joinpath(path, "raw_output")
    filenames = readdir(output_path, sort=false)
    first_time_index = 0
    pattern = r"particles_(\d+).dat"
    for filename in filenames
        m = match(pattern, filename)
        if m !== nothing
            time_index = parse(Int64, m.captures[1])
            if time_index > first_time_index
                first_time_index = time_index
            end
        end
    end
    return first_time_index
end

"""

    read_config_file(filename::String)

Read all constant parameters necessary to run a simulation from a configuration file.
The parameters are written to a single `NamedTuple` returned by this function.

If the output directory specified in the configuration file does not exist, it is created. The configuration
file is also copied under the name "config.toml" in the output directory.
"""
function read_config_file(filename::String)
    config = TOML.parsefile(filename)
    # Paths
    initial_directory = config["paths"]["initial_directory"]
    save_directory = config["paths"]["output_directory"]
    if save_directory == "Same"
        save_directory = initial_directory
        first_start_index = get_last_time_index(initial_directory) + 1
    else
        isdir(save_directory) || mkdir(save_directory)
        isdir(joinpath(save_directory, "raw_output")) || mkdir(joinpath(save_directory, "raw_output"))
        cp(filename, joinpath(save_directory, "config.toml"), force=true)
        first_start_index = 1
    end
    # Dimensions
    Lx = 1.0
    Ly = config["spatial"]["aspect_ratio"]
    ncells_x = config["spatial"]["vertical_cells"]
    ncells_y = config["spatial"]["horizontal_cells"]
    nmodes = config["spatial"]["number_modes"]
    # Time
    ntime = config["time"]["number_time_steps"]
    dt = config["time"]["time_step"]
    alpha = config["time"]["alpha"]
    beta = config["time"]["beta"]
    # Particles
    nparticles = config["particles"]["number_particles"]
    init_omega = config["particles"]["initial_turbulent_frequency"]
    init_ustd = config["particles"]["initial_velocity_std"]
    init_estd = config["particles"]["initial_energy_std"]
    is_frequency_fixed = config["particles"]["is_frequency_fixed"]
    # Output
    neval_x = config["output"]["vertical_grid_points"]
    neval_y = config["output"]["horizontal_grid_points"]
    freq_out = config["output"]["output_frequency"]
    # Physical parameters
    npoly = config["physics"]["polytropic_index"]
    gamma1 = config["physics"]["first_adiabatic_index"]
    gravity = config["physics"]["gravity"]
    kappa = config["physics"]["radiative_diffusivity"]
    is_incomp = config["physics"]["is_incomp"]
    # Boundary conditions
    BC_type = config["boundary_conditions"]["type"]
    etop = config["boundary_conditions"]["energy_top"]
    frad_bottom = config["boundary_conditions"]["flux_bottom"]
    if BC_type == "temp"
        grad_e = gamma1 * gravity * (1.0-npoly) / npoly
        ebottom = etop - grad_e * Lx
    elseif BC_type == "flux"
        grad_e = gamma1 * gravity * (1.0-npoly) / npoly
        frad_bottom = -kappa * grad_e
        ebottom = etop - grad_e * Lx
    elseif (BC_type == "no_slip") || (BC_type == "none") # Dummy values
        grad_e = 1.0
        ebottom = 1.0
    else
        error("Wrong bondary conditions specified (should be 'temp', 'flux', 'no_slip' or 'none')")
    end
    # Stochastic model constants
    C0 = config["stochastic_constants"]["C0"]
    C1 = config["stochastic_constants"]["C1"]
    C2 = config["stochastic_constants"]["C2"]
    Cmu = config["stochastic_constants"]["Cmu"]
    Ceps1 = config["stochastic_constants"]["Ceps1"]
    Ceps2 = config["stochastic_constants"]["Ceps2"]
    # Wrapping into metadata NamedTuple
    params_simu = (initial_directory=initial_directory, save_directory=save_directory, Lx=Lx, Ly=Ly,
                   ncells_x=ncells_x, ncells_y=ncells_y, hx=Lx/ncells_x, hy=Ly/ncells_y, nmodes=nmodes, ntime=ntime,
                   dt=dt, alpha=alpha, beta=beta, nparticles=nparticles, init_omega=init_omega, init_ustd=init_ustd,
                   init_estd=init_estd, neval_x=neval_x, neval_y=neval_y, freq_out=freq_out, npoly=npoly,
                   gamma1=gamma1, gravity=gravity, kappa=kappa, is_incomp=is_incomp, BC_type=BC_type, etop=etop, ebot=ebottom,
                   grad_e=grad_e, frad_bottom=frad_bottom, C0=C0, C1=C1, C2=C2, Cmu=Cmu, Ceps1=Ceps1, Ceps2=Ceps2,
                   it_start=first_start_index, is_frequency_fixed=is_frequency_fixed)
    return params_simu
end

#=
Initialising particles
=#

"""

    energy_profile(x::Float64, kwparams::NamedTuple)
    energy_profile(x::Vector{Float64}, kwparams::NamedTuple)

Return the internal energy ``\\widetilde{e}(x)`` of a steady state in hydrostatic and thermal equilibrium at vertical position `x`.
It is given by
```
\\widetilde{e}(x) = \\frac{e_\\mathrm{top} - e_\\mathrm{bot}}{L_x} ~,
```
where ``e_\\mathrm{top}`` and ``e_\\mathrm{bot}`` are specified values of the internal energy at the top and bottom boundaries,
and ``L_x`` is the vertical size of the domain.

The named tuple `kwparams` must contain corresponding fields `"ebot"`, `"etop"` and `"Lx"`.
"""
function energy_profile(x::Float64, kwparams::NamedTuple)
    grad_e = (kwparams.etop - kwparams.ebot) / kwparams.Lx
    return kwparams.ebot + grad_e * x
end
energy_profile(x::Vector{Float64}, kwparams::NamedTuple) = [energy_profile(e, kwparams) for e in x]

"""

    density_profile(x::Float64, kwparams::NamedTuple)
    density_profile(x::Vector{Float64}, kwparams::NamedTuple)

Return the density ``\\overline{\\rho}(x)`` of a polytropic steady state in hydrostatic and thermal equilibrium at vertical position `x`.
It is given by
```
\\overline{\\rho}(x) = \\left( \\frac{\\widetilde{e}(x)}{e_\\mathrm{bot}} \\right)^m ~,
```
where ``\\widetilde{e}(x)`` is the internal energy profile (see [`energy_profile`](@ref)), and `m` is given by
```
m = \\frac{1}{n-1} ~,
```
where `n` is the polytropic index.

The named tuple `kwparams` must contain the necessary fields `"ebot"`, `"etop"` and `"Lx"` for the function [`energy_profile`](@ref),
as well as a field `"npoly"` for the polytropic index.
"""
function density_profile(x::Float64, kwparams::NamedTuple)
    e = energy_profile(x, kwparams)
    mpoly = 1.0 / (kwparams.npoly - 1.0)
    return (e / kwparams.ebot)^mpoly
end
density_profile(x::Vector{Float64}, kwparams::NamedTuple) = [density_profile(e, kwparams) for e in x]

"""

    initial_energy_spline(kwparams::NamedTuple)

Return a spline interpolation of the steady energy profile based on [`energy_profile`](@ref)
"""
function initial_energy_spline(kwparams::NamedTuple)
    x = [i*kwparams.Lx/10000 for i in 0:10000]
    e = energy_profile(x, kwparams)
    spline = linear_interpolation(x, e, extrapolation_bc=Interpolations.Line())
    return spline
end

"""

    initial_density_spline(kwparams::NamedTuple)

Return a spline interpolation of the steady density profile based on [`density_profile`](@ref)
"""
function initial_density_spline(kwparams::NamedTuple)
    x = [i*kwparams.Lx/10000 for i in 0:10000]
    rho = density_profile(x, kwparams)
    spline = linear_interpolation(x, rho, extrapolation_bc=Interpolations.Line())
    return spline
end

"""

    maximum_function(spline::Interpolations.Extrapolation, kwparams::NamedTuple)

Return the maximum of a 1D function on [0,Lx] based on its spline interpolation.
The named tuple `kwparams` must contain a field `"Lx"` up to which the maximum must be searched.
"""
function maximum_function(spline::Interpolations.Extrapolation, kwparams::NamedTuple)
    x = [i*kwparams.Lx/10000 for i in 0:10000]
    return maximum(spline(x))
end

"""

    particle_mass(kwparams::NamedTuple)

Return the mass of fluid represented by each particle, based on simulation parameters. It is calculated via
```
\\Delta m = \\frac{L_y}{N} \\displaystyle\\int_0^{L_x} \\overline{\\rho}(x) ~ \\mathrm{d}x ~,
```
where `L_x` and `L_y` are the vertical and horizontal domain size, `N` the number of particles, and ``\\overline{\\rho}(x)``
the equilibrium density profile specified by the function [`density_profile`](@ref).

The named tuple `kwparams` must contain all the fields necessary for [`initial_density_spline`](@ref),
in addition to fields `"Lx"` and `"Ly"` for the dimensions of the domain, and a field `"nparticles"`
for the total number of particles representing the flow.
"""
function particle_mass(kwparams::NamedTuple)
    spl_rho = initial_density_spline(kwparams)
    x = [i*kwparams.Lx/10000 for i in 0:10000]
    particle_mass = trapz(x, spl_rho(x)) * kwparams.Ly / kwparams.nparticles
    return particle_mass
end

"""

    initialise_one_particle(id::Int64, spline_density::Interpolations.Extrapolation,
                            spline_energy::Interpolations.Extrapolation,
                            max_density::Float64, mass::Float64, kwparams::NamedTuple)

Construct a [`Particle`](@ref) and randomly initialises its properties based on a given steady state
(`spline_density` for mean density, `spline_energy` for mean specific internal energy), and other simulation parameters
contained in the named tuple `kwparams`.

The `id` and `mass` of the particle must be passed as arguments. The rest of the properties are drawn at random:
* the vertical position is drawn randomly so that the distribution matches the specified density profile;
* the horizontal position is drawn randomly with a uniform distribution;
* the three velocity components are drawn randomly according to a Gaussian distribution centered on `0` and with a standard deviation
specified in `kwparams`, under the field `"init_ustd"`;
* the internal energy is drawn randomly according to a Gaussian distribution with a mean equal to the specified internal energy profile
and a standard deviation specified in `kwparams`, under the field `"init_estd"`;
* the turbulent frequency is either set to a fixed value if `kwparams.is_frequency_fixed` is `true`, otherwise drawn randomly with a
``\\Gamma`` distribution (the value specified in `kwparams`, under the field `"init_omega"`, is used in both cases for the mean).

The volume of the particle is initialised as if all particles in the simulation had the same volume.

The maximum of the density profile `max_density` must be passed as a separate argument because computing it independently
for each particle is very time consuming.
"""
function initialise_one_particle(id::Int64, spline_density::Interpolations.Extrapolation, spline_energy::Interpolations.Extrapolation,
                                 max_density::Float64, mass::Float64, kwparams::NamedTuple)
    particle = Particle(id, mass)
    while particle.x == 0.0
        x_rand = rand() * kwparams.Lx
        density_rand = rand() * max_density
        if density_rand <= spline_density(x_rand)
            particle.x = x_rand
        end
    end
    particle.y = rand() * kwparams.Ly
    particle.y_left = particle.y - kwparams.Ly
    particle.y_right = particle.y + kwparams.Ly
    particle.ux = rand(Normal(0.0, kwparams.init_ustd))
    particle.uy = rand(Normal(0.0, kwparams.init_ustd))
    particle.uz = rand(Normal(0.0, kwparams.init_ustd))
    particle.e = rand(Normal(spline_energy(particle.x), kwparams.init_estd))
    if kwparams.is_frequency_fixed
        particle.omega = kwparams.init_omega
    else
        particle.omega = kwparams.init_omega * rand(Gamma(4.0, 0.25))
    end
    particle.volume = kwparams.Lx * kwparams.Ly / kwparams.nparticles
    return particle
end

"""

    initialise_all_particles(kwparams::NamedTuple)

Construct all [`Particle`](@ref)s that will be used in a simulation, based on simulation parameters contained within
the named tuple `kwparams`. The particles are initialised from scratch, at random.
"""
function initialise_all_particles(kwparams::NamedTuple)
    particles::Vector{Particle} = []
    mass = particle_mass(kwparams)
    spline_energy = initial_energy_spline(kwparams)
    spline_density = initial_density_spline(kwparams)
    max_density = maximum_function(spline_density, kwparams)
    for id in 1:kwparams.nparticles
        particle = initialise_one_particle(id, spline_density, spline_energy, max_density, mass, kwparams)
        push!(particles, particle)
    end
    return particles
end

"""

    read_all_particles_from_file!(kwparams::NamedTuple)

Construct all [`Particle`](@ref)s that will be used in a simulation, based on simulation parameters contained within
the named tuple `kwparams`. The particles are read from a text file `"initial_directory/raw_output/final_state.dat"`,
where `initial_directory` must be a field of `kwparams`.

The number of particles specified by `kwparams`, under the field `nparticles`, must correspond to the number of particles
in the input file.
"""
function read_all_particles_from_file(kwparams::NamedTuple)
    mass = particle_mass(kwparams)
    path_to_file = joinpath(kwparams.initial_directory, "raw_output", "final_state.dat")
    particle_properties = readdlm(path_to_file, '\t', Float64, '\n', skipstart=1)
    if kwparams.nparticles != size(particle_properties, 1)
        error("Number of particles does not match input file")
    end
    particles::Vector{Particle} = []
    for id in 1:kwparams.nparticles
        x = particle_properties[id,1]
        y = particle_properties[id,2]
        ux = particle_properties[id,3]
        uy = particle_properties[id,4]
        uz = particle_properties[id,5]
        e = particle_properties[id,6]
        omega = particle_properties[id,7]
        volume = kwparams.Lx * kwparams.Ly / kwparams.nparticles
        particle = Particle(id, mass, x, y, y-kwparams.Ly, y+kwparams.Ly, ux, uy, uz, e, omega, volume, Dict())
        push!(particles, particle)
    end
    return particles
end

#=
Initialising simulation
=#

"""

    init_simulation(filename::String)

Read all simulation parameters from a configuration file `filename`, construct all [`Particle`](@ref)s
that will be used in the simulation, as well as the [`MeanComputer`](@ref) for the kernel-mean estimation
"""
function init_simulation(filename::String)
    simu_params = read_config_file(filename)
    if simu_params.initial_directory == "None"
        particles = initialise_all_particles(simu_params)
    else
        particles = read_all_particles_from_file(simu_params)
    end
    kernel = (x, h) -> @. ifelse(abs(x) <= h, 5.0/4.0/h*(1.0 + 3.0*abs(x/h)) * (1.0 - abs(x/h))^3, 0.0)
    mean_computer = MeanComputer(kernel, simu_params)
    volume_computer = MeanComputer(10, length(particles), 1, 1, 10, 10, simu_params.Lx, simu_params.Ly, kernel)
    return particles, mean_computer, volume_computer, simu_params
end