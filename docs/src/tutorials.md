# Tutorials

There are essentially two parts to this package. We will first describe how the package can be used to simulate a flow where the coefficients of the Lagrangian stochastic model only depend on the stochastic variables themselves, and not on the mean flow. This is useful for the modelling of homogeneous flows. In a second part, we will describe how the particle behaviour can be coupled to the mean flow variables, and how to use the tools offered by the package to estimate these mean flow variables.

## Hands-on guide: homogeneous flow

### Initialisation

The basic block of the simulation is the [`Particle`](@ref) structure, which encompasses all the physical properties (including the position) of a notional particle. Therefore the first step to running a simulation is to create a set of such [`Particle`](@ref)s, and to correctly initialise them. There are two ways or doing this:
* If you know what you are doing, you may initialise each particle yourself, by setting its position and physical state by hand. This is done via
```julia
particle = Particle(id, mass, x, y, ux, uy, uz, e, omega)
```
where `id` is a unique integer identifier, `mass` is the (constant) mass of the particle, `x` and `y` its 2D coordinates, `ux`, `uy` and `uz` its 3D velocity, `e` its specific internal energy and `omega` its turbulent frequency. If you have computed vectors of particle properties, you may define the full set of notional particles via, for example
```julia
particles = [Particle(id, mass[id], x[id], y[id], ux[id], uy[id], uz[id], e[id], omega[id]) for id in eachindex(mass)]
```
* Alternatively, the package offers a convenient way to initialise the flow through a dedicated configuration file. The file should be read via the [`read_config_file`](@ref) method
```julia
simu_params = read_config_file("path/to/config/file")
```
where "path/to/config/file" is the full path to the configuration file. The method returns a named tuple which contains all the global parameters of the simulation necessary for the rest of the computation. Then, a set of particles can be created via the [`initialise_all_particles`](@ref) method, like so
```julia
particles = initialise_all_particles(simu_params)
```
You may also want to take the final state of a previous simulation and copy the initial state from there. This is done through the [`read_all_particles_from_file`](@ref) method, like so
```julia
particles = read_all_particles_from_file(simu_params)
```
Note that the directory containing the previous simulation used for the initialisation should be contained in the configuration file. It should be a valid simulation directory, and the number of particles should correspond to the specification of the configuration file.

### Setting the Lagrangian stochastic model

In a homogeneous flow, the Lagrangian stochastic model used to model particle evolution only depends on the state of the particle itself, or else on model constants. Each stochastic coefficient should be specified in the form of a `Function`, which should take 2 arguments (a [`Particle`](@ref) and a `NamedTuple` containing all the model constants necessary to the model), and return a `Float`.
As an example, let us assume we want to implement the Langevin Model
```math
d x^\ast = u_x^\ast dt \\
d y^\ast = u_y^\ast dt \\
d u_x^\ast = -u_x^\ast \dfrac{dt}{T} + \sqrt{\dfrac{2u'^2}{T}} dW_x \\
d u_y^\ast = -u_y^\ast \dfrac{dt}{T} + \sqrt{\dfrac{2u'^2}{T}} dW_y \\
d u_z^\ast = -u_z^\ast \dfrac{dt}{T} + \sqrt{\dfrac{2u'^2}{T}} dW_z \\
```
which only models the position and velocities of the particles, not their internal energy or their turbulent frequency. The first two equations are handled by default, and the user does not need to specify them explicitly. The drift coefficients should be defined by
```julia
function drift_ux(particle, params)
    return -particle.ux/params.T
end

function drift_uy(particle, params)
    return -particle.uy/params.T
end

function drift_uz(particle, params)
    return -particle.uz/params.T
end
```
where it can be seen that the particle velocities are attributes of each [`Particle`](@ref), whereas the constant parameter `T` (which is a model constant representing the decay time of turbulent kinetic energy) is an attribute of the global `params` named tuple. In general, each model constant upon which depends the Lagrangian stochastic model must have a dedicated key in a global, common named tuple. Likewise, the diffusion coefficient common to all three components of the velocity should be defined by
```julia
function diffusion(particle, params)
    return sqrt(2.0*params.usquared/params.T)
end
```
The velocity variance `usquared`, which is another model constant, is contained in the same named tuple as `T`, so that a single named tuple can be passed to all these functions. Note that this coefficient does not depend on the particle state. Nevertheless, the function must still expect a [`Particle`](@ref) as argument, even though it does not use it.

If a stochastic variable is not used (as are, in our example, the internal energy and the turbulent frequency), or else if a stochastic variable should in fact remain constant, simply pass "zero" functions to its coefficients
```julia
zero_coeff = (particle, params) -> 0.0
```

In the end, the user should have defined 11 different stochastic coefficients: a drift and diffusion coefficient for each of the 5 stochastic variables implemented in the code (3 components of the velocity + internal energy + turbulent frequency), to which must be added the derivative of the turbulent frequency diffusion coefficient with respect to the turbulent frequency itself.

### Advancing the flow by one time step

In order to advance one [`Particle`](@ref) by one time step, simply use the [`advance_particle!`](@ref) method. The method must be passed not only the `particle`, but also the 11 coefficients of the Lagrangian stochastic model in the form defined in the previous section, and finally a named tuple containing the simulation parameters. For example, to use the Langevin model defined above
```julia
advance_particle!(particle, drift_ux, drift_uy, drift_uz, zero_coeff, zero_coeff, diffusion, diffusion, diffusion, zero_coeff, zero_coeff, zero_coeff, simu_params)
```
The named tuple `simu_params` will be passed as argument to each SDE coefficient, so make sure it contains all the required model constants. Additionally, it should also contain information about the simulation, like the time step for instance. This is why it is recommended to use the configuration file provided with the package to setup the simulation, as this will ensure that the parameters named tuple contains all the information needed. Otherwise, for the list of required fields in `simu_params`, see [`advance_particle!`](@ref).

The package also provides a convenient wrapup method to advance a whole set of particles, in the form of [`advance_flow!`](@ref). It is called in exactly the same way as [`advance_particle!`](@ref), except that it takes a vector of [`Particle`](@ref) instead of a single one as its first argument. For the Langevin model for example
```julia
advance_flow!(particles, drift_ux, drift_uy, drift_uz, zero_coeff, zero_coeff, diffusion, diffusion, diffusion, zero_coeff, zero_coeff, zero_coeff, simu_params)
```
In addition to advancing all the `particles` by one time step, it also takes care of the boundary conditions and updates the `volume` attribute of all particles according to their new position. For that reason, it is recommended to only update the flow through this method, rather than handle each particle independently yourself.

For the moment, there is no choice for the boundary conditions: periodic BC are enforced in the horizontal direction, and wall BC are enforced in the vertical direction.

### Saving particles

In order to save the state of the flow (i.e. of its constitutive particles) at any moment during the simulation, simply call the [`save_particle_output`](@ref) method, like so
```julia
save_particle_output(it, save_dir, particles)
```
where `particles` is the vector of [`Particle`](@ref)s whose complete state will be saved. The output will be written in a plain text file in the form of a table, where each row will contain one `particle` and each column a stochastic variable (from left to right: the vertical position, horizontal position, vertical velocity, horizontal velocity, out-of-plane velocity, specific internal energy, and turbulent frequency). The file will be save in the directory "save_dir/raw_output", either under the name `particles_XXX.dat` with "XXX" the integer `it` padded with 0s, if `it > 0`, or else under the name `final_state.dat` if `it < 0`. This last option allows you to create a special 'end-of-simulation' output file, meant to be read when initialising a new simulatino with this one.

## Hands-on guide: inhomogeneous flows

In most cases, the SDE constitutive of the particle evolution depend on the mean flow at the point where the particle is located. This package offers tools to compute the local mean flow directly from the simulated set of particles, by averaging particle properties over neighbouring particles. In order to filter the neighbouring particles, a weighting kernel function is used, so that we will refer to this scheme as _kernel-mean estimation_. For more details about how it works, and how it is implemented in the code, see [Kernel-mean algorithm](@ref).

The simulation is initialised in the same way as for homogeneous flows (see [Initialisation](@ref)), and the Lagrangian stochastic model is set up in the same way as well (see [Setting the Lagrangian stochastic model](@ref)). The fundamental difference lies in the fact that the coefficients may now depend not only on the state of the particle, but also on the state of the mean flow. As an example, let us consider the following model for velocity
```math
d u_x^\ast = -\left(\dfrac{1}{2} + \dfrac{3}{4}C_0) \omega \left( u_x^\ast - \widetilde{u_x} \right) dt + \sqrt{C_0 \omega k} dW_x \\
d u_y^\ast = -\left(\dfrac{1}{2} + \dfrac{3}{4}C_0) \omega \left( u_y^\ast - \widetilde{u_y} \right) dt + \sqrt{C_0 \omega k} dW_y \\
d u_z^\ast = -\left(\dfrac{1}{2} + \dfrac{3}{4}C_0) \omega \left( u_z^\ast - \widetilde{u_z} \right) dt + \sqrt{C_0 \omega k} dW_z
```
where ``C_0`` is the Kolmogorov constant, ``\omega`` is a fixed frequency, and the equations depend on the mean flow velocity ``\widetilde{u_x}``, ``\widetilde{u_y}``, and ``\widetilde{u_z}``, as well as the turbulent kinetic energy ``k \equiv \left( \widetilde{u_x''^2} + \widetilde{u_y''^2} + \widetilde{u_z''^2} \right) / 2``, where ``u_i'' \equiv u_i - \widetilde{u_i}``.

### Defining mean variables

The first step towards making the Lagrangian stochastic model depend on the mean flow is to define the mean variables on which those coefficients depend. The user may define any mean variable they want, by specifying the particle-based quantity ``Q^{\ast}`` that must be averaged. It must take the form of a `Function` that takes a [`Particle`](@ref) as argument and returns a `Float`. For the density-weighted velocities ``\widetilde{u_x}``, ``\widetilde{u_y}``, and ``\widetilde{u_z}``, we would define
```julia
function recipe_ux(particle)
    return particle.mass * particle.ux
end
function recipe_uy(particle)
    return particle.mass * particle.uy
end
function recipe_uz(particle)
    return particle.mass * particle.uz
end
```
Note that the kernel-mean estimation naturally yields density-weighted average, so that the density itself does not need to appear in the mean variable definitions. For the mean density, one would simply define
```julia
function recipe_rho(particle)
    return particle.mass
end
```
The package provides a convenient [`MeanVariable`](@ref) object to handle all operations involving the mean flow. To construct this object, it is necessary to know in advance
* the `name` of the mean variable, i.e. a `String` uniquely designating the nature of the mean variable
* the number `npart` of notional particle, which corresponds to the number of locations in the domain where the mean flow must be estimated in order to compute the SDE coefficients
* the number `neval` of nodes in the 2D regular grid that will be used to save the output of the code (see [Saving mean flow](@ref))
The reason that `npart` and `neval` must be specified in advance is that they define the size of large arrays that are used to store the mean flow at every location of interest within the domain, and that these large arrays are preallocated for efficiency. The mean vertical velocity ``\widetilde{u_x}``, for instance, may be constructed like so
```julia
name = "ux"
npart = 100000
neval = 10000
mean_ux = MeanVariable(name, npart, neval, recipe_ux)
```
Of course, if the user is not interested in saving any output related to the mean flow, they may simply pass some arbitrary `neval` that will never be used.

In case you are using the configuration file provided with the code, the named tuple `simu_params` contains the information about both `npart` and `neval`, so that you may pass the tuple in place of these parameters. The mean horizontal velocity ``\widetilde{u_y}``, for instance, may be constructed like so
```julia
name = "uy"
mean_uy = MeanVariable(name, recipe_uy, simu_params)
```
The case of the turbulent kinetic energy is a bit more complex, because its recipe itself depends on the mean flow; we will come back to it later

### Computing mean variables

#### Setting up the mean computer

The package provides another convenient [`MeanComputer`](@ref) object encompassing all the information and able to store all the data necessary to compute these [`MeanVariable`](@ref)s. It must be created beforehand, and requires the following parameters (see [Kernel-mean algorithm](@ref) for a definition of the concepts involved):
* `nmode` is the maximum Fourier mode on which the kernel function will be expanded (equals ``M`` in [Kernel-mean algorithm](@ref))
* `npart` is the total number of particles, which must be provided for preallocation purposes
* `(neval_x, neval_y)` is the size of the fixed 2D grid on which the mean flow properties ought to be ouput by the code (the product equals `neval` above)
* `Nx` and `Ny` are the number of rows and columns in the R cell decomposition of the domain. They define the vertical and horizontal size of the kernel.
* `(Lx, Ly)` is the size of the domain.
In addition to these parameters, the [`MeanComputer`](@ref) must be assigned a kernel function that will be used during the kernel-mean estimation. The code assumes that the kernel function is separable (again, see [Kernel-mean algorithm](@ref) for more details), so that the user only needs to define a 1D kernel function. It should take the form ``K(r,h)``, and should therefore take two `Float` arguments and return a single `Float`. For instance
```julia
function kernel(r, h)
    C = 5.0 / 4.0 / h
    return ifelse(abs(r) <= h, C * (1.0 + 3.0*abs(r/h)) * (1.0 - abs(r/h))^3, 0.0)
end
```
Then the [`MeanComputer`](@ref) maybe set up thus
```julia
nmode = 10
npart = 100000
neval_x = 100
neval_y = 100
Nx = 50
Ny = 50
Lx = 1.0
Ly = 3.0
mean_computer = MeanComputer(nmode, npart, neval_x, neval_y, Nx, Ny, Lx, Ly, kernel)
```
Since the `simu_params` named tuple contains all these parameters, it may be passed to the constructor instead, like so
```julia
mean_computer = MeanComputer(kernel, simu_params)
```

#### Computing mean variables

When estimating mean flow variables, the mean computer must first be fed the set of notional particles it will use for the kernel-mean estimation. This is done via the [`set_particle_positions!`](@ref) method, like so
```julia
set_particle_positions!(mean_computer, particles)
```
where `particles` is a `Vector` of [`Particle`](@ref)s. *Note that this must be done each time the particles are susceptible to have evolved, and it is the responsibility of the user to update the mean computer.*

Then the mean variables may be computed via the [`standard_means!`](@ref) method, which updates the storing attributes of each [`MeanVariable`](@ref) fed to it. For instance, if we want to compute all 3 components of the mean flow velocity
```julia
standard_means!(mean_computer, mean_ux, mean_uy, mean_uz)
```
By default, the mean variable is estimated at the particle locations. For code output purposes, the user may want to estimate the means on the nodes of the regular 2D grid that was predetermined when setting up the `mean_computer` instead. This is done by setting the `is_on_grid` option to `true`, like so
```julia
standard_means!(mean_computer, mean_ux, mean_uy, mean_uz, is_on_grid=true)
```

#### Correcting the kernel-mean estimate

The standard scheme described in [Philosophy](@ref) does not have the required consistency properties, and must therefore be corrected. In this package, this is done via the corrective scheme described by [^Korzilius2016]. First, the user will need extra mean variables, related to the position of the particles only. The package provides the [`common_mean_variables`](@ref), which returns the six [`MeanVariable`](@ref) instances that are needed in the corrective kernel-mean estimation
```julia
W, X, Y, XX, XY, YY = common_mean_variables(npart, neval)
```
where `npart` and `neval` are as defined in [Defining mean variables](@ref). Just like the other mean variables, they have to be computed everytime the particles have evolved
```julia
standard_means!(mean_computer, W, X, Y, XX, XY, YY)
```
The corrective kernel-mean estimations can now be computed for any [`MeanVariable`](@ref)s via the [`corrective_SPH!`](@ref) method. For example, to correct the 3 components of the mean velocity
```julia
corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, mean_ux, mean_uy, mean_uy)
```

#### Dividing by mean density

The kernel-mean estimation yields ``\overline{\rho Q}``. In order to obtain ``\widetilde{Q}``, the package offers the method [`divide_by_density!`](@ref). The density [`MeanVariable`](@ref) must be constructed and properly updated beforehand
```julia
density = MeanVariable("density", recipe_rho, simu_params)
standard_means!(mean_computer, density)
```
Then
```julia
divide_by_density!(density, mean_ux, mean_uy, mean_uz)
```

### Using mean variables

#### Transmitting the mean flow state to the particles

So far, we have written the estimate of each mean variable at every particle position in dedicated arrays stored inside [`MeanVariable`](@ref) object. In order to feed these mean variables to the Lagrangian stochastic model, we need to transfer this information to each [`Particle`](@ref). This is done through the [`write_means_to_particles!`](@ref) method, like so
```julia
write_means_to_particles!(particles, ux_mean, uy_mean, uz_mean)
```
*The method assumes that the `particles` are the same that were used to set up the `mean_computer`: it is up to the user to make sure that this is the case!*

#### Using mean flow variables in the Lagrangian stochastic model

Each [`Particle`](@ref) has a `means` attribute which takes the form of a `Dict` relating `String` keys to `Float` values. The keys have the form `XXX_YYY`, where
* `XXX` is the `name` attribute of each [`MeanVariable](@ref) that was computed by the [`MeanComputer`](@ref)
* the `mean_computer` estimates not only the value of the mean variable at each particle location (`YYY == "mean"`), but also its vertical and horizontal gradient (`YYY == "gradx"` and `YYY == "grady"`) and its laplacian (`YYY == "laplacian"`)

Since the information of the mean flow is now contained within each [`Particle`](@ref), it may be used in the definition of new mean variables. For instance, in the Simplified Langevin Model described above, we need to compute the turbulent kinetic energy. This may be done via
```julia
function recipe_k(particle)
    dux = particle.ux - particle.means["ux_mean"]
    duy = particle.uy - particle.means["uy_mean"]
    duz = particle.uz - particle.means["uz_mean"]
    return particle.mass * (dux^2 + duy^2 + duz^2)
end
mean_kinetic_energy = MeanVariable("k", recipe_k, simu_params)
standard_means!(mean_computer, mean_kinetic_energy)
corrective_SPH!(mean_computer, W, X, Y, XX, XY, YY, mean_kinetic_energy)
divide_by_density!(density, mean_kinetic_energy)
write_means_to_particles!(particles, mean_kinetic_energy)
```

Likewise, it may be used in defining the coefficients of the SDE. For instance
```julia
function drift_ux(particle, params)
    dux = particle.ux - particle.means["ux_mean"]
    omega_eff = (0.5 + 0.75*params.C0)*params.omega
    return -omega_eff * dux
end
function drift_uy(particle, params)
    duy = particle.uy - particle.means["uy_mean"]
    omega_eff = (0.5 + 0.75*params.C0)*params.omega
    return -omega_eff * duy
end
function drift_uz(particle, params)
    duz = particle.uz - particle.means["uz_mean"]
    omega_eff = (0.5 + 0.75*params.C0)*params.omega
    return -omega_eff * duz
end
function diffusion(particle, params)
    return sqrt(params.C0 * params.omega * particle.means["k_mean"])
end
```

#### Saving mean flow

In addition to the state of the particles, the user may want to save the mean flow quantities as well. In order to facilitate the visualisation and data analysis, the mean variables may be computed on the nodes of a predetermined regular 2D grid instead of the particle locations. The regular grid is built when setting up the [`MeanComputer`](@ref), based on the number of rows `neval_x` and of columns `neval_y` specified by the user, either by hand or through the configuration file (see [Setting up the mean computer](@ref)). In order to compute the mean variables on this grid rather than at the particle locations, each of the methods [`standard_means!](@ref), [`corrective_SPH!`](@ref) and [`divide_by_density!`](@ref) must be passed a keyword argument `is_on_grid = true`.

The mean flow variables may be saved to an output text file through the [`save_mean_output`](@ref) method. For instance, to save the mean density, the three components of the mean flow velocity and the turbulent kinetic energy
```julia
save_particle_output(it, save_dir, mean_rho, mean_ux, mean_uy, mean_uz, mean_kinetic_energy)
```
As with the [`save_particle_output`](@ref) method, the mean variables file will be saved in the directory "save_dir/raw_output", under the name `means_XXX.dat` with "XXX" the integer `it` padded with 0s.

[^Korzilius2016]: Korzilius, S., Schilders, W. H., & Anthonissen, M. J. 2016, Journal of Applied Mathematics and Physics, 5, 168