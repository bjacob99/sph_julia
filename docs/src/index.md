# Getting started

## Overview

The `LagrangianStochasticConvection` package offers a convenient framework to simulate 2D turbulent flows through Lagrangian stochastic implementation of PDF methods. In the most general case, the flow is represented by a large number of notional particles, which are tracked through time and evolve according to a set of stochastic differential equations (SDEs). The user is free to define the stochastic model used to evolve the velocity, specific internal energy and turbulent frequency of each notional particle. The coefficients of the SDE may depend on local mean flow variables, which are computed by a kernel-mean estimation scheme, also handled by the `LagrangianStochasticConvection` package.

Even though they can be handled separately by the user, all the tools offered by this package can be conveniently interfaced together via a single configuration file, which can be found along with the source code.

## Installation

The package is not available through the public Julia registry at the moment. You can retrieve it via the dedicated [git repository](https://gitlab.obspm.fr/jphilidet/lagrangian_stochastic_convection):
```
git clone https://gitlab.obspm.fr/jphilidet/lagrangian_stochastic_convection.git
```
Once the package has been downloaded, it can be installed in any Julia session through
```
] add "path/to/git/repo"
```
where `"path/to/git/repo"` is the path to the directory in which you copied the git repository locally. The package should now be available through
```julia
using LagrangianStochasticConvection
```

## No time to read?

Two example scripts may be found in the code repository, in the `scripts` folder
* `SLM.jl` performs a simulation with the Simplified Langevin Model;
* `complex_model.jl` performs a simulation with the more complex Lagrangian stochastic model described in [^Philidet]

To run either script, head to the `scripts` folder, open a Julia session, and type
```
include("SLM.jl")
main_SLM("config.toml")
```
or
```
include("complex_model.jl")
main_complex("config.toml")
```
Once the simulation is finished, the output can be found in `example`. Python routines to take care or post-treatment can be found in the `python` folder (see `--help` of available scripts for more info on how to use them).

## Want to play with the knobs?

Both scripts rely on simulation parameters read from the configuration file `config.toml` found in the same folder. We describe the
different fields found in the configuration file below:
* `initial_directory` is the directory from which the output of a previously run simulation will be read and used to initialise this simulation. It must either be a valid simulation directory, or `"None"`, in which case the simulation will be initialised from scratch;
* `output_directory` is the directory in which this simulation output will be saved. It is created if it does not already exist;
* `aspect_ratio` is the horizontal to vertical domain size ratio;
* `vertical_cells` and `horizontal_cells` are the number of ```\\mathcal{R}``` cells used in the kernel-estimate algorithm (see [Kernel-mean algorithm](@ref) for context);
* `number_modes` is the number of Fourier modes on which the kernel function is expanded (see [Kernel-mean algorithm](@ref) for context);
* `number_time_steps` is the number of time steps for which the simulation will be run;
* `time_step` is the duration of each time step;
* `alpha` and `beta` are the implicitness parameters in the Predictor/Corrector scheme used to integrate the stochastic differential equations forward in time (see [Particle representation](@ref) for context);
* `number_particles` is the number of particles included in the simulation to represent the flow;
* `initial_turbulent_frequency`, `initial_velocity_std` and `initial_energy_std` are the standard deviation of the ```\\omega^\\ast```, ```u^\\ast``` and ```e^\\ast``` distributions according to which the particle properties are initialised. They are only used if `initial_directory == "None"`;
* `is_frequency_fixed` is a Boolean specifying whether the turbulent frequency ```\\omega^\\ast``` of the particles should be set to a fixed value or drawn randomly at the start of the simulation;
* `vertical_grid_points` and `horizontal_grid_points` are the number of rows and columns in the fixed Eulerian 2D grid on which the mean flow will be estimated and saved as output;
* `output_frequency` is the frequency at which the output of the simulation will be saved
* the various field in the `[physics]` section speak for themselves, except perhaps for `is_incomp`, which is a Boolean specifying whether the simulated flow should be incompressible or not. If it is, the volume of the particles remains constant, otherwise it is computed from the local mean density (see [`particle_volumes!`](@ref) for context);
* `type` is a string specifying the type of boundary conditions imposed at the top and bottom boundaries of the domain. It must be either `"temp"`, `"flux"`, `"no_slip"` or `"none"`. See [`apply_boundary_conditions!`](@ref) for an explanation;
* `energy_top` and `energy_bottom` are the set values of internal energy at the top and bottom boundaries, used if the boundary condition `type` is `"temp"` or `"flux"`. Additionally, `energy_top` can be negative, in which case it will instead be computed so that the global energy gradient matches the value expected of a polytropic hydrostatic and thermal equilibrium;
* `flux_bottom` is the energy flux imposed at the bottom boundary, used only if boundary condition `type` is `"flux"`;
* the fields in the `[stochastic_constants]` section give the values of the dimensionless model constants in the complex model described in [^Philidet] and implemented in `complex_model.jl`.

For more info about how to build your own script, head to [Tutorials](@ref).

[^Philidet]: Philidet, J. and Belkacem, K., *in prep*