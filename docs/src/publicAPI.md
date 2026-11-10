```@meta
CurrentModule = LagrangianStochasticConvection
```

# Public API

## Types

```@docs
MeanVariable
Particle
MeanComputer
```

## Fluid particles

```@docs
write_means_to_particles!
advance_flow!
```

## Mean flow estimation

### Standard kernel scheme

```@docs
set_particle_positions!
standard_means!
```

### Corrective scheme

```@docs
common_mean_variables
corrective_SPH!
divide_by_density!
```

## Simulation initialisation

```@docs
read_config_file
initialise_all_particles
read_all_particles_from_file
init_simulation
```

## Output

```@docs
display_simu_params
save_mean_output
save_particle_output
```
