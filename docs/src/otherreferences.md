```@meta
CurrentModule = LagrangianStochasticConvection
```

# Other references

## Types

```@docs
Positions
SmallCell
BigCell
```

## Fluid particles

```@docs
write_one_mean_to_particles!
advance_particle!
apply_boundary_conditions!
volume_correction
particle_volumes!
check_particle_energy!
check_particle_frequency!
courant_number
check_particle_sanity!
```

## Mean flow estimation

### Standard kernel scheme

```@docs
center_cells
closest_cell
set_particles!
get_particle_ids_in_cell
is_small_cell_in_big_cell
filter_small_cells
set_small_cells!
horizontal_domain_size
small_cell_increment
init_cells
init_evaluating_points
set_positions!
set_eval_positions!
fourier_coef
set_test_data_kernel
set_kernel_coefs!
eval_var
filter_positions
reinitialise_rsum!
update_inside_rsum!
compute_rsum!
ssum_size
compute_ssum
positions_to_cells
standard_mean_one_position
update_means_one_cell!
```

### Corrective scheme

```@docs
M_grad!
M_hess!
A_hess!
C_hess!
grad_RHS!
hess_RHS!
corrective_SPH_one_variable!
divide_mean_by_density
divide_gradient_by_density
divide_hessian_by_density
divide_by_density_one_variable!
```

## Simulation initialisation

```@docs
get_last_time_index
energy_profile
density_profile
initial_energy_spline
initial_density_spline
maximum_function
particle_mass
initialise_one_particle
```