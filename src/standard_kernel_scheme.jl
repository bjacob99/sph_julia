#=
This file defines all methods related to the standard kernel estimation of ensemble averages.
=#

#=
Update of SmallCell
=#

"""

    center_cells(cells::Vector{SmallCell})
    center_cells(cells::Vector{BigCell})

Compute the 2D coordinates of the centers of a set of ``\\mathcal{R}`` cells or ``\\mathcal{S}`` cells,
and organise them into tuples

## Arguments:
* `cells` is a `Vector` of ``N`` cells (either [`SmallCell`](@ref)s or [`BigCell`](@ref)s)

## Returns:
* a 2D-array `center_cells` (2x``N``) containing the 2D coordinates of the cell centers, with the first index referring to the
  cell, and the second index to the coordinate
"""
function center_cells(cells::Union{Vector{SmallCell},Vector{BigCell}})
    center_cells_tuple = [((cell.x0 + cell.x1)/2.0, (cell.y0 + cell.y1)/2.0) for cell in cells]
    center_cells = [center_cells_tuple[icell][icoor] for icoor in 1:2, icell in eachindex(cells)]
end

"""

    closest_cell(tree::KDTree, x::Float64, y::Float64, y_to_add::Float64)
    closest_cell(tree::KDTree, particle::Particle, y_to_add::Float64)

Identifies the cell in which a given point is located. This is done on the basis of a `tree` which is passed
as argument, and must be computed beforehand.

## Arguments:
* `tree` is a `KDTree` with the cell centers as generators
* `x` and `y` are the 2D coordinates of the point to be located, or...
* ... `particle` is the particle to be located
* the horizontal position of the point is incremented by `y_to_add` before being located

## Return:
* the index `id` of the `tree` element in which the point in located
"""
function closest_cell(tree::KDTree, x::Float64, y::Float64, y_to_add::Float64)
    id_cell, _ = knn(tree, [x, y + y_to_add], 1)
    return id_cell[1]
end
closest_cell(tree::KDTree, particle::Particle, y_to_add::Float64) = closest_cell(tree, particle.x, particle.y, y_to_add)

"""

    set_particles!(particules::Vector{Particle}, cells::Vector{SmallCell}, hy::Float64, Ly::Float64)

Update the `particles` attribute of a `Vector` of [`SmallCell`](@ref)s based on where each particle in a
given set is located.

## Arguments:
* `cells` is the `Vector` of [`SmallCell`](@ref)s to be updated
* `particles` is a `Vector` of [`Particle`](@ref) to be distributed to each cell
* `hy` is the maximum distance from the left (resp. right) edge for a particle to be added to the cell on the
extreme right (resp. left) of the same row
* `Ly` is the horizontal shift incurred by a particle which is added to the extreme right (resp. left) cell on the
same row if it is less than `hy` away from the left (resp. right) edge of the domain
"""
function set_particles!(particules::Vector{Particle}, cells::Vector{SmallCell}, hy::Float64, Ly::Float64)
    for cell in cells
        cell.particles = []
    end
    centers = center_cells(cells)
    tree = KDTree(centers)
    for particle in particules
        id_cell = closest_cell(tree, particle, 0.0)
        push!(cells[id_cell].particles, particle)
        if particle.y < hy
            id_cell = closest_cell(tree, particle, Ly)
            push!(cells[id_cell].particles, particle)
        elseif particle.y > Ly - hy
            id_cell = closest_cell(tree, particle, -Ly)
            push!(cells[id_cell].particles, particle)
        end
    end
end

"""

    get_particle_ids_in_cell(cell::SmallCell)

Return a `Vector` of the `id`s of all particles in a given cell.
"""
function get_particle_ids_in_cell(cell::SmallCell)
    return [particle.id for particle in cell.particles]
end

#=
Update of BigCell
=#

"""

    is_small_cell_in_big_cell(sc::SmallCell, bc::BigCell)

Determine whether a given ``\\mathcal{R}`` cell belongs to a given ``\\mathcal{S}`` cell.
"""
function is_small_cell_in_big_cell(sc::SmallCell, bc::BigCell)
    sc_center_x = (sc.x0 + sc.x1) / 2.0
    sc_center_y = (sc.y0 + sc.y1) / 2.0
    return (bc.x0 <= sc_center_x) && (sc_center_x <= bc.x1) && (bc.y0 <= sc_center_y) && (sc_center_y <= bc.y1)
end

"""
    filter_small_cells(sc::Vector{SmallCell}, bc::BigCell)

Filter a set of ``\\mathcal{R}`` cells based on whether they belong to a given ``\\mathcal{S}`` cell.
"""
function filter_small_cells(sc::Vector{SmallCell}, bc::BigCell)
    return filter(x -> is_small_cell_in_big_cell(x, bc), sc)
end

"""

    set_small_cells!(sc::Vector{SmallCell}, bc::BigCell)

Update the `smallcells` attribute of a given ``\\mathcal{S}`` cell based on which of a given set of
``\\mathcal{R}`` cells belongs to it.
"""
function set_small_cells!(sc::Vector{SmallCell}, bc::BigCell)
    new_sc = filter_small_cells(sc, bc)
    bc.smallcells = new_sc
end

#=
Initialisation and update of MeanComputer
=#

"""

    horizontal_domain_size(mean_computer::MeanComputer)

Return the horizontal domain size from a given [`MeanComputer`](@ref), based on its set of ``\\mathcal{R}`` cells
"""
function horizontal_domain_size(mean_computer::MeanComputer)
    right_side_rcells = [cell for cell in mean_computer.small_cells if cell.increment_y < 0.0]
    for i in eachindex(right_side_rcells)
        if right_side_rcells[i].y0 != right_side_rcells[1].y0
            throw(ArgumentError("R cells in right column have differing left edges ($(right_side_rcells[i].y0) and $(right_side_rcells[1].y0)) "))
        end
    end
    return right_side_rcells[1].y0
end

"""

    small_cell_increment(y0::Float64, y1::Float64, Ly::Float64)

Determine by how much the horizontal position of a [`Particle`](@ref) must be shifted to assess
whether it is in a given ``\\mathcal{R}`` cell: if the ``\\mathcal{R}`` cell is on the extreme left
(resp. right) column, it contains the particles located in the extreme right (resp. left) column.
"""
function small_cell_increment(y0::Float64, y1::Float64, Ly::Float64)
    if (y0+y1)/2.0 < 0.0
        return Ly
    elseif (y0+y1)/2.0 > Ly
        return -Ly
    else
        return 0.0
    end
end

"""

    init_cells(nmodes::Int64, hx::Float64, hy::Float64, Nx::Int64, Ny::Int64, Ly::Float64)

Determine the [`SmallCell`](@ref)s and [`BigCell`](@ref)s for a given 2D domain

## Arguments:
* `nmodes` is the attribute of the same name with which each of the ``\\mathcal{R}`` cells are instanciated
* `(Nx, Ny)` are the number of rows and columns in the ``\\mathcal{R}`` cell paving of the domain
* `(hx, hy)` are the vertical and horizontal size of the kernel function (defining the size of the cells)
* `Ly` is the horizontal size of the domain

## Returns:
* `rcells`, a `Vector` of [`SmallCell`](@ref)s completely paving the domain.
* `scells`, the associated `Vector` of [`BigCell`](@ref)s
"""
function init_cells(nmodes::Int64, hx::Float64, hy::Float64, Nx::Int64, Ny::Int64, Ly::Float64)
    rcells = vcat([SmallCell(nmodes, i*hx, j*hy, (i+1)*hx, (j+1)*hy, small_cell_increment(j*hy, (j+1)*hy, Ly)) for i in 0:Nx-1, j in -1:Ny]...)
    scells = vcat([BigCell(i*hx, j*hy, (i+3)*hx, (j+3)*hy) for i in -1:Nx-2, j in -1:Ny-2]...)
    return rcells, scells
end

"""

    init_evaluating_points(neval_x::Int64, neval_y::Int64, Lx::Float64, Ly::Float64)

Determine the 2D coordinates of a regular Cartesian grid

## Arguments:
* `neval_x` and `neval_y` are the number of rows and columns in the grid
* `(Lx, Ly)` are the dimensions of the domain

## Returns:
* `xeval` a 1D-array of the vertical coordinate of each point in the grid
* `yeval` a 1D-array of the horizontal coordinate of each point in the grid
"""
function init_evaluating_points(neval_x::Int64, neval_y::Int64, Lx::Float64, Ly::Float64)
    xeval = vcat([Lx*i/(neval_x+1) for i in 1:neval_x, j in 1:neval_y]...)
    yeval = vcat([Ly*j/(neval_y+1) for i in 1:neval_x, j in 1:neval_y]...)
    return xeval, yeval
end

"""

    set_positions!(positions::Positions, x::Vector{Float64}, y::Vector{Float64}, lx::Float64, ly::Float64)

Update a [`Positions`](@ref) object based on the provided set of 2D coordinates

## Arguments:
* `positions` is the [`Positions`](@ref) to be updated
* `x` and `y` are the new 2D coordinates
* `lx` and `ly` are the quantities ``l_x`` and ``l_y`` appearing in the definition of the trigonometric functions
  `sine_x`, `cosine_x`, `sine_y` and `cosine_y` (see [Positions](@ref) for more context)
"""
function set_positions!(positions::Positions, x::Vector{Float64}, y::Vector{Float64}, lx::Float64, ly::Float64)
    @. positions.x = x
    @. positions.y = y
    for i in 0:positions.nmodes-1
        @. positions.sine_x[i] = sin(2pi * i * x / lx)
        @. positions.cosine_x[i] = cos(2pi * i * x / lx)
        @. positions.sine_y[i] = sin(2pi * i * y / ly)
        @. positions.cosine_y[i] = cos(2pi * i * y / ly)
    end
end

"""

    set_eval_positions!(mean_computer::MeanComputer, x::Vector{Float64}, y::Vector{Float64})

Update the points of the fixed 2D grid on which the mean variables will be saved by the code

## Arguments:
* `mean_computer` is the [`MeanComputer`](@ref) whose attribute `eval_pos` is updated
* `x` and `y` are the 2D coordinates of the fixed grid points
"""
function set_eval_positions!(mean_computer::MeanComputer, x::Vector{Float64}, y::Vector{Float64})
    set_positions!(mean_computer.eval_pos, x, y, mean_computer.lx, mean_computer.ly)
end

"""

    fourier_coef(x::Vector{Float64}, y::Vector{Float64}, freq::Float64)

Return the Fourier coefficient of a given function at a given frequency

## Arguments:
* the function is passed in the form of a `Vector` `y` evaluated on a `Vector` `x`
* `freq` is the frequency for which the coefficient is returned

## Returns:
* the Fourier coefficient (`Float`)
"""
function fourier_coef(x::Vector{Float64}, y::Vector{Float64}, freq::Float64)
    cosx = @. cos(2pi * x * freq)
    return trapz(x, y .* cosx) / trapz(x, cosx.^2)
end

"""

    set_test_data_kernel(kernel_function::Function, hx::Float64, hy::Float64, lx::Float64, ly::Float64)

Return the vertical and horizontal kernel functions evaluated on regular grid spanning a whole ``\\mathcal{S}`` cell.

## Arguments:
* `kernel_function` should take two arguments (a `Vector` of distance `Float`, and a size `Float`), and return a `Float`
* `(hx, hy)` are the dimensions of the kernel function
* `(lx, ly)` are the dimensions of the ``\\mathcal{S}`` cells

## Returns:
* `Kx` is the vertical kernel function evaluated on the regular grid `x` spanning `-lx/2.0, lx/2.0`
* `Ky` is the horizontal kernel function evaluated on the regular grid `y` spanning `-ly/2.0, ly/2.0`
"""
function set_test_data_kernel(kernel_function::Function, hx::Float64, hy::Float64, lx::Float64, ly::Float64)
    x = collect(LinRange{Float64}(-lx/2.0, lx/2.0, 10000))
    y = collect(LinRange{Float64}(-ly/2.0, ly/2.0, 10000))
    Kx = kernel_function(x, hx)
    Ky = kernel_function(y, hy)
    return x, y, Kx, Ky
end

"""

    set_kernel_coefs!(mean_computer::MeanComputer)

Compute the Fourier coefficients of the kernel function of a [`MeanComputer`](@ref), and update its `ap` and `aq` attributes accordingly.
"""
function set_kernel_coefs!(mean_computer::MeanComputer)
    x, y, Kx, Ky = set_test_data_kernel(mean_computer.kernel_function, mean_computer.hx, mean_computer.hy,
                                        mean_computer.lx, mean_computer.ly)
    for p in 0:mean_computer.nmodes-1
        mean_computer.ap[p+1] = fourier_coef(x, Kx, p/mean_computer.lx)
        mean_computer.aq[p+1] = fourier_coef(y, Ky, p/mean_computer.ly)
    end
end

"""

    set_particle_positions!(mean_computer::MeanComputer, particles::Vector{Particle})

Update the 2D coordinates of the particles in a [`MeanComputer`](@ref), based on a set of [`Particle`](@ref)s passed as argument
"""
function set_particle_positions!(mean_computer::MeanComputer, particles::Vector{Particle})
    xpart = [particle.x for particle in particles]
    ypart = [particle.y for particle in particles]
    set_positions!(mean_computer.part_pos, xpart, ypart, mean_computer.lx, mean_computer.ly)
    set_particles!(particles, mean_computer.small_cells, mean_computer.hy, horizontal_domain_size(mean_computer))
end

#=
Computation of R sums
=#

"""

    eval_var(variable::MeanVariable, particle::Particle, y_to_add::Float64)

Evaluate the quantity ``Q^\\ast`` of a [`MeanVariable`](@ref) for a given [`Particle`](@ref), with its horizontal
position incremented by `y_to_add` beforehand
"""
function eval_var(variable::MeanVariable, particle::Particle, y_to_add::Float64)
    if y_to_add == 0.0
        return variable.func(particle)::Float64
    elseif y_to_add > 0.0
        return variable.func_right(particle)::Float64
    else
        return variable.func_left(particle)::Float64
    end
end

"""

    filter_positions(positions::Positions, ids::Vector{Int64}, y_to_add::Float64, ly::Float64)
    filter_positions(positions::Positions, ids::Vector{Int64})

Initialise a new [`Positions`](@ref) from an old [`Positions`](@ref), only keeping the indices contained in `ids`.
The horizontal coordinates are shifted by ``y_to_add`` in the process (defaults to 0). If `y_to_add` is provided,
`positions.sine_y` and `positions.cosine_y` must be recomputed, and the spatial frequency `l_y` must therefore
also be passed as argument.
"""
function filter_positions(positions::Positions, ids::Vector{Int64}, y_to_add::Float64, ly::Float64)
    if length(ids) == 0
        return Positions(positions.nmodes, 0)
    elseif minimum(ids) < 1 || maximum(ids) > positions.npos
        throw(ArgumentError("Invalid slice of positions"))
    else
        new_positions = Positions(positions.nmodes, length(ids))
        @. new_positions.x = positions.x[ids]
        @. new_positions.y = positions.y[ids] - y_to_add
        for p in 0:positions.nmodes-1
            @. new_positions.cosine_x[p] = positions.cosine_x[p][ids]
            @. new_positions.sine_x[p] = positions.sine_x[p][ids]
            @. new_positions.cosine_y[p] = positions.cosine_y[p][ids] * cos(2pi * p * y_to_add / ly) + positions.sine_y[p][ids] * sin(2pi * p * y_to_add / ly)
            @. new_positions.sine_y[p] = positions.sine_y[p][ids] * cos(2pi * p * y_to_add / ly) - positions.cosine_y[p][ids] * sin(2pi * p * y_to_add / ly)
        end
        return new_positions
    end
end
filter_positions(positions::Positions, ids::Vector{Int64}) = filter_positions(positions, ids, 0.0, 1.0)

"""
    update_inside_rsum!(positions::Positions, evaluated_variable::Vector{Float64}, var_cosx::Vector{Float64}, var_sinx::Vector{Float64},
                        var_cosx_cosy::Vector{Float64}, var_cosx_siny::Vector{Float64}, var_sinx_cosy::Vector{Float64},
                        var_sinx_siny::Vector{Float64}, need_to_compute_p::Bool, p::Int64, q::Int64)

Update `var_cosx_cosy`, `var_sinx_cosy`, `var_sinx_cosy`, `var_sinx_siny` with the arguments of the four *R sums* in the
kernel-means formalism for one given Fourier mode `(p,q)`, and a set `evaluated_variable` of ``Q^\\ast`` to be averaged.
The [`Positions`](@ref) containing the cosines and sines are passed in `positions`.

If `need_to_compute_p == true`, also updates the intermediaries `var_cosx` and `var_cosy`.
"""
function update_inside_rsum!(positions::Positions, evaluated_variable::Vector{Float64}, var_cosx::Vector{Float64}, var_sinx::Vector{Float64},
                             var_cosx_cosy::Vector{Float64}, var_cosx_siny::Vector{Float64}, var_sinx_cosy::Vector{Float64},
                             var_sinx_siny::Vector{Float64}, need_to_compute_p::Bool, p::Int64, q::Int64)
    if (p < 0) || (positions.nmodes-1 < p) || (q < 0) || (positions.nmodes-1 < q)
        throw(ArgumentError("Invalid kernel Fourier mode selected (maximum is $(positions.nmodes-1))"))
    else
        if need_to_compute_p
            @. var_cosx = evaluated_variable * positions.cosine_x[p]
            @. var_sinx = evaluated_variable * positions.sine_x[p]
        end
        @. var_cosx_cosy = var_cosx * positions.cosine_y[q]
        @. var_cosx_siny = var_cosx * positions.sine_y[q]
        @. var_sinx_cosy = var_sinx * positions.cosine_y[q]
        @. var_sinx_siny = var_sinx * positions.sine_y[q]
    end
end

"""

    reinitialise_rsum!(cell::SmallCell, nvar::Int64)

Reset the `rsum_CC`, `rsum_SC`, `rsum_CS` and `rsum_SS` attributes of a [`SmallCell`](@ref) to 3D-arrays full of 0s,
with a new first-axis of length `nvar` (the other two axes have untouched dimension).
"""
function reinitialise_rsum!(cell::SmallCell, nvar::Int64)
    nmode = size(cell.rsum_CC, 2)
    cell.rsum_CC = zeros(nvar, nmode, nmode)
    cell.rsum_CS = zeros(nvar, nmode, nmode)
    cell.rsum_SC = zeros(nvar, nmode, nmode)
    cell.rsum_SS = zeros(nvar, nmode, nmode)
end

"""

    compute_rsum!(cell::SmallCell, mean_computer::MeanComputer, variables::MeanVariable...)

Update the `rsum_CC`, `rsum_CS`, `rsum_SC` and `rsum_SS` attribute of `cell` with all *R sums* for each element 
in `variables` (see [Kernel-mean algorithm](@ref) for more context)

## Arguments:
* `cell` is the [`SmallCell`](@ref) to be updated
* `mean_computer` is the [`MeanComputer`](@ref) containing all the information required for the *R sums* to be computed
* `variables` is a set of [`MeanVariable`](@ref)s for which the *R sums* should be computed
"""
function compute_rsum!(cell::SmallCell, mean_computer::MeanComputer, variables::MeanVariable...)
    reinitialise_rsum!(cell, length(variables))
    positions = filter_positions(mean_computer.part_pos, get_particle_ids_in_cell(cell), cell.increment_y, mean_computer.ly)
    npart = positions.npos
    vcx, vsx, CC, CS, SC, SS = zeros(npart), zeros(npart), zeros(npart), zeros(npart), zeros(npart), zeros(npart)
    for (ivar, variable) in enumerate(variables)
        evaluated_variable = [eval_var(variable, particle, cell.increment_y) for particle in cell.particles]
        for p in 0:mean_computer.nmodes-1
            for q in 0:mean_computer.nmodes-1
                update_inside_rsum!(positions, evaluated_variable, vcx, vsx, CC, CS, SC, SS, q == 0, p, q)
                cell.rsum_CC[ivar, p+1, q+1] = sum(CC)
                cell.rsum_CS[ivar, p+1, q+1] = sum(CS)
                cell.rsum_SC[ivar, p+1, q+1] = sum(SC)
                cell.rsum_SS[ivar, p+1, q+1] = sum(SS)
            end
        end
    end
    positions = 0
    vcx = 0
    vsx = 0
    CC = 0
    CS = 0
    SC = 0
    SS = 0
    evaluated_variable = 0
end

#=
Computation of S sums
=#

"""

    ssum_size(cell::BigCell)

Return the number of variables `nvar` and the number of Fourier modes `nmode`
for which *S sums* are computed in a given [`BigCell`](@ref).
"""
function ssum_size(cell::BigCell)
    if length(cell.smallcells) == 0
        return 1, 1
    else
        nvar = size(cell.smallcells[1].rsum_CC, 1)
        nmode = size(cell.smallcells[1].rsum_CC, 2)
        for i in eachindex(cell.smallcells)
            if size(cell.smallcells[i].rsum_CC, 1) != nvar
                throw(ArgumentError("R cells sums have been computed for different number of variables"))
            elseif size(cell.smallcells[i].rsum_CC, 2) != nmode || size(cell.smallcells[i].rsum_CC, 3) != nmode
                throw(ArgumentError("R cell sums have been computed for different number of modes"))
            end
        end
        return nvar, nmode
    end
end

"""

    compute_ssum(cell::BigCell)

Compute all *S sums* of a given [`BigCell`](@ref)s by summing over all ``\\mathcal{R}`` cells within.
"""
function compute_ssum(cell::BigCell)
    nvar, nmode = ssum_size(cell)
    CC, CS, SC, SS = zeros(nvar, nmode, nmode), zeros(nvar, nmode, nmode), zeros(nvar, nmode, nmode), zeros(nvar, nmode, nmode)
    for small_cell in cell.smallcells
        @. CC += small_cell.rsum_CC
        @. CS += small_cell.rsum_CS
        @. SC += small_cell.rsum_SC
        @. SS += small_cell.rsum_SS
    end
    return CC, CS, SC, SS
end

#=
Standard kernel estimation scheme
=#

"""

    positions_to_cells(positions::Positions, cells::Union{Vector{SmallCell}, Vector{BigCell}})

Map each [`SmallCell`](@ref) or [`BigCell`](@ref) to a list of all the points inside

## Arguments:
* `positions` is the list of point to be distributed among the different `cells`
* `cells` is the list of cells to which the points will be distributed

## Returns:
* a `Dict` `map_cells_to_positions` where the keys are indices referring to the cells, and the values are `Vector` of
  indices referring to all the points from `positions` that are inside the cell in question.
"""
function positions_to_cells(positions::Positions, cells::Union{Vector{SmallCell}, Vector{BigCell}})
    centers = center_cells(cells)
    tree = KDTree(centers)
    map_cells_to_positions = Dict{Int64,Vector{Int64}}(icell => [] for icell in eachindex(cells))
    for (ipos, x, y) in collect(zip(eachindex(positions.x), positions.x, positions.y))
        id_cell = closest_cell(tree, x, y, 0.0)
        push!(map_cells_to_positions[id_cell], ipos)
    end
    centers = 0
    tree = 0
    return map_cells_to_positions
end

"""
    standard_mean_one_position(lx::Float64, ly::Float64,
                               cosx::Vector{Float64}, sinx::Vector{Float64}, cosy::Vector{Float64}, siny::Vector{Float64},
                               ap::Vector{Float64}, aq::Vector{Float64},
                               CC::Matrix{Float64}, CS::Matrix{Float64}, SC::Matrix{Float64}, SS::Matrix{Float64})

Compute the standard kernel estimate of one mean variable, at one point in the domain.

## Arguments:
All the arguments can be unambiguously related to the equations presented in [Kernel-mean algorithm](@ref)

## Returns:
* `var` is a `Float` representing the mean field at the input position
* `gradx` and `grady` are two `Float`s representing its first-order derivatives
* `hessxx`, `hessxy` and `hessyy` are three `Float`s representing its second-order derivatives
"""
function standard_mean_one_position(lx::Float64, ly::Float64,
                                    cosx::Vector{Float64}, sinx::Vector{Float64}, cosy::Vector{Float64}, siny::Vector{Float64},
                                    ap::Vector{Float64}, aq::Vector{Float64},
                                    CC::SubArray{Float64}, CS::SubArray{Float64}, SC::SubArray{Float64}, SS::SubArray{Float64})
    var, gradx, grady, hessxx, hessxy, hessyy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    nmode = length(cosx)
    for ip in 1:nmode
        ax = ap[ip]
        cx = cosx[ip]
        sx = sinx[ip]
        for iq in 1:nmode
            ay = aq[iq]
            cy = cosy[iq]
            sy = siny[iq]
            cosx_cosy = CC[ip,iq]
            cosx_siny = CS[ip,iq]
            sinx_cosy = SC[ip,iq]
            sinx_siny = SS[ip,iq]
            var += ax*ay*(cx*cy*cosx_cosy + cx*sy*cosx_siny + sx*cy*sinx_cosy + sx*sy*sinx_siny)
            gradx += ax*ay*2pi*(ip-1)/lx*(-sx*cy*cosx_cosy - sx*sy*cosx_siny + cx*cy*sinx_cosy + cx*sy*sinx_siny)
            grady += ax*ay*2pi*(iq-1)/ly*(-cx*sy*cosx_cosy + cx*cy*cosx_siny - sx*sy*sinx_cosy + sx*cy*sinx_siny)
            hessxx += ax*ay*(2pi*(ip-1)/lx)^2*(-cx*cy*cosx_cosy - cx*sy*cosx_siny - sx*cy*sinx_cosy - sx*sy*sinx_siny)
            hessxy += ax*ay*(2pi*(ip-1)/lx)*(2pi*(iq-1)/ly)*(sx*sy*cosx_cosy - sx*cy*cosx_siny - cx*sy*sinx_cosy + cx*cy*sinx_siny)
            hessyy += ax*ay*(2pi*(iq-1)/ly)^2*(-cx*cy*cosx_cosy - cx*sy*cosx_siny - sx*cy*sinx_cosy - sx*sy*sinx_siny)
        end
    end
    return var, gradx, grady, hessxx, hessxy, hessyy
end

"""

    update_means_one_cell!(average::Matrix{Float64}, gradx::Matrix{Float64}, grady::Matrix{Float64},
                           hessxx::Matrix{Float64}, hessxy::Matrix{Float64}, hessyy::Matrix{Float64},
                           mean_computer::MeanComputer, cell::BigCell, ids::Vector{Int64}; is_on_grid::Bool=false)

Compute the kernel estimate of all required mean variables inside a given ``\\mathcal{S}`` cell.
By default, the means are estimated at every particle location, based on `mean_computer.part_pos`. If `is_on_grid == true`,
they are estimated at every node of a regular rectangular grid instead, based on `mean_computer.eval_pos`.

## Arguments:
* `average`, `gradx`, `grady`, `hessxx`, `hessxy`, and `hessyy` are 2D-arrays containing all the required means, where the
  first index refers to the different mean variables being computed, and the second index refers to the locations at which
  the means are estimated. Only the locations inside the cell will be updated
* `mean_computer` is the [`MeanComputer`](@ref) wrapping all relevant information for the kernel-means algorithm
* `cell` is the [`BigCell`](@ref) containing all the locations for which the means will be computed
* `ids` is the `Vector` of indices of the array `mean_computer.part_pos` or `mean_computer.eval_pos`
  pointing to all the positions inside `cell`. It is passed as argument to the method call because computing it separately for each
  `cell` independently is very time consuming
"""
function update_means_one_cell!(average::Matrix{Float64}, gradx::Matrix{Float64}, grady::Matrix{Float64}, hessxx::Matrix{Float64}, hessxy::Matrix{Float64}, hessyy::Matrix{Float64},
                                mean_computer::MeanComputer, cell::BigCell, ids::Vector{Int64}; is_on_grid::Bool=false)
    CC, CS, SC, SS = compute_ssum(cell)
    is_on_grid ? positions = filter_positions(mean_computer.eval_pos, ids) : positions = filter_positions(mean_computer.part_pos, ids)
    cosx_2D = [positions.cosine_x[imode][ipos] for imode in 0:positions.nmodes-1, ipos in eachindex(ids)]
    sinx_2D = [positions.sine_x[imode][ipos] for imode in 0:positions.nmodes-1, ipos in eachindex(ids)]
    cosy_2D = [positions.cosine_y[imode][ipos] for imode in 0:positions.nmodes-1, ipos in eachindex(ids)]
    siny_2D = [positions.sine_y[imode][ipos] for imode in 0:positions.nmodes-1, ipos in eachindex(ids)]
    cosx, sinx, cosy, siny = zeros(positions.nmodes), zeros(positions.nmodes), zeros(positions.nmodes), zeros(positions.nmodes)
    for (ipos, id) in enumerate(ids)
        @. cosx = cosx_2D[:,ipos]
        @. sinx = sinx_2D[:,ipos]
        @. cosy = cosy_2D[:,ipos]
        @. siny = siny_2D[:,ipos]
        for ivar in axes(CC, 1)
            @inbounds CC_loc = @view CC[ivar,:,:]
            @inbounds CS_loc = @view CS[ivar,:,:]
            @inbounds SC_loc = @view SC[ivar,:,:]
            @inbounds SS_loc = @view SS[ivar,:,:]
            average[ivar, id], gradx[ivar, id], grady[ivar, id], hessxx[ivar, id], hessxy[ivar, id], hessyy[ivar, id] =
                        standard_mean_one_position(mean_computer.lx, mean_computer.ly, cosx, sinx, cosy, siny, mean_computer.ap, mean_computer.aq, CC_loc, CS_loc, SC_loc, SS_loc)
        end
    end
    CC = 0
    CS = 0
    SC = 0
    SS = 0
    cosx_2D = 0
    sinx_2D = 0
    cosy_2D = 0
    siny_2D = 0
    cosx = 0
    sinx = 0
    cosy = 0
    siny = 0
    CC_loc = 0
    CS_loc = 0
    SC_loc = 0
    SS_loc = 0
end

"""

    standard_means!(mean_computer::MeanComputer, evaluating_on_particle_positions::Bool, variables::MeanVariable...)

Compute mean flow variables and their spatial derivatives through a kernel-means formalism. By default, the means are
estimated at every particle location, based on `mean_computer.part_pos`. If `is_on_grid == true`,
they are estimated at every node of a regular rectangular grid instead, based on `mean_computer.eval_pos`.
The means are stored in `variable.XXX_part` (if `is_on_grid == false`) or `variable.XXX_eval` (if `is_on_grid == true`) for every
`variable` in `variables`, where `XXX` refers to `"mean"`, `"gradx"`, `"grady"`, `"hessxx"`, `"hessxy"` and `"hessyy"` for the value
of the mean variable, its gradient, and its hessian matrix.

## Arguments:
* `mean_computer` is the [`MeanComputer`](@ref) wrapping all relevant information for the kernel-means algorithm
* `variables` is the set of [`MeanVariable`](@ref)s updated by the method
"""
function standard_means!(mean_computer::MeanComputer, variables::MeanVariable...; is_on_grid::Bool=false)
    is_on_grid ? positions = mean_computer.eval_pos : positions = mean_computer.part_pos
    map_cells_to_positions = positions_to_cells(positions, mean_computer.big_cells)
    nvar = length(variables)
    npos = positions.npos
    aver, gradx, grady, hessxx, hessxy, hessyy = zeros(nvar, npos), zeros(nvar, npos), zeros(nvar, npos), zeros(nvar, npos), zeros(nvar, npos), zeros(nvar, npos)
    for cell in mean_computer.small_cells
        compute_rsum!(cell, mean_computer, variables...)
    end
    for (icell, cell) in enumerate(mean_computer.big_cells)
        update_means_one_cell!(aver, gradx, grady, hessxx, hessxy, hessyy, mean_computer, cell, map_cells_to_positions[icell], is_on_grid=is_on_grid)
    end
    for (ivar, var) in enumerate(variables)
        if is_on_grid
            var.mean_eval .= @view aver[ivar,:]
            var.gradx_eval .= @view gradx[ivar,:]
            var.grady_eval .= @view grady[ivar,:]
            var.hessxx_eval .= @view hessxx[ivar,:]
            var.hessxy_eval .= @view hessxy[ivar,:]
            var.hessyy_eval .= @view hessyy[ivar,:]
        else
            var.mean_part .= @view aver[ivar,:]
            var.gradx_part .= @view gradx[ivar,:]
            var.grady_part .= @view grady[ivar,:]
            var.hessxx_part .= @view hessxx[ivar,:]
            var.hessxy_part .= @view hessxy[ivar,:]
            var.hessyy_part .= @view hessyy[ivar,:]
        end
    end
    aver = 0
    gradx = 0
    grady = 0
    hessxx = 0
    hessxy = 0
    hessyy = 0
end
