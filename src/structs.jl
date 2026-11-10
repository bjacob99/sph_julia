#=
This file defines all the custom structures used by the other parts of the code.
=#

"""

    Positions(nmodes::Int64, npos::Int64)

Construct a set of `npos` points in 2D, designed to store sine and cosine functions of their Cartesian coordinates
for `nmodes` difference spatial frequencies.

The fast kernel-means algorithm described in [Kernel-mean algorithm](@ref) involves 
```
s_p = \\sin \\left( \\dfrac{2 \\pi x p}{l_x} \\right)~, c_p = \\cos \\left( \\dfrac{2 \\pi x p}{l_x} \\right) \\\\
\\vspace{1cm} \\\\
s_q = \\sin \\left( \\dfrac{2 \\pi y q}{l_y} \\right)~, c_q = \\cos \\left( \\dfrac{2 \\pi y q}{l_y} \\right) \\\\
```
for each particle coordinate, for `nmodes` values of ``p`` and ``q``, and ``l_x`` and ``l_y``
are length scales related to the size of the kernel function. The attributes `Positions.sine_x`, `Positions.cosine_x`,
`Positions.sine_y`, and `Positions.cosine_y` are dictionaries relating each mode `p` or `q` to the vector
of ``s_p``, ``s_q``, ``c_p``and ``c_q`` for each point of `Position`.
"""
mutable struct Positions
    const nmodes::Int64
    const npos::Int64
    x::Vector{Float64}
    y::Vector{Float64}
    sine_x::Dict{Int64,Vector{Float64}}
    cosine_x::Dict{Int64,Vector{Float64}}
    sine_y::Dict{Int64,Vector{Float64}}
    cosine_y::Dict{Int64,Vector{Float64}}
end

function Positions(nmodes, npos)
	Positions(nmodes, npos, zeros(npos), zeros(npos),
			  Dict((i, zeros(npos)) for i in 0:nmodes-1), Dict((i, zeros(npos)) for i in 0:nmodes-1),
			  Dict((i, zeros(npos)) for i in 0:nmodes-1), Dict((i, zeros(npos)) for i in 0:nmodes-1)
			  )
end

"""

    MeanVariable(name::String, nparticles::Int64, neval::Int64, func::Function)
    MeanVariable(name::String, func::Function, kwparams::NamedTuple)

Construct a mean flow quantity from a given recipe `func`, to be estimated by the kernel-means algorithm

The mean variable ``\\overline{Q}`` is defined by the particle property ``Q^\\ast`` in the kernel-means formalism
(see [Kernel-mean algorithm](@ref)). The attribute `func` gives the recipe to build this particle property:
it must be a function which takes a [`Particle`](@ref) as argument, and returns a `Float`.

The `name` of the mean variable serves as unique identifier, and should hint at the physical nature of the mean
(for example `"velocity"`).

`nparticles` (resp. `neval`) is the number of particle locations (resp. fixed regular grid locations) at which the mean flow
variable is expected to be estimated. A `NamedTuple` can be passed instead, provided in contains a field `"nparticles`", 
a field `"neval_x"` and a field `"neval_y"` (the latter two should correspond to the number of rows and columns in the grid,
the product of which corresponds to `neval`).

The values and derivatives of the mean variable estimated at the particle locations are stored as `Vector`s of `Float` in
`MeanVariable.mean_part`, `MeanVariable.gradx_part`, `MeanVariable.grady_part`, `MeanVariable.hessxx_part`, `MeanVariable.hessxy_part`
and `MeanVariable.hessyy_part`. Those estimated on the fixed regular grid are stored in `MeanVariable.mean_eval`, `MeanVariable.gradx_eval`,
`MeanVariable.grady_eval`, `MeanVariable.hessxx_eval`, `MeanVariable.hessxy_eval` and `MeanVariable.hessyy_eval`.
"""
mutable struct MeanVariable
    const name::String
    const npart::Int64
    const neval::Int64
    const func::Function
    const func_right::Function
    const func_left::Function
    mean_part::Vector{Float64}
    gradx_part::Vector{Float64}
    grady_part::Vector{Float64}
    hessxx_part::Vector{Float64}
    hessxy_part::Vector{Float64}
    hessyy_part::Vector{Float64}
    mean_eval::Vector{Float64}
    gradx_eval::Vector{Float64}
    grady_eval::Vector{Float64}
    hessxx_eval::Vector{Float64}
    hessxy_eval::Vector{Float64}
    hessyy_eval::Vector{Float64}
end

MeanVariable(name, nparticles, neval, f1, f2, f3) = MeanVariable(name, nparticles, neval, f1, f2, f3, zeros(nparticles), zeros(nparticles),
																 zeros(nparticles), zeros(nparticles), zeros(nparticles), zeros(nparticles),
                                                            	 zeros(neval), zeros(neval), zeros(neval), zeros(neval), zeros(neval), zeros(neval))
MeanVariable(name, nparticles, neval, f) = MeanVariable(name, nparticles, neval, f, f, f)
MeanVariable(name, f1, f2, f3, kwparams) = MeanVariable(name, kwparams.nparticles, kwparams.neval_x * kwparams.neval_y, f1, f2, f3)
MeanVariable(name, f, kwparams) = MeanVariable(name, f, f, f, kwparams)

"""
    Particle(id::Int64, mass::Float64)
    Particle(id::Int64, mass::Float64, x::Float64, y::Float64, ux::Float64, uy::Float64, uz::Float64, e::Float64, omega::Float64)

Construct a notional particle identified by an integer `id` and a (constant) `mass`. Its initial position, velocity, energy and
turbulent frequency can also be passed to the constructor (if not provided, default to 0.0).

The physical properties of the particle are its 2D coordinates `Particle.x` and `Particle.y`, its 3D velocity `Particle.ux`,
`Particle.uy` and `Particle.uz`, its specific internal energy `Particle.e`, its turbulent frequency `Particle.omega` and the
lumped volume of fluid it represents `Particle.volume`.

In addition, the properties of the mean flow at the particle positions are stored in a dictionnary `Particle.means`, where the
keys are in the form `"XXX_YYY"`, where `"XXX"` is the `name` of the [`MeanVariable`](@ref), and `"YYY"` is either `"mean"`,
`"gradx"`, `"grady"` or `"laplacian"`.
"""
mutable struct Particle
    const id::Int64
    const mass::Float64
    x::Float64
    y::Float64
    y_left::Float64
    y_right::Float64
    ux::Float64
    uy::Float64
    uz::Float64
    e::Float64
    omega::Float64
    volume::Float64
    means::Dict{String,Float64}
end

function Base.show(io::IO, p::Particle)
    print(io, "Particle #$(p.id): Position = ($(p.x), $(p.y)), velocity = ($(p.ux), $(p.uy), $(p.uz)), energy = $(p.e), frequency = $(p.omega)")
end

Particle(id, mass, x, y, ux, uy, uz, e, omega) = Particle(id, mass, x, y, y, y, ux, uy, uz, e, omega, 0.0, Dict())
Particle(id, mass) = Particle(id, mass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

"""

	SmallCell(nmodes::Int64, x0::Float64, y0::Float64, x1::Float64, y1::Float64, increment_y::Float64, particles::Vector{Particle})
	SmallCell(nmodes::Int64, x0::Float64, y0::Float64, x1::Float64, y1::Float64, increment_y::Float64)

Construct a ``\\mathcal{R}`` cell (see [Efficient algorithm](@ref) for its definition and its use).

The portion of domain covered by the cell is defined by the 2D coordinates of its bottom-left corner `(x0, y0)` and its
top-right corner `(x1, y1)`, and *R sums* will be computed for ``\\mathrm{nmodes}^2`` Fourier components.

The list of [`Particle`](@ref)s contained in the cell is meant to be stored in `SmallCell.particles`, and the different
*R sums* in `rsums_CC`, `rsums_SC`, `rsums_CS` and `rsums_SS` as 3D arrays (first index refers to different [`MeanVariable`](@ref),
and the other two indices to ``p`` and ``q``).

With periodic boundary conditions in ``y``, the kernel-means algorithm requires two rows of ``\\mathcal{R}`` cells on either side of
the domain, on the outside. To define which particles are inside, their horizontal position is shifted by `increment_y` beforehand.
"""
mutable struct SmallCell
    const nmodes::Int64
    const x0::Float64
    const y0::Float64
    const x1::Float64
    const y1::Float64
    const increment_y::Float64
    particles::Vector{Particle}
    rsum_CC::Array{Float64,3}
    rsum_CS::Array{Float64,3}
    rsum_SC::Array{Float64,3}
    rsum_SS::Array{Float64,3}
end

function SmallCell(nmodes, x0, y0, x1, y1, increment_y, p)
	return SmallCell(nmodes, x0, y0, x1, y1, increment_y, p,
					 zeros(1, nmodes, nmodes), zeros(1, nmodes, nmodes),
					 zeros(1, nmodes, nmodes), zeros(1, nmodes, nmodes)
					 )
end

function SmallCell(nmodes, x0, y0, x1, y1, increment_y)
	return SmallCell(nmodes, x0, y0, x1, y1, increment_y, [])
end

function Base.show(io::IO, cell::SmallCell)
    nparticle = length(cell.particles)
    print(io, "Small cell between x = $(cell.x0) and $(cell.x1), y = $(cell.y0) and $(cell.y1), containing $(nparticle) particles. Horizontal shift when filtering particles = $(cell.increment_y)")
end

"""

	BigCell(x0::Float64, y0::Float64, x1::Float64, y1::Float64)

Construct a ``\\mathcal{S}`` cell (see [Efficient algorithm](@ref) for its definition and its use).

The portion of domain covered by the cell is defined by the 2D coordinates of its bottom-left corner `(x0, y0)` and its
top-right corner `(x1, y1)`. The list of [`SmallCell`](@ref)s contained in the cell is meant to be stored in `BigCell.smallcells`.
"""
mutable struct BigCell
    const x0::Float64
    const y0::Float64
    const x1::Float64
    const y1::Float64
    smallcells::Vector{SmallCell}
end

BigCell(x0, y0, x1, y1) = BigCell(x0, y0, x1, y1, [])

function Base.show(io::IO, c::BigCell)
    ncell = length(c.smallcells)
    print(io, "Big cell between x = $(c.x0) and $(c.x1), y = $(c.y0) and $(c.y1), containing $(ncell) small cells.")
end

"""

	MeanComputer(nmodes::Int64, nparticles::Int64, neval_x::Int64, neval_y::Int64,
				 ncellx::Int64, ncells_y::Int64, Lx::Float64, Ly::Float64, kernel_func::Function)
	MeanComputer(kernel_func::Function, kwparams::NamedTuple)

Construct a wrapper with all necessary information to feed the kernel-means algorithm.

The kernel function `kernel_func` used in the algorithm should take two `Float`s as argument (distance where the function
must be estimated, and size of compact support), and return a `Float`. It will be expanded onto `nmodes` Fourier components.
Means will be estimated at `nparticles` particle locations, or alternatively at the nodes of a fixed, regular 2D grid with
`neval_x` rows and `neval_y` columns. The full domain in which the particles are expected to live has dimensions `(Lx, Ly)`.
The kernel-means algorithm requires the domain be paved by a regular 2D grid of rectangular cells. The size of the grid is
`(ncells_x, ncells_y)`.

Instead of being passed one by one to the constructor, the parameters can be embedded in a single named tuple `kwparams`,
provided it has all the required field `"nmodes"`, `"nparticles"`, `"neval_x"`, `"neval_y"`, `"ncells_x"`, `"ncells_y"`,
"`Lx"` and `"Ly"`.

In addition, the size of the compact support of the kernel function is stored as `(MeanComputer.hx, MeanComputer.hy)`. By
construction, they give the dimensions of the ``\\mathcal{R}`` cells, and they equal `Lx/ncells_x` and `Ly/ncells_y`.
The dimensions of the ``\\mathcal{S}`` cells are also stored in `(MeanComputer.lx, MeanComputer.ly)`, and equal ``3 h_x``
and ``3 h_y`` by default.

The 2D coordinates of all the [`Particle`](@ref)s representing the flow are meant to be stored
as a [`Positions`](@ref) in `MeanComputer.part_pos`, and likewise the nodes of the fixed, regular 2D grid is stored in
`MeanComputer.eval_pos`.

The Fourier coefficients of the kernel function are meant to be stored in `MeanComputer.ap` and `MeanComputer.aq` as vectors
of size `nmodes` (respectively for the horizontal and vertical parts), and the full list of [`SmallCell`](@ref)s and
[`BigCell`](@ref)s used to pave the domain should be stored in `MeanComputer.smallcells` and `MeanComputer.bigcells`
"""
mutable struct MeanComputer
    const nmodes::Int64
    const hx::Float64
    const hy::Float64
    const lx::Float64
    const ly::Float64
    const kernel_function::Function
    part_pos::Positions
    eval_pos::Positions
    ap::Vector{Float64}
    aq::Vector{Float64}
    small_cells::Vector{SmallCell}
    big_cells::Vector{BigCell}
end

function MeanComputer(nmodes, nparticles, neval, hx, hy, kernel_function, small_cells, big_cells)
	return MeanComputer(nmodes, hx, hy, 3.0*hx, 3.0*hy, kernel_function,
						Positions(nmodes, nparticles), Positions(nmodes, neval),
						zeros(nmodes), zeros(nmodes), small_cells, big_cells
						)
end

function MeanComputer(nmodes::Int64, nparticles::Int64, neval_x::Int64, neval_y::Int64,
                      Nx::Int64, Ny::Int64, Lx::Float64, Ly::Float64, kernel_func::Function)
	hx = Lx / Nx
	hy = Ly / Ny
	rcells, scells = init_cells(nmodes, hx, hy, Nx, Ny, Ly)
	xeval, yeval = init_evaluating_points(neval_x, neval_y, Lx, Ly)
	mean_computer = MeanComputer(nmodes, nparticles, neval_x * neval_y, hx, hy, kernel_func, rcells, scells)
	for cell in scells
		set_small_cells!(rcells, cell)
	end
	set_eval_positions!(mean_computer, xeval, yeval)
	set_kernel_coefs!(mean_computer)
	return mean_computer
end

MeanComputer(kernel_func, kwparams) = MeanComputer(kwparams.nmodes, kwparams.nparticles, kwparams.neval_x, kwparams.neval_y,
												   kwparams.ncells_x, kwparams.ncells_y, kwparams.Lx, kwparams.Ly, kernel_func)