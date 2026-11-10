#=
This file defines all methods related to fluid particle evolution
=#

#=
Dispatching mean variables to the particles where they were estimated
=#

"""

    write_one_mean_to_particles!(particles::Vector{Particle}, variable::MeanVariable, mean_type::String)

Transfer the value of a [`MeanVariable`](@ref) to each [`Particle`](@ref) location were it was estimated.

The function first reads either `variable.mean_part`, `variable.gradx_part`, `variable.grady_part` or
`variable.hessxx_part + variable.hessyy_part`, depending on whether `mean_type == "mean"`, `"gradx"`,
`"grady"` or `"laplacian"`. It then creates or updates the dictionnary `particle.means` for each `particle`
in `particles`, the updated key being composed of the `name` of the `variable`, followed by `mean_type`.
"""
function write_one_mean_to_particles!(particles::Vector{Particle}, variable::MeanVariable, mean_type::String)
    if mean_type == "mean"
        mean_to_write = variable.mean_part
    elseif mean_type == "gradx"
        mean_to_write = variable.gradx_part
    elseif mean_type == "grady"
        mean_to_write = variable.grady_part
    elseif mean_type == "laplacian"
        mean_to_write = variable.hessxx_part .+ variable.hessyy_part
    else
        throw(ArgumentError("mean_type argument not valid (got $(mean_type))"))
    end
    key = "$(variable.name)_$(mean_type)"
    for (ip, particle) in enumerate(particles)
        particle.means[key] = mean_to_write[ip]
    end
end

"""

    write_means_to_particles!(particles::Vector{Particle}, variables::MeanVariable...)

Transfer the value of each [`MeanVariable`](@ref) (and their spatial derivatives) to each [`Particle`](@ref) location were it was estimated.

For each `variable`, the function first reads `variable.mean_part`, `variable.gradx_part`, `variable.grady_part`
and `variable.hessxx_part + variable.hessyy_part`. It then creates or updates `particle.means["XXX_mean"]`,
`particle.means["XXX_gradx"]`, `particle.means["XXX_grady"]` and `particle.means["XXX_laplacian"]` for each `particle` in
`particles`, with `"XXX"` being the `name` of the `variable`.
"""
function write_means_to_particles!(particles::Vector{Particle}, variables::MeanVariable...)
    for variable in variables
        write_one_mean_to_particles!(particles, variable, "mean")
        write_one_mean_to_particles!(particles, variable, "gradx")
        write_one_mean_to_particles!(particles, variable, "grady")
        write_one_mean_to_particles!(particles, variable, "laplacian")
    end
end

#=
Advancing flow by one time step
=#

"""

    advance_particle!(particle::Particle, drift_velocity_x::Function, drift_velocity_y::Function,
                      drift_velocity_z::Function, drift_energy::Function, drift_frequency::Function,
                      diffusion_velocity_x::Function, diffusion_velocity_y::Function,
                      diffusion_velocity_z::Function, diffusion_energy::Function, diffusion_frequency::Function,
                      extra_diffusion_frequency::Function, kwparams::NamedTuple)

Evolves a [`Particle`](@ref) by one time step, according to a given stochastic model, using a Predictor/Corrector scheme

The functions `drift_velocity_x`, `drift_velocity_y`, `drift_velocity_z`, `drift_energy` and `drift_frequency` represent
the drift coefficients in the Lagrangian stochastic model. They must all take a [`Particle`](@ref) and a NamedTuple as argument,
and return a `Float`. The named tuple `kwparams` will be passed to each of them, and must contain all constant parameters
required for each of the coefficients of the stochastic model.

Likewise, `diffusion_velocity_x`, `diffusion_velocity_y`, `diffusion_velocity_z`, `diffusion_energy` and `diffusion_frequency`
represent the diffusion coefficients in the Lagrangian stochastic model.

The function `extra_diffusion_frequency` represent the derivative of the square of `diffusion_frequency`
with respect to ``\\omega^\\ast``.

In addition to the constant parameters of the stochastic model, `kwparams` must contain a field `"dt"` (representing the time step),
a field `"alpha"` (representing the implicitness parameter in the drift coefficients), a field `"beta"` (representing the
implicitness parameter in the diffusion coefficients), and a field `"Ly"` (representing the horizontal size of the domain).
"""
function advance_particle!(particle::Particle, drift_velocity_x::Function, drift_velocity_y::Function,
                           drift_velocity_z::Function, drift_energy::Function, drift_frequency::Function,
                           diffusion_velocity_x::Function, diffusion_velocity_y::Function,
                           diffusion_velocity_z::Function, diffusion_energy::Function, diffusion_frequency::Function,
                           extra_diffusion_frequency::Function, kwparams::NamedTuple)
    alpha = kwparams.alpha
    beta = kwparams.beta
    dt = kwparams.dt
    Ly = kwparams.Ly
    dW = rand(Normal(0.0, sqrt(dt)), 5)
    # Predictor step
    ux = particle.ux
    uy = particle.uy
    aux = drift_velocity_x(particle, kwparams)
    auy = drift_velocity_y(particle, kwparams)
    auz = drift_velocity_z(particle, kwparams)
    ae = drift_energy(particle, kwparams)
    aomega = drift_frequency(particle, kwparams)
    bux = diffusion_velocity_x(particle, kwparams)
    buy = diffusion_velocity_y(particle, kwparams)
    buz = diffusion_velocity_z(particle, kwparams)
    be = diffusion_energy(particle, kwparams)
    bomega = diffusion_frequency(particle, kwparams)
    dbomega = extra_diffusion_frequency(particle, kwparams)
    particle.x += ux*dt
    particle.y += uy*dt
    particle.y_left += uy*dt
    particle.y_right += uy*dt
    particle.ux += aux*dt + bux*dW[1]
    particle.uy += auy*dt + buy*dW[2]
    particle.uz += auz*dt + buz*dW[3]
    particle.e += ae*dt + be*dW[4]
    particle.omega += aomega*dt + bomega*dW[5]
    # Corrector step
    ux_corr = particle.ux
    uy_corr = particle.uy
    aux_corr = drift_velocity_x(particle, kwparams)
    auy_corr = drift_velocity_y(particle, kwparams)
    auz_corr = drift_velocity_z(particle, kwparams)
    ae_corr = drift_energy(particle, kwparams)
    aomega_corr = drift_frequency(particle, kwparams)
    bux_corr = diffusion_velocity_x(particle, kwparams)
    buy_corr = diffusion_velocity_y(particle, kwparams)
    buz_corr = diffusion_velocity_z(particle, kwparams)
    be_corr = diffusion_energy(particle, kwparams)
    bomega_corr = diffusion_frequency(particle, kwparams)
    dbomega_corr = extra_diffusion_frequency(particle, kwparams)
    ux_tot = alpha*ux + (1.0 - alpha)*ux_corr
    uy_tot = alpha*uy + (1.0 - alpha)*uy_corr
    aux_tot = alpha*aux + (1.0 - alpha)*aux_corr
    auy_tot = alpha*auy + (1.0 - alpha)*auy_corr
    auz_tot = alpha*auz + (1.0 - alpha)*auz_corr
    ae_tot = alpha*ae + (1.0 - alpha)*ae_corr
    aomega_tot = alpha*(aomega - beta*dbomega/2.0) + (1.0 - alpha)*(aomega_corr - beta*dbomega_corr/2.0)
    bux_tot = beta*bux + (1.0 - beta)*bux_corr
    buy_tot = beta*buy + (1.0 - beta)*buy_corr
    buz_tot = beta*buz + (1.0 - beta)*buz_corr
    be_tot = beta*be + (1.0 - beta)*be_corr
    bomega_tot = beta*bomega + (1.0 - beta)*bomega_corr
    # Updating particle
    particle.x += -ux*dt + ux_tot*dt
    particle.y += -uy*dt + uy_tot*dt
    particle.y_left = particle.y - Ly
    particle.y_right = particle.y + Ly
    particle.ux += -(aux*dt + bux*dW[1]) + aux_tot*dt + bux_tot*dW[1]
    particle.uy += -(auy*dt + buy*dW[2]) + auy_tot*dt + buy_tot*dW[2]
    particle.uz += -(auz*dt + buz*dW[3]) + auz_tot*dt + buz_tot*dW[3]
    particle.e += -(ae*dt + be*dW[4]) + ae_tot*dt + be_tot*dW[4]
    particle.omega += -(aomega*dt + bomega*dW[5]) + aomega_tot*dt + bomega_tot*dW[5]
end

"""

    apply_boundary_conditions!(particle::Particle, kwparams::NamedTuple)

Apply boundary conditions to a [`Particle`](@ref) if it has exited the domain through any of the four boundaries.

If the `particle` has exited through the left of right boundary, periodic BC are applied: the particle is reintroduced on the
other side of the domain, as if it was looping back on itself.

If the `particle` has exited thought the top or bottom boundary, the particle is reintroduced as its mirror image with respect to the
boundary, in the sense that its vertical position ``x^\\ast`` and velocity ``u_x^\\ast`` is be reflected. Its turbulent frequency
``\\omega^\\ast`` is left unchanged. The treatment of the other variables depends on `BC_type`:
* if `BC_type == "temp"`, its internal energy ``e^\\ast`` is set to a fixed value determined by `kwparams.etop` at the top boundary or
`kwparams.ebot` at the bottom boundary. Its horizontal position ``y^\\ast`` and velocity ``u_y^\\ast`` are left unchanged;
* if `BC_type == "flux"`, the same boundary conditions are applied, except for the bottom internal energy condition, where ``e^\\ast`` is
set to fixed profile
```
e^\\ast(x) = e_\\mathrm{bot} + \\nabla_e x
```
whenever the particle is within kernel size reach of the bottom boundary, and is left unchanged if the particle has exited through the top
boundary;
* if `BC_type == "no_slip"`, ``e^\\ast`` and ``y^\\ast`` is left unchanged at both boundaries, and ``u_y^\\ast`` is set to zero;
* if `BC_type == "none"`, ``e^\\ast``, ``y^\\ast`` and ``u_y^\\ast`` are all left unchanged;
* any other `BC_type` will result in an error.

`kwparams` must contain a field `"Lx"` and a field `"Ly"`, representing the dimensions of the domain, as well as
a field `"BC_type"` representing the type of boundary conditions (must be `"temp"`, `"flux"`, `"no_slip"` or `"none"`).
Furthermore, if `BC_type == "flux"`, it must contain a field `"hx"` representing the vertical kernel size, as well as fields
`"ebot"` and `"grad_e"` used in the bottom energy profile. If `BC_type == "temp"`, it must instead contain fields `"etop"` and `"ebot"`
used for the top and bottom internal energy boundary conditions.
"""
function apply_boundary_conditions!(particle::Particle, kwparams::NamedTuple)
    Lx = kwparams.Lx
    Ly = kwparams.Ly
    BC_type = kwparams.BC_type
    if BC_type == "temp"
        etop = kwparams.etop
        ebot = kwparams.ebot
    elseif BC_type == "flux"
        hx = kwparams.hx
        etop = kwparams.etop
        ebot = kwparams.ebot
        grad_e = kwparams.grad_e
    end

    if particle.y <= 0.0
        particle.y += Ly
        particle.y_left += Ly
        particle.y_right += Ly
    elseif particle.y >= Ly
        particle.y -= Ly
        particle.y_left -= Ly
        particle.y_right -= Ly
    end

    if particle.x <= 0.0
        particle.x = -particle.x
        particle.ux = -particle.ux
        if BC_type == "temp"
            particle.e = ebot
        elseif BC_type == "flux"
        elseif BC_type == "no_slip"
            particle.uy = 0.0
        elseif BC_type == "none"
        else
            error("Wrong boundary conditions specified (should be 'temp', 'flux', 'no_slip' or 'none')")
        end
    elseif particle.x >= Lx
        particle.x = 2.0*Lx - particle.x
        particle.ux = -particle.ux
        if BC_type == "temp"
            particle.e = etop
        elseif BC_type == "flux"
        elseif BC_type == "no_slip"
            particle.uy = 0.0
        elseif BC_type == "none"
        else
            error("Wrong boundary conditions specified (should be 'temp', 'flux', 'no_slip' or 'none')")
        end
    end

    if (BC_type == "flux") && ((particle.x <= hx) || (particle.x >= Lx - hx))
        particle.e = ebot + particle.x * grad_e
    end
end

"""

    number_bad_particles = advance_flow!(particles::Vector{Particle}, drift_velocity_x::Function, drift_velocity_y::Function,
                                         drift_velocity_z::Function, drift_energy::Function, drift_frequency::Function,
                                         diffusion_velocity_x::Function, diffusion_velocity_y::Function,
                                         diffusion_velocity_z::Function, diffusion_energy::Function, diffusion_frequency::Function,
                                         extra_diffusion_frequency::Function, kwparams::NamedTuple)

Evolves each [`Particle`](@ref) in `particles` by one time step, according to a given stochastic model,
using a Predictor/Corrector scheme. Also applies boundary conditions, checks Courant accuracy condition,
and compute the new lumped volume of every particle.

The functions `drift_velocity_x`, `drift_velocity_y`, `drift_velocity_z`, `drift_energy` and `drift_frequency` represent
the drift coefficients in the Lagrangian stochastic model. They must all take a [`Particle`](@ref) and a NamedTuple as argument,
and return a `Float`. The named tuple `kwparams` will be passed to each of them, and must contain all constant parameters
required for each of the coefficients of the stochastic model.

Likewise, `diffusion_velocity_x`, `diffusion_velocity_y`, `diffusion_velocity_z`, `diffusion_energy` and `diffusion_frequency`
represent the diffusion coefficients in the Lagrangian stochastic model.

The function `extra_diffusion_frequency` represent the derivative of the square of `diffusion_frequency`
with respect to ``\\omega^\\ast``.

In addition to the constant parameters of the stochastic model, `kwparams` must contain all the fields necessary for
the methods [`apply_boundary_conditions!`](@ref) and [`check_particle_sanity!`](@ref).

Return an integer `number_bad_particles` representing the number of particles that were deemed invalid after the time step
(see [`check_particle_sanity!`](@ref) for more info).
"""
function advance_flow!(particles::Vector{Particle}, drift_velocity_x::Function, drift_velocity_y::Function,
                       drift_velocity_z::Function, drift_energy::Function, drift_frequency::Function,
                       diffusion_velocity_x::Function, diffusion_velocity_y::Function,
                       diffusion_velocity_z::Function, diffusion_energy::Function, diffusion_frequency::Function,
                       extra_diffusion_frequency::Function, kwparams::NamedTuple)
    number_bad_particles = 0
    for particle in particles
        advance_particle!(particle, drift_velocity_x, drift_velocity_y,
                          drift_velocity_z, drift_energy, drift_frequency,
                          diffusion_velocity_x, diffusion_velocity_y,
                          diffusion_velocity_z, diffusion_energy, diffusion_frequency,
                          extra_diffusion_frequency, kwparams)
        apply_boundary_conditions!(particle, kwparams)
        is_bad = check_particle_sanity!(particle, kwparams)
        if is_bad
            number_bad_particles += 1
        end
    end
    return number_bad_particles
end

#=
Computing particle volume
=#

"""

    volume_correction(x::Float64, h::Float64)

Portion of the kernel integral inside the domain when the center is within distance `x` of the edge. If `x < h` (where `h` is the
kernel size), this is computed analytically for the following kernel function
```
K(r, h) = \\frac{5}{4h} \\left( 1 + 3 \\frac{x}{h} \\right) \\left( 1 - \\frac{x}{h} \\right)^3
```
It varies from `0.5` (when the center is right at the edge) to `1` (when `x = h`). If `x > h`, the portion of the kernel integral inside
the domain is always `1`.
"""
function volume_correction(x::Float64, h::Float64)
    xred = x/h
    v_inside = 0.5 + 1.25 * (xred - 2.0*xred^3 + 2.0*xred^4 - 0.6*xred^5)
    return ifelse(xred <= 1.0, v_inside, 1.0)
end

"""

    particle_volumes!(particles::Vector{Particle}, volume_computer::MeanComputer, rho_mean::MeanVariable, kwparams::NamedTuple)

Compute the volume of each [`Particle`](@ref) in `particles`, and update each `particle.volume` accordingly.

If the field `is_incomp` in `kwparams` is `true`, each particle will be assigned an equal volume, set to the total volume of the domain
divided by the number of particles. Otherwise, the volume of each particle is
```
\\mathcal{V} = \\frac{\\Delta m}{\\overline{\\rho}} ~,
```
where ``\\Delta m`` is the mass of the particle, and ``\\overline{\\rho}`` is the mean flow density at the particle location.

Arguments:
* `particles` is the set of [`Particle`](@ref)s whose volume is computed
* `volume_computer` is the [`MeanComputer`](@ref) used to compute ``\\overline{\\rho}``
* `rho_mean` is the [`MeanVariable`](@ref) representing the mean density
* `kwparams` must contain a field `"Lx"` representing the vertical size of the domain, as well as a Boolean field `"is_incomp"`. If the
latter is `true`, `kwparams` must also contain a field `"nparticles"` representing the number of particles, and a field `"Ly"` representing
the horizontal size of the domain.
"""
function particle_volumes!(particles::Vector{Particle}, volume_computer::MeanComputer, rho_mean::MeanVariable, kwparams::NamedTuple)
    is_incomp = kwparams.is_incomp
    Lx = kwparams.Lx
    hx = volume_computer.hx
    if is_incomp
        npart = kwparams.nparticles
        Ly = kwparams.Ly
    end

    if is_incomp
        for particle in particles
            particle.volume = Lx * Ly / npart
        end
    else
        set_particle_positions!(volume_computer, particles)
        standard_means!(volume_computer, rho_mean)
        write_means_to_particles!(particles, rho_mean)
        for particle in particles
            vcorr_bottom = volume_correction(particle.x, hx)
            vcorr_top = volume_correction(Lx - particle.x, hx)
            particle.volume = particle.mass / particle.means["$(rho_mean.name)_mean"] * vcorr_bottom * vcorr_top
        end
    end
end

#=
Checking sanity of particles
=#

"""

    is_bad = check_particle_energy!(particle::Particle)

Sets the energy ``e^\\ast`` of a [`Particle`](@ref) to the local mean if it is negative.
If the local mean energy is not defined, sets it to `1.0`.

Return a Boolean `is_bad` set to `true` if the particle energy had to be changed, `false` otherwise.
"""
function check_particle_energy!(particle::Particle)
    if particle.e < 0.0
        if "e_mean" in collect(keys(particle.means))
            particle.e = particle.means["e_mean"]
        else
            particle.e = 1.0
        end
        is_bad = true
    else
        is_bad = false
    end
    return is_bad
end

"""

    is_bad = check_particle_frequency!(particle::Particle)

Sets the turbulent frequency ``\\omega^\\ast`` of a [`Particle`](@ref) to the local mean if it is negative.
If the local mean turbulent frequency is not defined, sets it to 1.

Return a Boolean `is_bad` set to `true` if the particle turbulent frequency had to be changed, `false` otherwise.
"""
function check_particle_frequency!(particle::Particle)
    if particle.omega < 0.0
        if "omega_mean" in collect(keys(particle.means))
            particle.omega = particle.means["omega_mean"]
        else
            particle.omega = 1.0
        end
        is_bad = true
    else
        is_bad = false
    end
    return is_bad
end

"""

    C, flag = courant_number(particle::Particle, kwparams::NamedTuple)

Return the Courant number of a [`Particle`](@ref), as well as a flag indicating if the Courant number
is dominated by the turbulent frequency of the particle or its velocity.

`kwparams` must contain fields `"hx"` and `"hy"` (representing the size of the kernel function, see
[Kernel-mean algorithm](@ref) for more context), as well as a field `"dt"` representing the time
step of the simulation.

The particle Courant number is defined as
```math
    C = \\dfrac{1}{\\Delta t} \\mathrm{min} \\left( \\dfrac{1}{\\omega^\\ast}, \\dfrac{h_x}{|u_x^\\ast|}, \\dfrac{h_y}{|u_y^\\ast|})
```
where ``\\omega^\\ast``, ``u_x^\\ast`` and ``u_y^\\ast`` are the turbulent frequency, vertical velocity and horizontal
velocity of the particle, ``h_x`` and ``h_y`` are the vertical and horizontal size of the kernel function, and ``\\Delta t``
is the time step of the simulation. If the miminum defining ``C`` is the first value, then `flag = "omega"`. Otherwise,
`flag = "velocity"`.
"""
function courant_number(particle::Particle, kwparams::NamedTuple)
    hx = kwparams.hx
    hy = kwparams.hy
    courant_omega = 1.0/(abs(particle.omega)*kwparams.dt)
    courant_velocity = minimum((hx/abs(particle.ux), hy/abs(particle.uy)))/kwparams.dt
    if haskey(kwparams, :kappa)
        courant_radiative = (hx^2 + hy^2) / kwparams.kappa
    else
        courant_radiative = courant_omega + 1.0
    end
    if (courant_omega < courant_velocity) && (courant_omega < courant_radiative)
        courant = courant_omega
        flag = "omega"
    elseif (courant_velocity < courant_omega) && (courant_velocity < courant_radiative)
        courant = courant_velocity
        flag = "velocity"
    else
        courant = courant_radiative
        flag = "radiative"
    end
    return courant, flag
end

"""

    is_bad = check_particle_sanity!(particle::Particle, kwparams::NamedTuple)

Ensure that the properties of a [`Particle`](@ref) make sense (non negative energy and turbulent frequency),
and that the `particle` Courant number is larger than one. If the particle has a high turbulent frequency leading
to a Courant number smaller than 1, the turbulent frequency is reassigned to the local mean. If the particle has a
high velocity leading to a Courant number smaller than 1, the velocity is reassigned to the local mean.

Return a Boolean `is_bad` set to `true` if any of those problems occured, `false` otherwise.
"""
function check_particle_sanity!(particle::Particle, kwparams::NamedTuple)
    courant, flag = courant_number(particle, kwparams)
    if (courant < 1.0) && (flag == "omega")
        is_bad_courant = true
        if "omega_mean" in collect(keys(particle.means))
            particle.omega = particle.means["omega_mean"]
        else
            particle.omega = 1.0
        end
    elseif (courant < 1.0) && (flag == "velocity")
        is_bad_courant = true
        if "ux_mean" in collect(keys(particle.means))
            particle.ux = particle.means["ux_mean"]
        else
            particle.ux = 0.0
        end
        if "uy_mean" in collect(keys(particle.means))
            particle.uy = particle.means["uy_mean"]
        else
            particle.uy = 0.0
        end
    elseif (courant < 1.0) && (flag == "radiative")
        is_bad_courant = true
    else
        is_bad_courant = false
    end
    is_bad_energy = check_particle_energy!(particle)
    is_bad_frequency = check_particle_frequency!(particle)
    is_bad = is_bad_courant || is_bad_energy || is_bad_frequency
    return is_bad
end