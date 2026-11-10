using LagrangianStochasticConvection
using Plots
using Distributions
using Measures
using VoronoiCells
using GeometryBasics: Point2
theme(:ggplot2)

"""
Mean variable profiles
"""

function mass(L::Float64, Q::Int64, rho_type::String)
    if rho_type == "uniform"
        int_rho = 2.0*L
    elseif rho_type == "linear"
        int_rho = 2.0*L + 1.5*L^2
    elseif rho_type == "quadratic"
        int_rho = 2.0*L + 1.5*L^2 + L^3/3.0
    elseif rho_type == "exponential"
        int_rho = 4.0*(1.0 - exp(-L/2.0))
    elseif rho_type == "localised"
        int_rho = 1.0
    elseif rho_type == "step"
        int_rho = 2.0 * 0.5 + 1.0 * 0.5
    end
    mass = L*int_rho/Q^2
    return mass
end

function rho_profile(rho_type::String)
    if rho_type == "uniform"
        rho_func = x -> 2.0
    elseif rho_type == "linear"
        rho_func = x -> 2.0 + 3.0*x
    elseif rho_type == "quadratic"
        rho_func = x -> 2.0 + 3.0*x + x^2
    elseif rho_type == "exponential"
        rho_func = x -> 2.0*exp(-x/2.0)
    elseif rho_type == "localised"
        rho_func = x -> exp(-(x-0.5)^2/(2*0.1^2)) / sqrt(2*pi*0.1^2)
    elseif rho_type == "step"
        rho_func = x -> ifelse(x < 0.5, 2.0, 1.0)
    end
    return rho_func
end

function grad_rho_profile(rho_type::String)
    if rho_type == "uniform"
        rho_func = x -> 0.0
    elseif rho_type == "linear"
        rho_func = x -> 3.0
    elseif rho_type == "quadratic"
        rho_func = x -> 3.0 + 2.0*x
    elseif rho_type == "exponential"
        rho_func = x -> -exp(-x/2.0)
    elseif rho_type == "localised"
        rho_func = x -> -(x-0.5)/(0.1^2)*exp(-(x-0.5)^2/(2*0.1^2)) / sqrt(2*pi*0.1^2)
    elseif rho_type == "step"
        rho_func = x -> ifelse(abs(x - 0.5) < 0.005, -1/0.01, 0.0)
    end
    return rho_func
end

function rho_max(rho_type::String, L::Float64)
    if rho_type == "uniform"
        rho_max = 2.0
    elseif rho_type == "linear"
        rho_max = 2.0 + 3.0*L
    elseif rho_type == "quadratic"
        rho_max = 2.0 + 3.0*L + L^2
    elseif rho_type == "exponential"
        rho_max = 2.0
    elseif rho_type == "localised"
        rho_max = 1.0 / sqrt(2*pi*0.1^2)
    elseif rho_type == "step"
        rho_max = 2.0
    end
    return rho_max
end

function ux(x::Float64, y::Float64)
    return cos(y * 2pi) / cos(x)
end

function dux_dx(x::Float64, y::Float64)
    return cos(y * 2pi) * sin(x) / (cos(x))^2
end

function dux_dy(x::Float64, y::Float64)
    return -2pi * sin(y * 2pi) / cos(x)
end

"""
Creating particles
"""

function row_positions(m::Float64, d::Float64, Q::Int64, rho_func::Function)
    # Row positions from bottom
    xrow_bottom::Vector{Float64} = zeros(Q)
    rho_bottom = rho_func(0.0)
    xrow_bottom[1] = 0.5 * m/(d*rho_bottom)
    for irow in 2:Q
        rho = rho_func(xrow_bottom[irow-1])
        xrow_bottom[irow] = xrow_bottom[irow-1] + m/(d*rho)
    end
    # Row positions from top
    xrow_top::Vector{Float64} = zeros(Q)
    rho_top =  rho_func(d*Q)
    xrow_top[Q] = L - 0.5 * m/(d*rho_top)
    for irow in Q-1:-1:1
        rho = rho_func(xrow_top[irow+1])
        xrow_top[irow] = xrow_top[irow+1] - m/(d*rho)
    end
    # Combining
    xrow = (xrow_bottom .+ xrow_top) / 2.0
    return xrow
end

function col_positions(d::Float64, Q::Int64)
    ycol = [d/2 + d*k for k in 0:Q-1]
    return ycol
end

function draw_position(rho_func::Function, rho_m::Float64, L::Float64)
    not_found = true
    while not_found
        global x_rand = rand() * L
        density_rand = rand() * rho_m
        if density_rand <= rho_func(x_rand)
            not_found = false
        end
    end
    y_rand = rand() * L
    return x_rand, y_rand
end

function volumes!(particles::Vector{Particle}, L::Float64)
    rect = Rectangle(Point2(0.0, 0.0), Point2(L, L))
    points = [Point2(particle.x, particle.y) for particle in particles]
    tess = voronoicells(points, rect)
    deltav = voronoiarea(tess)
    for (i, particle) in enumerate(particles)
        particle.volume = deltav[i]
    end
end

function init_particles(Q::Int64, L::Float64, rho_type::String, is_random::Bool)
    particles::Vector{Particle} = []
    rho_func = rho_profile(rho_type)
    rho_m = rho_max(rho_type, L)
    m = mass(L, Q, rho_type)
    d = L/Q
    xrow = row_positions(m, d, Q, rho_func)
    ycol = col_positions(d, Q)
    for ipart in 1:Q^2
        if is_random
            x, y = draw_position(rho_func, rho_m, L)
        else
            j, k = divrem(ipart-1, Q)
            x = xrow[j+1]
            y = ycol[k+1]
        end
        velocity = ux(x, y) + rand(Normal(0.0, 0.1))
        particle = Particle(ipart, m, x, y, y - L, y + L, velocity, 0.0, 0.0, 0.0, 0.0, 1.0, Dict())
        push!(particles, particle)
    end
    volumes!(particles, L)
    return particles
end

"""
Computing mean density and ux from particles
"""

function kernel_lucy(x::Vector{Float64}, h::Float64)
    return @. ifelse(abs(x) <= h, 5.0/4.0/h * (1.0 + 3.0*abs(x/h)) * (1.0 - abs(x/h))^3, 0.0)
end

function kernel_gaussian(x::Vector{Float64}, h::Float64)
    hred = h / 3.0
    return @. exp(-(x/hred)^2)/(sqrt(pi)*hred)
end

function kernel_uniform(x::Vector{Float64}, h::Float64)
    return @. ifelse(abs(x) <= h, 1.0/(2.0*h), 0.0)
end

function kernel_spline(x::Vector{Float64}, h::Float64)
    R = @. abs(2.0*x/h)
    temp = @. ifelse(R <= 1.0, 2.0/3.0 - R^2 + 0.5*R^3, (2.0-R)^3/6.0)
    return @. ifelse(R <= 2.0, temp / h, 0.0)
end

function kernel_morris(x::Vector{Float64}, h::Float64)
    R = @. abs(2.5*x/h)
    temp = @. ifelse(R <= 0.5, (R+2.5)^4 - 5*(R+1.5)^4 + 10*(R+0.5)^4, (2.5-R)^4 - 5*(1.5-R)^4)
    temp2 = @. ifelse(R <= 1.5, temp, (2.5-R)^4)
    return @. ifelse(R <= 2.5, temp2 / (24*h), 0.0)
end

function kernel_johnson(x::Vector{Float64}, h::Float64)
    R = @. abs(2.0*x/h)
    return @. ifelse(R <= 1.0, 3*R^2/16 - 3*R/4 + 0.75, 0.0)
end

function mean_standard(particles::Vector{Particle}, Q::Int64, L::Float64,
                       Neval::Int64, Ncells::Int64, Nmode::Int64, kernel_func::Function)
    rho_mean = MeanVariable("density", Q^2, Neval^2, p::Particle -> p.mass)
    ux_mean = MeanVariable("ux", Q^2, Neval^2, p::Particle -> p.mass * p.ux)
    w, x, y, xx, xy, yy = common_mean_variables(Q^2, Neval^2)
    mean_computer = MeanComputer(Nmode, Q^2, Neval, Neval, Ncells, Ncells, L, L, kernel_func)
    set_particle_positions!(mean_computer, particles)
    standard_means!(mean_computer, w, x, y, xx, xy, yy, rho_mean, ux_mean, is_on_grid=true)
    corrective_SPH!(mean_computer, w, x, y, xx, xy, yy, rho_mean, ux_mean, is_on_grid=true)
    divide_by_density!(rho_mean, ux_mean, is_on_grid=true)
    return rho_mean, ux_mean
end

function volume_correction(x::Float64, h::Float64, kernel_func::Function)
    xred = x/h
    if kernel_func == kernel_uniform
        v_inside = 0.5 * (1.0 + xred)
        volume_correction = ifelse(xred <= 1.0, v_inside, 1.0)
    elseif kernel_func == kernel_lucy
        v_inside = 0.5 + 1.25 * (xred - 2.0*xred^3 + 2.0*xred^4 - 0.6*xred^5)
        volume_correction = ifelse(xred <= 1.0, v_inside, 1.0)
    else
        volume_correction = 1.0
    end
    return volume_correction
end

function mean_smooth(particles::Vector{Particle}, Q::Int64, L::Float64,
                     Neval::Int64, Ncells::Int64, Nmode::Int64, kernel_func::Function)
    # Initialising mean computer and variables
    Nmodes_density = 7
    Ncells_density = 10
    kernel_volume = kernel_lucy
    rho_mean = MeanVariable("density", Q^2, Neval^2, p::Particle -> p.mass)
    ux_mean = MeanVariable("ux", Q^2, Neval^2, p::Particle -> p.mass * p.ux)
    w, x, y, xx, xy, yy = common_mean_variables(Q^2, Neval^2)
    mean_computer = MeanComputer(Nmode, Q^2, Neval, Neval, Ncells, Ncells, L, L, kernel_func)
    mean_computer_density = MeanComputer(Nmodes_density, Q^2, 1, 1, Ncells_density, Ncells_density, L, L, kernel_volume)
    set_particle_positions!(mean_computer, particles)
    set_particle_positions!(mean_computer_density, particles)
    # Estimating particle volume
    standard_means!(mean_computer_density, rho_mean)
    write_means_to_particles!(particles, rho_mean)
    for particle in particles
        vcorr_bottom = volume_correction(particle.x, L/Ncells_density, kernel_volume)
        vcorr_top = volume_correction(L - particle.x, L/Ncells_density, kernel_volume)
        particle.volume = particle.mass / particle.means["density_mean"] * vcorr_bottom * vcorr_top
    end
    # Estimating mean variables
    standard_means!(mean_computer, w, x, y, xx, xy, yy, rho_mean, ux_mean, is_on_grid=true)
    corrective_SPH!(mean_computer, w, x, y, xx, xy, yy, rho_mean, ux_mean, is_on_grid=true)
    divide_by_density!(rho_mean, ux_mean, is_on_grid=true)
    return rho_mean, ux_mean
end


"""
Plots
"""

function scatter_particles(particles::Vector{Particle})
    npart = min(50000, length(particles))
    xpart = [p.x for p in particles[1:npart]]
    ypart = [p.y for p in particles[1:npart]]
    display(scatter(ypart, xpart, color=:oslo, markersize=1.0))
end

function extract_map(variable::MeanVariable, kind::String, Neval::Int64)
    if kind == "mean"
        return reshape(variable.mean_eval, (Neval, Neval))
    elseif kind == "gradx"
        return reshape(variable.gradx_eval, (Neval, Neval))
    elseif kind == "grady"
        return reshape(variable.grady_eval, (Neval, Neval))
    elseif kind == "hessxx"
        return reshape(variable.hessxx_eval, (Neval, Neval))
    elseif kind == "hessxy"
        return reshape(variable.hessxy_eval, (Neval, Neval))
    elseif kind == "hessyy"
        return reshape(variable.hessyy_eval, (Neval, Neval))
    end
end

function plot_map(map::Matrix{Float64})
    display(heatmap(map, margin=10mm, color=:oslo))
end

function plot_rho(map::Matrix{Float64}, rho_type::String, L::Float64)
    # x array
    Neval = size(map, 1)
    x = [i*L/Neval for i in 1:Neval]
    # y array
    islice = div(Neval, 2)
    slice_var = map[:,islice]
    # Expected y array
    rho_func = rho_profile(rho_type)
    exp_var = rho_func.(x)
    # Plotting
    p = plot(x, slice_var)
    plot!(p, x, exp_var)
    display(p)
end

function plot_drho_dx(map::Matrix{Float64}, rho_type::String, L::Float64)
    # x array
    Neval = size(map, 1)
    x = [i*L/Neval for i in 1:Neval]
    # y array
    islice = div(Neval, 2)
    slice_var = map[:,islice]
    # Expected y array
    rho_func = grad_rho_profile(rho_type)
    exp_var = rho_func.(x)
    # Plotting
    p = plot(x, slice_var)
    plot!(p, x, exp_var)
    display(p)
end

function plot_ux(map::Matrix{Float64}, L::Float64)
    # x array
    Neval = size(map, 1)
    x = [i*L/Neval for i in 1:Neval]
    # y array
    islice = div(Neval, 2)
    slice_var = map[:,islice]
    y = x[islice]
    # Expected y array
    exp_var = [ux(xloc, y) for xloc in x]
    # Plotting
    p = plot(x, slice_var)
    plot!(p, x, exp_var)
    display(p)
end

function plot_dux_dx(map::Matrix{Float64}, L::Float64)
    # x array
    Neval = size(map, 1)
    x = [i*L/Neval for i in 1:Neval]
    # y array
    islice = div(Neval, 2)
    slice_var = map[:,islice]
    y = x[islice]
    # Expected y array
    exp_var = [dux_dx(xloc, y) for xloc in x]
    # Plotting
    p = plot(x, slice_var)
    plot!(p, x, exp_var)
    display(p)
end

function plot_dux_dy(map::Matrix{Float64}, L::Float64)
    # x array
    Neval = size(map, 1)
    x = [i*L/Neval for i in 1:Neval]
    # y array
    islice = div(Neval, 2)
    slice_var = map[:,islice]
    y = x[islice]
    # Expected y array
    exp_var = [dux_dy(xloc, y) for xloc in x]
    # Plotting
    p = plot(x, slice_var)
    plot!(p, x, exp_var)
    display(p)
end

function plot_all(rho::MeanVariable, ux::MeanVariable, particles::Vector{Particle},
                  Neval::Int64, L::Float64, rho_type::String)
    map_rho = extract_map(rho, "mean", Neval)
    map_drho_dx = extract_map(rho, "gradx", Neval)
    map_ux = extract_map(ux, "mean", Neval)
    map_dux_dx = extract_map(ux, "gradx", Neval)
    map_dux_dy = extract_map(ux, "grady", Neval)
    # scatter_particles(particles)
    plot_rho(map_rho, rho_type, L)
    plot_drho_dx(map_drho_dx, rho_type, L)
    plot_ux(map_ux, L)
    plot_dux_dx(map_dux_dx, L)
    plot_dux_dy(map_dux_dy, L)
    plot_map(map_rho)
    plot_map(map_drho_dx)
end

"""
Main calls
"""

Q = 600
L = 1.0
Neval = 500
Ncell = 60
Nmode = 7
rho_type = "step"
is_random = true
kernel_func = kernel_lucy

particles = init_particles(Q, L, rho_type, is_random)
rho_standard, ux_standard = mean_standard(particles, Q, L, Neval, Ncell, Nmode, kernel_func)
rho_smooth, ux_smooth = mean_smooth(particles, Q, L, Neval, Ncell, Nmode, kernel_func)

plot_all(rho_standard, ux_standard, particles, Neval, L, rho_type)
plot_all(rho_smooth, ux_smooth, particles, Neval, L, rho_type)