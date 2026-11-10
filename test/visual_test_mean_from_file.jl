include("../src/LagrangianStochasticConvection.jl")
using .LagrangianStochasticConvection
lsc = LagrangianStochasticConvection
using Plots
using VoronoiCells
using GeometryBasics: Point2
using TOML
using DelimitedFiles

theme(:ggplot2)

function kernel_func(x::Vector{Float64}, h::Float64)
    return [kernel_func(elem, h) for elem in x]
end

function kernel_func(x::Float64, h::Float64)
    C = 5.0 / 4.0 / h
    return ifelse(abs(x) <= h, C * (1.0 + 3.0*abs(x/h)) * (1.0 - abs(x/h))^3, 0.0)
end

function volumes!(particles::Vector{lsc.Particle}, Lx::Float64, Ly::Float64)
    rect = Rectangle(Point2(0.0, 0.0), Point2(Lx, Ly))
    points = [Point2(particle.x, particle.y) for particle in particles]
    tess = voronoicells(points, rect)
    deltav = voronoiarea(tess)
    for (i, particle) in enumerate(particles)
        particle.volume = deltav[i]
        # particle.volume = 1.0 / particle.means["rho_mean"]
    end
end

function init_particles(simu_path::String)
    config = TOML.parsefile(joinpath(simu_path, "config.toml"))
    Lx = 1.0
    Ly = config["spatial"]["aspect_ratio"]
    path_to_file = joinpath(simu_path, "raw_output", "final_state.dat")
    particle_properties = readdlm(path_to_file, '\t', Float64, '\n', skipstart=1)
    npart = size(particle_properties, 1)
    particles::Vector{lsc.Particle} = []
    for id in 1:npart
        x = particle_properties[id,1]
        y = particle_properties[id,2]
        ux = particle_properties[id,3]
        uy = particle_properties[id,4]
        uz = particle_properties[id,5]
        e = particle_properties[id,6]
        omega = particle_properties[id,7]
        particle = lsc.Particle(id, 1.0, x, y, y-Ly, y+Ly, ux, uy, uz, e, omega, 1.0, Dict())
        push!(particles, particle)
    end
    volumes!(particles, Lx, Ly)
    return particles, Lx, Ly, npart
end

function maps(variable::lsc.MeanVariable, neval_x::Int64, neval_y::Int64)
    map = reshape(variable.mean_eval, (neval_x, neval_y))
    map_gradx = reshape(variable.gradx_eval, (neval_x, neval_y))
    map_grady = reshape(variable.grady_eval, (neval_x, neval_y))
    return map, map_gradx, map_grady
end

function plot_output(xpart::Vector{Float64}, ypart::Vector{Float64}, xeval::Vector{Float64}, yeval::Vector{Float64},
                     var_grid::Matrix{Float64}, var_part::Vector{Float64})
    
    # display(heatmap(var_grid, margin=10mm, color=:oslo))
    # display(scatter(ypart[1:100000], xpart[1:100000]; zcolor=var_part[1:100000], margin=10mm, markersize=1.0, color=:oslo))
    
    ix = 100
    slice_var_grid = var_grid[ix,:]
    slice_mask = (xpart .>= xeval[ix-1]) .&& (xpart .<= xeval[ix+1])
    slice_var_part = var_part[slice_mask]
    slice_ypart = ypart[slice_mask]

    p1 = scatter(slice_ypart, slice_var_part; margin=10mm, markersize=3.0, color=:oslo)
    plot!(p1, yeval, slice_var_grid)
    display(p1)

    iy = 25
    slice_var_grid = var_grid[:,iy]
    slice_mask = (ypart .>= yeval[iy-1]) .&& (ypart .<= yeval[iy+1])
    slice_var_part = var_part[slice_mask]
    slice_xpart = xpart[slice_mask]

    p2 = scatter(slice_xpart, slice_var_part; margin=10mm, markersize=3.0, color=:oslo)
    plot!(p2, xeval, slice_var_grid)
    display(p2)

end

function run_test()

    nmode = 7
    neval_x = 500
    neval_y = 500
    Nx = 200
    Ny = 200

    particles, Lx, Ly, npart = init_particles("/Users/jphilidet/sph/benchmark/SLM/dt3e-3_N3e5_n40")
    mean_computer = lsc.MeanComputer(nmode, npart, neval_x, neval_y, Nx, Ny, Lx, Ly, kernel_func)
    
    rho = lsc.MeanVariable("rho", npart, neval_x * neval_y, p::lsc.Particle -> p.mass)
    ux = lsc.MeanVariable("ux", npart, neval_x * neval_y, p::lsc.Particle -> p.mass * p.ux)
    w, x, y, xx, xy, yy = lsc.common_mean_variables(npart, neval_x * neval_y)

    lsc.set_particle_positions!(mean_computer, particles)
    lsc.standard_means!(mean_computer, w, x, y, xx, xy, yy, rho, ux, is_on_grid=true)
    lsc.corrective_SPH!(mean_computer, w, x, y, xx, xy, yy, rho, ux, is_on_grid=true)
    lsc.divide_by_density!(rho, ux, is_on_grid=true)
    lsc.standard_means!(mean_computer, w, x, y, xx, xy, yy, rho, ux, is_on_grid=false)
    lsc.corrective_SPH!(mean_computer, w, x, y, xx, xy, yy, rho, ux, is_on_grid=false)
    lsc.divide_by_density!(rho, ux, is_on_grid=false)
    lsc.write_means_to_particles!(particles, rho, ux)

    xpart = [p.x for p in particles]
    ypart = [p.y for p in particles]
    xeval = [Lx*i/(neval_x+1) for i in 1:neval_x]
    yeval = [Ly*j/(neval_y+1) for j in 1:neval_y]
    rho_grid, drho_dx_grid, drho_dy_grid = maps(rho, neval_x, neval_y)
    ux_grid, dux_dx_grid, dux_dy_grid = maps(ux, neval_x, neval_y)
    rho_part = [p.means["rho_mean"] for p in particles]
    drho_dx_part = [p.means["rho_gradx"] for p in particles]
    drho_dy_part = [p.means["rho_grady"] for p in particles]
    ux_part = [p.ux for p in particles]
    dux_dx_part = [p.means["ux_gradx"] for p in particles]
    dux_dy_part = [p.means["ux_grady"] for p in particles]

    plot_output(xpart, ypart, xeval, yeval, rho_grid, rho_part)
    plot_output(xpart, ypart, xeval, yeval, drho_dx_grid, drho_dx_part)
    plot_output(xpart, ypart, xeval, yeval, drho_dy_grid, drho_dy_part)

    plot_output(xpart, ypart, xeval, yeval, ux_grid, ux_part)
    plot_output(xpart, ypart, xeval, yeval, dux_dx_grid, dux_dx_part)
    plot_output(xpart, ypart, xeval, yeval, dux_dy_grid, dux_dy_part)

end

run_test()