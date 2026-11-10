include("../src/LagrangianStochasticConvection.jl")
using .LagrangianStochasticConvection
lsc = LagrangianStochasticConvection
using BenchmarkTools
using ProfileView
using Plots
using Distributions
using Measures
using VoronoiCells
using GeometryBasics: Point2

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
    end
end

function init_particles(npart::Int64, Lx::Float64, Ly::Float64)
    x = rand(Float64, npart) * Lx
    y = rand(Float64, npart) * Ly
    ux = @. cos(8.0 * 2pi * y / Ly) * cos(8.0 * 2pi * x / Lx)
    uy = @. 10.0*x^2 + 4.0*x - 3.0
    ux .+= rand(Normal(0.0, 1.0), npart)
    uy .+= rand(Normal(0.0, 0.5), npart)
    particles = [lsc.Particle(i, 1.0, x[i], y[i], y[i] - Ly, y[i] + Ly, ux[i], uy[i], 0.0, 0.0, 0.0, 1.0, Dict()) for i in 1:npart]
    volumes!(particles, Lx, Ly)
    return particles
end

function maps(variable::lsc.MeanVariable, neval_x::Int64, neval_y::Int64)
    map = reshape(variable.mean_eval, (neval_x, neval_y))
    map_gradx = reshape(variable.gradx_eval, (neval_x, neval_y))
    map_grady = reshape(variable.grady_eval, (neval_x, neval_y))
    map_hessxx = reshape(variable.hessxx_eval, (neval_x, neval_y))
    map_hessxy = reshape(variable.hessxy_eval, (neval_x, neval_y))
    map_hessyy = reshape(variable.hessyy_eval, (neval_x, neval_y))
    return map, map_gradx, map_grady, map_hessxx, map_hessxy, map_hessyy
end

function plot_map_difference(map::Matrix{Float64}, target::Vector{Float64})
    neval_x, neval_y = size(map)
    map_target = reshape(target, (neval_x, neval_y))
    display(heatmap(map .- map_target, margin=10mm))
end

function plot_slice_difference(ix::Int64, iy::Int64, map::Matrix{Float64}, target::Vector{Float64})
    neval_x, neval_y = size(map)
    map_target = reshape(target, (neval_x, neval_y))
    p1 = plot(1:neval_y, [map[ix,:], map_target[ix,:]])
    p2 = plot(1:neval_x, [map[:,iy], map_target[:,iy]])
    # p1 = plot(1:neval_y, [mean(map, dims=1)[1,:], mean(map_target, dims=1)[1,:]])
    # p2 = plot(1:neval_x, [mean(map, dims=2)[:,1], mean(map_target, dims=2)[:,1]])
    display(plot(p1, p2, layout = 2))
end

function run_test()

    npart = 300000
    nmode = 7
    neval_x = 200
    neval_y = 200
    Nx = 40
    Ny = 40
    Lx = 1.0
    Ly = 1.0

    mean_computer = lsc.MeanComputer(nmode, npart, neval_x, neval_y, Nx, Ny, Lx, Ly, kernel_func)
    particles = init_particles(npart, Lx, Ly)
    rho = lsc.MeanVariable("density", npart, neval_x * neval_y, p::lsc.Particle -> p.mass)
    ux = lsc.MeanVariable("cos_y", npart, neval_x * neval_y, p::lsc.Particle -> p.mass * p.ux)
    uy = lsc.MeanVariable("lin_x", npart, neval_x * neval_y, p::lsc.Particle -> p.mass * p.uy)
    varuy = lsc.MeanVariable("var_lin_x", npart, neval_x * neval_y, p::lsc.Particle -> p.mass * (p.uy - p.means["lin_x_mean"])^2)
    w, x, y, xx, xy, yy = lsc.common_mean_variables(npart, neval_x * neval_y)

    lsc.set_particle_positions!(mean_computer, particles)
    lsc.standard_means!(mean_computer, w, x, y, xx, xy, yy, rho, ux, uy, is_on_grid=true)
    lsc.corrective_SPH!(mean_computer, w, x, y, xx, xy, yy, rho, ux, uy, is_on_grid=true)
    lsc.standard_means!(mean_computer, w, x, y, xx, xy, yy, rho, uy, is_on_grid=false)
    lsc.corrective_SPH!(mean_computer, w, x, y, xx, xy, yy, rho, uy, is_on_grid=false)
    lsc.divide_by_density!(rho, ux, uy, is_on_grid=true)
    lsc.divide_by_density!(rho, uy, is_on_grid=false)
    lsc.write_means_to_particles!(particles, uy)
    lsc.standard_means!(mean_computer, varuy, is_on_grid=true)
    lsc.corrective_SPH!(mean_computer, w, x, y, xx, xy, yy, varuy, is_on_grid=true)
    lsc.divide_by_density!(rho, varuy, is_on_grid=true)

    return neval_x, neval_y, mean_computer, ux, uy, varuy, Lx, Ly

end

function plot_all_results(mean_computer::lsc.MeanComputer, variable::lsc.MeanVariable, target_mean::Function,
                          target_gradx::Function, target_grady::Function, target_hessxx::Function,
                          target_hessxy::Function, target_hessyy::Function, neval_x::Int64, neval_y::Int64, Lx::Float64, Ly::Float64)
    map, map_gradx, map_grady, map_hessxx, map_hessxy, map_hessyy = maps(variable, neval_x, neval_y)
    xpoints = mean_computer.eval_pos.x
    ypoints = mean_computer.eval_pos.y
    ix = Int(floor(neval_x/3.25))
    iy = Int(floor(neval_y/3.25))
    plot_map_difference(map, target_mean(xpoints, ypoints, Lx, Ly))
    plot_slice_difference(ix, iy, map, target_mean(xpoints, ypoints, Lx, Ly))
    plot_map_difference(map_gradx, target_gradx(xpoints, ypoints, Lx, Ly))
    plot_slice_difference(ix, iy, map_gradx, target_gradx(xpoints, ypoints, Lx, Ly))
    plot_map_difference(map_grady, target_grady(xpoints, ypoints, Lx, Ly))
    plot_slice_difference(ix, iy, map_grady, target_grady(xpoints, ypoints, Lx, Ly))
    plot_map_difference(map_hessxx, target_hessxx(xpoints, ypoints, Lx, Ly))
    plot_slice_difference(ix, iy, map_hessxx, target_hessxx(xpoints, ypoints, Lx, Ly))
    plot_map_difference(map_hessxy, target_hessxy(xpoints, ypoints, Lx, Ly))
    plot_slice_difference(ix, iy, map_hessxy, target_hessxy(xpoints, ypoints, Lx, Ly))
    plot_map_difference(map_hessyy, target_hessyy(xpoints, ypoints, Lx, Ly))
    plot_slice_difference(ix, iy, map_hessyy, target_hessyy(xpoints, ypoints, Lx, Ly))
end

function plot_mean_results(mean_computer::lsc.MeanComputer, variable::lsc.MeanVariable, target_mean::Function,
                           neval_x::Int64, neval_y::Int64)
    map, _, _, _, _, _ = maps(variable, neval_x, neval_y)
    xpoints = mean_computer.eval_pos.x
    ypoints = mean_computer.eval_pos.y
    ix = Int(neval_x/2)
    iy = Int(neval_y/2)
    plot_map_difference(map, target_mean(xpoints, ypoints))
    plot_slice_difference(ix, iy, map, target_mean(xpoints, ypoints))
end

function visual_tests()
    nx, ny, mean_computer, ux, _, _, Lx, Ly = run_test()
    plot_all_results(mean_computer, ux, uxm, gradx_ux, grady_ux, hessxx_ux, hessxy_ux, hessyy_ux, nx, ny, Lx, Ly)
    # func_uym = (x, y) -> [10.0*elem^2 + 4.0*elem - 3.0 for elem in x]
    # func_gradx_uy = (x, y) -> [20.0*elem + 4.0 for elem in x]
    # func_hessxx_uy = (x, y) -> [20.0 for _ in x]
    # plot_all_results(mean_computer, variable_uy, func_uym, func_gradx_uy, func_zero, func_hessxx_uy, func_zero, func_zero, neval_x, neval_y)
    # func_varuym = (x, y) -> [0.25 for _ in x]
    # plot_mean_results(mean_computer, variable_varuy, func_varuym, neval_x, neval_y)
end

function uxm(x::Vector{Float64}, y::Vector{Float64}, Lx::Float64, Ly::Float64)
    return @. cos(8.0 * 2pi * x / Lx) * cos(8.0 * 2pi * y / Ly)
end

function gradx_ux(x::Vector{Float64}, y::Vector{Float64}, Lx::Float64, Ly::Float64)
    return @. -8.0*2pi/Lx * sin(8.0 * 2pi * x / Lx) * cos(8.0 * 2pi * y / Ly)
end

function grady_ux(x::Vector{Float64}, y::Vector{Float64}, Lx::Float64, Ly::Float64)
    return @. -8.0*2pi/Ly * cos(8.0 * 2pi * x / Lx) * sin(8.0 * 2pi * y / Ly)
end

function hessxx_ux(x::Vector{Float64}, y::Vector{Float64}, Lx::Float64, Ly::Float64)
    return @. -(8.0*2pi/Lx)^2 * cos(8.0 * 2pi * x / Lx) * cos(8.0 * 2pi * y / Ly)
end

function hessxy_ux(x::Vector{Float64}, y::Vector{Float64}, Lx::Float64, Ly::Float64)
    return @. (8.0*2pi/Lx)*(8.0*2pi/Ly) * sin(8.0 * 2pi * x / Lx) * sin(8.0 * 2pi * y / Ly)
end

function hessyy_ux(x::Vector{Float64}, y::Vector{Float64}, Lx::Float64, Ly::Float64)
    return @. -(8.0*2pi/Ly)^2 * cos(8.0 * 2pi * x / Lx) * cos(8.0 * 2pi * y / Ly)
end

@time visual_tests()