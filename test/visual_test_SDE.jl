include("../src/LagrangianStochasticConvection.jl")
using .LagrangianStochasticConvection
lsc = LagrangianStochasticConvection
using Statistics
using Plots

function init_particles(npart::Int64, X0::Float64)
    ux0 = X0
    omega0 = X0
    Lx = 1.0
    Ly = 1.0
    particles = []
    for i in 1:npart
        x = rand() * Lx
        y = rand() * Ly
        particle = lsc.Particle(i, 1.0, x, y, y-Ly, y+Ly, ux0, 0.0, 0.0, 0.0, omega0, 0.0, Dict())
        push!(particles, particle)
    end
    return particles
end

function dummy_coef(particle::lsc.Particle, kwparams::NamedTuple)
    return 0.0
end

function drift_OU(particle::lsc.Particle, kwparams::NamedTuple)
    return -kwparams.theta*(particle.ux - kwparams.mu)
end

function diffusion_OU(particle::lsc.Particle, kwparams::NamedTuple)
    return kwparams.sigma
end

function drift_GBM(particle::lsc.Particle, kwparams::NamedTuple)
    return kwparams.mu_GBM*particle.omega
end

function diffusion_GBM(particle::lsc.Particle, kwparams::NamedTuple)
    return kwparams.sigma_GBM*particle.omega
end

function extra_diffusion_GBM(particle::lsc.Particle, kwparams::NamedTuple)
    return 2.0*kwparams.sigma_GBM^2*particle.omega
end

function gen_realisations(npart::Int64, Nt::Int64, alpha::Float64, beta::Float64, dt::Float64, X0::Float64,
                          theta::Float64, mu::Float64, sigma::Float64, mu_GBM::Float64, sigma_GBM::Float64)
    particles = init_particles(npart, X0)
    simu_params = (alpha = alpha, beta = beta, dt = dt, Ly = 1.0, theta = theta,
                   mu = mu, sigma = sigma, mu_GBM = mu_GBM, sigma_GBM = sigma_GBM)
    time = [i*dt for i in 1:Nt]
    mean_ux, var_ux, mean_omega, var_omega = zeros(Nt), zeros(Nt), zeros(Nt), zeros(Nt)
    for it in eachindex(time)
        for particle in particles
            lsc.advance_particle!(particle, drift_OU, dummy_coef, dummy_coef, dummy_coef, drift_GBM,
                                  diffusion_OU, dummy_coef, dummy_coef, dummy_coef, diffusion_GBM,
                                  extra_diffusion_GBM, simu_params)
        end
        mean_ux[it] = mean([particle.ux for particle in particles])
        var_ux[it] = var([particle.ux for particle in particles])
        mean_omega[it] = mean([particle.omega for particle in particles])
        var_omega[it] = var([particle.omega for particle in particles])
    end
    return time, mean_ux, var_ux, mean_omega, var_omega
end

alpha = 0.5
beta = 0.5
dt = 0.001
Nt = 2000
npart = 100000

X0 = 10.0
theta = 0.1
mu = 3.0
sigma = 1.0
mu_GBM = -0.5
sigma_GBM = 0.2

temps, mean_ux, var_ux, mean_omega, var_omega =
        gen_realisations(npart, Nt, alpha, beta, dt, X0, theta, mu, sigma, mu_GBM, sigma_GBM)

mean_ux_target = @. X0*exp(-theta*temps) + mu*(1.0 - exp(-theta*temps))
var_ux_target = @. (sigma^2/(2.0*theta))*(1.0 - exp(-2.0*theta*temps))
mean_omega_target = @. X0*exp(mu_GBM*temps)
var_omega_target = @. X0^2*exp(2.0*mu_GBM*temps)*(exp(sigma_GBM^2*temps) - 1.0)

p1 = plot(temps, [mean_ux, mean_ux_target])
p2 = plot(temps, [var_ux, var_ux_target])
p3 = plot(temps, [mean_omega, mean_omega_target])
p4 = plot(temps, [var_omega, var_omega_target])

display(plot(p1))
display(plot(p2))
display(plot(p3))
display(plot(p4))