function kernel_func_accuracy(x::Vector{Float64}, h::Float64)
    return [kernel_func_accuracy(elem, h) for elem in x]
end

function kernel_func_accuracy(x::Float64, h::Float64)
    C = 5.0 / 4.0 / h
    return ifelse(abs(x) <= h, C * (1.0 + 3.0*abs(x/h)) * (1.0 - abs(x/h))^3, 0.0)
end

function kernel_func_approximate(x::Float64, coefs::Vector{Float64}, l::Float64)
    if abs(x) > l / 2.0
        return 0.0
    else
        Kx = 0.0
        for i in eachindex(coefs)
            Kx += coefs[i] * cos(2pi * (i-1) * x / l)
        end
        return Kx
    end
end

function init_particles_accuracy(npart::Int64, Lx::Float64, Ly::Float64)
    x = rand(Float64, npart) * Lx
    y = rand(Float64, npart) * Ly
    ux = zeros(npart)
    ux[1] = -10.0
    particles = [lsc.Particle(i, 1.0, x[i], y[i], y[i] + Ly, y[i] - Ly, ux[i], 0.0, 0.0, 0.0, 0.0, 0.0, Dict()) for i in 1:npart]
    return particles
end

npart = 500000
nmode = 7
neval_x = 9
neval_y = 7
Nx = 30
Ny = 60
Lx = 1.0
Ly = 4.0
hx = Lx/Nx
hy = Ly/Ny

mean_computer = lsc.MeanComputer(nmode, npart, neval_x, neval_y, Nx, Ny, Lx, Ly, kernel_func_accuracy)
particles = init_particles_accuracy(npart, Lx, Ly)
variable_ux = lsc.MeanVariable("ux", npart, neval_x * neval_y, p::lsc.Particle -> p.ux)
variable_w = lsc.MeanVariable("w", npart, neval_x * neval_y, p::lsc.Particle -> 1.0)
variable_x = lsc.MeanVariable("x", npart, neval_x * neval_y, p::lsc.Particle -> p.x)
variable_y = lsc.MeanVariable("y", npart, neval_x * neval_y, p::lsc.Particle -> p.y, p::lsc.Particle -> p.y_right, p::lsc.Particle -> p.y_left)
variable_xx = lsc.MeanVariable("xx", npart, neval_x * neval_y, p::lsc.Particle -> p.x^2)
variable_xy = lsc.MeanVariable("xy", npart, neval_x * neval_y, p::lsc.Particle -> p.x * p.y, p::lsc.Particle -> p.x * p.y_right, p::lsc.Particle -> p.x * p.y_left)
variable_yy = lsc.MeanVariable("yy", npart, neval_x * neval_y, p::lsc.Particle -> p.y^2, p::lsc.Particle -> p.y_right^2, p::lsc.Particle -> p.y_left^2)
variable_test_x = lsc.MeanVariable("test_x", npart, neval_x * neval_y, p::lsc.Particle -> 10.0*p.x + 4.0)
variable_test_y = lsc.MeanVariable("test_y", npart, neval_x * neval_y, p::lsc.Particle -> cos(2pi * p.y / Ly))

@testset "Mean flow: accuracy of standard SPH" begin

    @testset "One particle far away from boundaries" begin
        particles[1].x = Lx * 0.5032390
        particles[1].y = Ly * 0.491092380
        lsc.set_particle_positions!(mean_computer, particles)
        lsc.standard_means!(mean_computer, variable_ux, is_on_grid=true)
        for ipos in eachindex(mean_computer.eval_pos.x)
            xeval = mean_computer.eval_pos.x[ipos]
            yeval = mean_computer.eval_pos.y[ipos]
            Kx = kernel_func_approximate((xeval - particles[1].x), mean_computer.ap, mean_computer.lx)
            Ky = kernel_func_approximate((yeval - particles[1].y), mean_computer.aq, mean_computer.ly)
            if (Kx * Ky == 0.0) || (variable_ux.mean_eval[ipos] == 0.0)
                @test isapprox(variable_ux.mean_eval[ipos], -10.0 * Kx * Ky, atol=1e-6)
            else
                @test isapprox(variable_ux.mean_eval[ipos], -10.0 * Kx * Ky, rtol=1e-6)
            end
        end
    end

    @testset "One particle close to top or bottom boundary" begin
        particles[1].x = Lx - 0.2*hx
        particles[1].y = Ly * 0.491092380
        lsc.set_particle_positions!(mean_computer, particles)
        lsc.standard_means!(mean_computer, variable_ux, is_on_grid=true)
        for ipos in eachindex(mean_computer.eval_pos.x)
            xeval = mean_computer.eval_pos.x[ipos]
            yeval = mean_computer.eval_pos.y[ipos]
            Kx = kernel_func_approximate((xeval - particles[1].x), mean_computer.ap, mean_computer.lx)
            Ky = kernel_func_approximate((yeval - particles[1].y), mean_computer.aq, mean_computer.ly)
            if (Kx * Ky == 0.0) || (variable_ux.mean_eval[ipos] == 0.0)
                @test isapprox(variable_ux.mean_eval[ipos], -10.0 * Kx * Ky, atol=1e-6)
            else
                @test isapprox(variable_ux.mean_eval[ipos], -10.0 * Kx * Ky, rtol=1e-6)
            end
        end
    end

    @testset "One particle close to left boundary" begin
        particles[1].x = 0.491092380
        particles[1].y = 0.1 * hy
        lsc.set_particle_positions!(mean_computer, particles)
        lsc.standard_means!(mean_computer, variable_ux, is_on_grid=true)
        for ipos in eachindex(mean_computer.eval_pos.x)
            xeval = mean_computer.eval_pos.x[ipos]
            yeval = mean_computer.eval_pos.y[ipos]
            Kx = kernel_func_approximate((xeval - particles[1].x), mean_computer.ap, mean_computer.lx)
            Ky = kernel_func_approximate((yeval - particles[1].y), mean_computer.aq, mean_computer.ly)
            Ky += kernel_func_approximate((yeval - particles[1].y + Ly), mean_computer.aq, mean_computer.ly)
            Ky += kernel_func_approximate((yeval - particles[1].y - Ly), mean_computer.aq, mean_computer.ly)
            if (Kx * Ky == 0.0) || (variable_ux.mean_eval[ipos] == 0.0)
                @test isapprox(variable_ux.mean_eval[ipos], -10.0 * Kx * Ky, atol=1e-6)
            else
                @test isapprox(variable_ux.mean_eval[ipos], -10.0 * Kx * Ky, rtol=1e-6)
            end
        end
    end

    @testset "One particle close to right boundary" begin
        particles[1].x = 0.491092380
        particles[1].y = Ly - 0.05 * hy
        lsc.set_particle_positions!(mean_computer, particles)
        lsc.standard_means!(mean_computer, variable_ux, is_on_grid=true)
        for ipos in eachindex(mean_computer.eval_pos.x)
            xeval = mean_computer.eval_pos.x[ipos]
            yeval = mean_computer.eval_pos.y[ipos]
            Kx = kernel_func_approximate((xeval - particles[1].x), mean_computer.ap, mean_computer.lx)
            Ky = kernel_func_approximate((yeval - particles[1].y), mean_computer.aq, mean_computer.ly)
            Ky += kernel_func_approximate((yeval - particles[1].y + Ly), mean_computer.aq, mean_computer.ly)
            Ky += kernel_func_approximate((yeval - particles[1].y - Ly), mean_computer.aq, mean_computer.ly)
            if (Kx * Ky == 0.0) || (variable_ux.mean_eval[ipos] == 0.0)
                @test isapprox(variable_ux.mean_eval[ipos], -10.0 * Kx * Ky, atol=1e-6)
            else
                @test isapprox(variable_ux.mean_eval[ipos], -10.0 * Kx * Ky, rtol=1e-6)
            end
        end
    end

end

@testset "Mean flow: accuracy of corrective SPH scheme" begin

    @testset "Linear function of x" begin
        xpoints = mean_computer.eval_pos.x
        ypoints = mean_computer.eval_pos.y
        lsc.set_particle_positions!(mean_computer, particles)
        lsc.standard_means!(mean_computer, variable_w, variable_x, variable_y, variable_xx, variable_xy, variable_yy, variable_test_x, is_on_grid=true)
        lsc.corrective_SPH!(mean_computer, variable_w, variable_x, variable_y, variable_xx, variable_xy, variable_yy, variable_test_x, is_on_grid=true)
        @testset "Average" for i in eachindex(xpoints)
            @test isapprox(variable_test_x.mean_eval[i], 10.0 * xpoints[i] + 4.0, atol=1e-8)
        end
        @testset "Gradient" for i in eachindex(xpoints)
            @test isapprox(variable_test_x.gradx_eval[i], 10.0, atol=1e-8)
            @test isapprox(variable_test_x.grady_eval[i], 0.0, atol=1e-8)
        end
        @testset "Second derivatives" for i in eachindex(xpoints)
            @test isapprox(variable_test_x.hessxx_eval[i], 0.0, atol=1e-8)
            @test isapprox(variable_test_x.hessxy_eval[i], 0.0, atol=1e-8)
            @test isapprox(variable_test_x.hessyy_eval[i], 0.0, atol=1e-8)
        end
    end

    @testset "Cosine function of y" begin
        xpoints = mean_computer.eval_pos.x
        ypoints = mean_computer.eval_pos.y
        lsc.set_particle_positions!(mean_computer, particles)
        lsc.standard_means!(mean_computer, variable_w, variable_x, variable_y, variable_xx, variable_xy, variable_yy, variable_test_y, is_on_grid=true)
        lsc.corrective_SPH!(mean_computer, variable_w, variable_x, variable_y, variable_xx, variable_xy, variable_yy, variable_test_y, is_on_grid=true)
        @testset "Average" for i in eachindex(xpoints)
            @test isapprox(variable_test_y.mean_eval[i], cos(2pi * ypoints[i] / Ly), atol=1e-2)
        end
        @testset "Gradient" for i in eachindex(xpoints)
            @test isapprox(variable_test_y.gradx_eval[i], 0.0, atol=1e-2)
            @test isapprox(variable_test_y.grady_eval[i], -sin(2pi * ypoints[i] / Ly) * 2pi/Ly, atol=1e-2)
        end
        @testset "Second derivatives" for i in eachindex(xpoints)
            @test isapprox(variable_test_y.hessxx_eval[i], 0.0, atol=1e-2)
            @test isapprox(variable_test_y.hessxy_eval[i], 0.0, atol=1e-2)
            @test isapprox(variable_test_y.hessyy_eval[i], -cos(2pi * ypoints[i] / Ly) * (2pi/Ly)^2, atol=1e-2)
        end
    end

end

@testset "Mean flow: division by density" begin
    
    npart = 1000
    xpart = [particle.x for particle in particles[1:npart]]
    ypart = [particle.y for particle in particles[1:npart]]
    rho = lsc.MeanVariable("rho", npart, neval_x * neval_y, p::lsc.Particle -> p.mass)
    rho_u = lsc.MeanVariable("u", npart, neval_x * neval_y, p::lsc.Particle -> p.mass * p.ux)
    @. rho.mean_part = xpart^2 * ypart^3
    @. rho.gradx_part = 2.0 * xpart * ypart^3
    @. rho.grady_part = 3.0 * xpart^2 * ypart^2
    @. rho.hessxx_part = 2.0 * ypart^3
    @. rho.hessxy_part = 6.0 * xpart * ypart^2
    @. rho.hessyy_part = 6.0 * xpart^2 * ypart
    @. rho_u.mean_part = xpart^4 * ypart^6
    @. rho_u.gradx_part = 4.0 * xpart^3 * ypart^6
    @. rho_u.grady_part = 6.0 * xpart^4 * ypart^5
    @. rho_u.hessxx_part = 12.0 * xpart^2 * ypart^6
    @. rho_u.hessxy_part = 24.0 * xpart^3 * ypart^5
    @. rho_u.hessyy_part = 30.0 * xpart^4 * ypart^4
    lsc.divide_by_density!(rho, rho_u)
    @testset "Mean quantity" begin
        for i in eachindex(particles[1:npart])
            @test isapprox(rho_u.mean_part[i], xpart[i]^2 * ypart[i]^3, rtol=1e-8)
        end
    end
    @testset "Gradient x" begin
        for i in eachindex(particles[1:npart])
            @test isapprox(rho_u.gradx_part[i], 2.0 * xpart[i] * ypart[i]^3, rtol=1e-8)
        end
    end
    @testset "Gradient y" begin
        for i in eachindex(particles[1:npart])
            @test isapprox(rho_u.grady_part[i], 3.0 * xpart[i]^2 * ypart[i]^2, rtol=1e-8)
        end
    end
    @testset "Hessian xx" begin
        for i in eachindex(particles[1:npart])
            @test isapprox(rho_u.hessxx_part[i], 2.0 * ypart[i]^3, rtol=1e-8)
        end
    end
    @testset "Hessian xy" begin
        for i in eachindex(particles[1:npart])
            @test isapprox(rho_u.hessxy_part[i], 6.0 * xpart[i] * ypart[i]^2, rtol=1e-8)
        end
    end
    @testset "Hessian yy" begin
        for i in eachindex(particles[1:npart])
            @test isapprox(rho_u.hessyy_part[i], 6.0 * xpart[i]^2 * ypart[i], rtol=1e-8)
        end
    end

end
