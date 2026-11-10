function kernel_func_basic(x::Vector{Float64}, h::Float64)
    return [kernel_func_basic(elem, h) for elem in x]
end

function kernel_func_basic(x::Float64, h::Float64)
    C = 5.0 / 4.0 / h
    return ifelse(abs(x) <= h, C * (1.0 + 3.0*abs(x/h)) * (1.0 - abs(x/h))^3, 0.0)
end

function init_variables(npart::Int64, neval::Int64)
    variable_ux = lsc.MeanVariable("ux", npart, neval,
                                   p::lsc.Particle -> p.ux)
    variable_e = lsc.MeanVariable("e", npart, neval,
                                  p::lsc.Particle -> p.e,
                                  p::lsc.Particle -> p.e - 0.5,
                                  p::lsc.Particle -> p.e + 0.5)
    return [variable_ux, variable_e]
end

function init_particles_basic(npart::Int64, Lx::Float64, Ly::Float64)
    x = rand(Float64, npart) * Lx
    y = rand(Float64, npart) * Ly
    ux = zeros(npart)
    e = zeros(npart)
    ux[1] = 1.0
    e[100] = -10.0
    particles = [lsc.Particle(i, 1.0, x[i], y[i], y[i] - Ly, y[i] + Ly, ux[i], 0.0, 0.0, e[i], 0.0, 0.0, Dict()) for i in 1:npart]
    return x, y, particles
end

npart = 123
nmode = 13
neval_x = 90
neval_y = 78
Nx = 10
Ny = 17
Lx = 0.54
Ly = 13.4
hx = Lx / Nx
hy = Ly / Ny

vars = init_variables(npart, neval_x * neval_y)
xpart, ypart, particles = init_particles_basic(npart, Lx, Ly)
rcells, scells = lsc.init_cells(nmode, hx, hy, Nx, Ny, Ly)
xeval, yeval = lsc.init_evaluating_points(neval_x, neval_y, Lx, Ly)
mean_computer = lsc.MeanComputer(nmode, npart, neval_x * neval_y, hx, hy, kernel_func_basic, rcells, scells)
for cell in scells
    lsc.set_small_cells!(rcells, cell)
end
lsc.set_particle_positions!(mean_computer, particles)
lsc.set_eval_positions!(mean_computer, xeval, yeval)
lsc.set_kernel_coefs!(mean_computer)

@testset "Mean flow: basic subfunctions" begin

    @testset "Cells update" begin
        total_particles = 0
        for cell in rcells
            if cell.increment_y == 0.0
                total_particles += length(cell.particles)
            end
        end
        @test total_particles == npart
        @testset "Small cells" for cell in rcells
            if length(cell.particles) != 0
                @test minimum([p.x for p in cell.particles]) >= cell.x0
                @test maximum([p.x for p in cell.particles]) <= cell.x1
                @test minimum([p.y for p in cell.particles]) >= cell.y0 + cell.increment_y
                @test maximum([p.y for p in cell.particles]) <= cell.y1 + cell.increment_y
            end
        end
        @testset "Big cells" for cell in scells
            @test minimum([c.x0 for c in cell.smallcells]) >= cell.x0
            @test maximum([c.x1 for c in cell.smallcells]) <= cell.x1
            @test minimum([c.y0 for c in cell.smallcells]) >= cell.y0
            @test maximum([c.y1 for c in cell.smallcells]) <= cell.y1
        end
    end

    @testset "Mean Computer update" begin
        @testset "Trigonometric coefficients for particles" for i in 0:nmode-1
            @test mean_computer.part_pos.sine_x[i] == @. sin(2pi * i * xpart / (3.0*hx))
            @test mean_computer.part_pos.cosine_x[i] == @. cos(2pi * i * xpart / (3.0*hx))
            @test mean_computer.part_pos.sine_y[i] == @. sin(2pi * i * ypart / (3.0*hy))
            @test mean_computer.part_pos.cosine_y[i] == @. cos(2pi * i * ypart / (3.0*hy))
        end
        @testset "Trigonometric coefficients for evaluation points" for i in 0:nmode-1
            @test mean_computer.eval_pos.sine_x[i] == @. sin(2pi * i * xeval / (3.0*hx))
            @test mean_computer.eval_pos.cosine_x[i] == @. cos(2pi * i * xeval / (3.0*hx))
            @test mean_computer.eval_pos.sine_y[i] == @. sin(2pi * i * yeval / (3.0*hy))
            @test mean_computer.eval_pos.cosine_y[i] == @. cos(2pi * i * yeval / (3.0*hy))
        end
        @testset "Kernel coefficients" begin
            x_kernel_test = collect(LinRange(-3.0*hx/2.0, 3.0*hx/2.0, 1000))
            y_kernel_test = collect(LinRange(-3.0*hy/2.0, 3.0*hy/2.0, 1000))
            Kx = zeros(1000)
            Ky = zeros(1000)
            for i in 0:nmode-1
                @. Kx += mean_computer.ap[i+1] * cos(2pi * i * x_kernel_test / (3.0*hx))
                @. Ky += mean_computer.aq[i+1] * cos(2pi * i * y_kernel_test / (3.0*hy))
            end
            relative_error_x = trapz(x_kernel_test, abs.(Kx .- kernel_func_basic(x_kernel_test, hx)))
            relative_error_y = trapz(y_kernel_test, abs.(Ky .- kernel_func_basic(y_kernel_test, hy)))
            @test isapprox(relative_error_x, 0.0, atol=1e-3)
            @test isapprox(relative_error_y, 0.0, atol=1e-3)
        end
    end

    @testset "Sum over Small Cell" begin
        @testset "Filtering positions without increment" begin
            ids = rand(1:npart, 10)
            new_pos = lsc.filter_positions(mean_computer.part_pos, ids, 0.0, 3.0*hy)
            @test new_pos.nmodes == nmode
            @test new_pos.npos == 10
            @test new_pos.x == [particles[id].x for id in ids]
            @test new_pos.y == [particles[id].y for id in ids]
            for i in 0:nmode-1
                @test new_pos.cosine_x[i] == [cos(2pi * i * particles[id].x / (3.0*hx)) for id in ids]
                @test new_pos.sine_x[i] == [sin(2pi * i * particles[id].x / (3.0*hx)) for id in ids]
                @test new_pos.cosine_y[i] == [cos(2pi * i * particles[id].y / (3.0*hy)) for id in ids]
                @test new_pos.sine_y[i] == [sin(2pi * i * particles[id].y / (3.0*hy)) for id in ids]
            end
        end
        @testset "Filtering positions with increment" begin
            ids = rand(1:npart, 10)
            new_pos = lsc.filter_positions(mean_computer.part_pos, ids, 0.1, 3.0*hy)
            @test new_pos.nmodes == nmode
            @test new_pos.npos == 10
            @test new_pos.x == [particles[id].x for id in ids]
            @test new_pos.y == [particles[id].y - 0.1 for id in ids]
            for i in 0:nmode-1
                @test new_pos.cosine_x[i] == [cos(2pi * i * particles[id].x / (3.0*hx)) for id in ids]
                @test new_pos.sine_x[i] == [sin(2pi * i * particles[id].x / (3.0*hx)) for id in ids]
                @test isapprox(new_pos.cosine_y[i], [cos(2pi * i * (particles[id].y - 0.1) / (3.0*hy)) for id in ids], atol=1e-10)
                @test isapprox(new_pos.sine_y[i], [sin(2pi * i * (particles[id].y - 0.1) / (3.0*hy)) for id in ids], atol=1e-10)
            end
        end
        @testset "Computing R sums" for cell in rcells
            lsc.compute_rsum!(cell, mean_computer, vars...)
            i = rand(0:nmode-1)
            j = rand(0:nmode-1)
            target_CC = p -> cos(2pi * i * p.x / (3.0*hx)) * cos(2pi * j * (p.y - cell.increment_y) / (3.0*hy))
            target_SC = p -> sin(2pi * i * p.x / (3.0*hx)) * cos(2pi * j * (p.y - cell.increment_y) / (3.0*hy))
            target_CS = p -> cos(2pi * i * p.x / (3.0*hx)) * sin(2pi * j * (p.y - cell.increment_y) / (3.0*hy))
            target_SS = p -> sin(2pi * i * p.x / (3.0*hx)) * sin(2pi * j * (p.y - cell.increment_y) / (3.0*hy))
            if 1 in [p.id for p in cell.particles]
                @test isapprox(cell.rsum_CC[1, i+1, j+1], target_CC(particles[1]), atol=1e-10)
                @test isapprox(cell.rsum_SC[1, i+1, j+1], target_SC(particles[1]), atol=1e-10)
                @test isapprox(cell.rsum_CS[1, i+1, j+1], target_CS(particles[1]), atol=1e-10)
                @test isapprox(cell.rsum_SS[1, i+1, j+1], target_SS(particles[1]), atol=1e-10)
            else
                @test cell.rsum_CC[1, i+1, j+1] == 0.0
                @test cell.rsum_SC[1, i+1, j+1] == 0.0
                @test cell.rsum_CS[1, i+1, j+1] == 0.0
                @test cell.rsum_SS[1, i+1, j+1] == 0.0
            end
        end
    end

    @testset "Sum over Big Cell" begin
        for small_cell in rcells
            small_cell.rsum_CC = ones(2, 3, 3)
            small_cell.rsum_CS = ones(2, 3, 3) * 2.0
            small_cell.rsum_SC = ones(2, 3, 3) * 3.0
            small_cell.rsum_SS = ones(2, 3, 3) * -10.0
        end
        for cell in scells
            sum1, sum2, sum3, sum4 = lsc.compute_ssum(cell)
            @test sum1 == ones(2, 3, 3) .* length(cell.smallcells)
            @test sum2 == ones(2, 3, 3) .* length(cell.smallcells) * 2.0
            @test sum3 == ones(2, 3, 3) .* length(cell.smallcells) * 3.0
            @test sum4 == ones(2, 3, 3) .* length(cell.smallcells) * -10.0
        end
    end

    @testset "Map positions to cells" begin
        positions = mean_computer.part_pos
        positions_to_cell = lsc.positions_to_cells(positions, mean_computer.small_cells)
        for icell in eachindex(mean_computer.small_cells)
            cell = mean_computer.small_cells[icell]
            for ipos in positions_to_cell[icell]
                @test particles[ipos] in cell.particles
            end
            @test sum(length(value) for value in collect(values(positions_to_cell))) == npart
        end
    end

end
