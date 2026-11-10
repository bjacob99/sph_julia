particle = lsc.Particle(1, 1.0)
particle_2 = lsc.Particle(2, 1.0)
particles = [particle, particle_2]
small_cell = lsc.SmallCell(10, 0.0, 0.0, 1.0, 2.0, 0.0)
small_cell_2 = lsc.SmallCell(10, 0.0, 0.0, 1.0, 2.0, -1.0, particles)
big_cell = lsc.BigCell(0.0, 0.0, 1.0, 2.0)
big_cell_2 = lsc.BigCell(0.0, 0.0, 1.0, 2.0, [small_cell, small_cell_2])
mean_computer = lsc.MeanComputer(10, 20, 30, 1.0, 1.0, x -> x, [small_cell, small_cell_2], [big_cell, big_cell_2])
positions = lsc.Positions(3, 10)

@testset "Display and constructors" begin

    @testset "Base.show" begin
        @test repr(particle) == "Particle #1: Position = (0.0, 0.0), velocity = (0.0, 0.0, 0.0), energy = 0.0, frequency = 0.0"
        @test repr(small_cell) == "Small cell between x = 0.0 and 1.0, y = 0.0 and 2.0, containing 0 particles. Horizontal shift when filtering particles = 0.0"
        @test repr(big_cell) == "Big cell between x = 0.0 and 1.0, y = 0.0 and 2.0, containing 0 small cells."
    end

    @testset "Constructors" begin
        @testset "Positions" begin
            @test positions.nmodes == 3
            @test positions.npos == 10
            @test positions.x == zeros(10)
            @test positions.y == zeros(10)
            for i in 0:2
                @test i in collect(keys(positions.sine_x))
                @test i in collect(keys(positions.cosine_x))
                @test i in collect(keys(positions.sine_y))
                @test i in collect(keys(positions.cosine_y))
            end
        end
        @testset "Particle" begin
            @test particle.id == 1
            @test particle.mass == 1.0
            @test particle.x == 0.0
            @test particle.y == 0.0
            @test particle.y_right == 0.0
            @test particle.y_left == 0.0
            @test particle.ux == 0.0
            @test particle.uy == 0.0
            @test particle.uz == 0.0
            @test particle.e == 0.0
            @test particle.omega == 0.0
            @test particle.volume == 0.0
            @test particle.means == Dict()
        end
        @testset "SmallCell" begin
            @test small_cell.nmodes == 10
            @test small_cell.x0 == 0.0
            @test small_cell.y0 == 0.0
            @test small_cell.x1 == 1.0
            @test small_cell.y1 == 2.0
            @test small_cell.increment_y == 0.0
            @test small_cell.particles == []
            @test small_cell.rsum_CC == zeros(1, 10, 10)
            @test small_cell.rsum_SC == zeros(1, 10, 10)
            @test small_cell.rsum_CS == zeros(1, 10, 10)
            @test small_cell.rsum_SS == zeros(1, 10, 10)
            @test small_cell_2.particles == [particle, particle_2]
            @test small_cell_2.increment_y == -1.0
        end
        @testset "BigCell" begin
            @test big_cell.x0 == 0.0
            @test big_cell.y0 == 0.0
            @test big_cell.x1 == 1.0
            @test big_cell.y1 == 2.0
            @test big_cell.smallcells == []
            @test big_cell_2.smallcells == [small_cell, small_cell_2]
        end
        @testset "MeanComputer" begin
            @test mean_computer.nmodes == 10
            @test mean_computer.hx == 1.0
            @test mean_computer.hy == 1.0
            @test mean_computer.lx == 3.0
            @test mean_computer.ly == 3.0
            @test mean_computer.part_pos.nmodes == 10
            @test mean_computer.part_pos.npos == 20
            @test mean_computer.eval_pos.nmodes == 10
            @test mean_computer.eval_pos.npos == 30
            @test mean_computer.ap == zeros(10)
            @test mean_computer.aq == zeros(10)
            @test mean_computer.small_cells == [small_cell, small_cell_2]
            @test mean_computer.big_cells == [big_cell, big_cell_2]
        end
    end

end