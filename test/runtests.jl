import LagrangianStochasticConvection as lsc
using Test
using Trapz
using Statistics
using Distributions
using Plots

include("./tests_structs.jl")
include("./tests_basic_mean_flow.jl")
include("./tests_accuracy_mean_flow.jl")
include("./tests_particles.jl")