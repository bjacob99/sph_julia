# Benchmark spectral : référence séquentielle vs deux boucles de cellules multithread.
# julia --project=. --threads=4 --gcthreads=1 test/benchmark_spectral_threads.jl scripts/config_general.toml [graine]
# --gcthreads=1 est disponible sous Julia 1.10 : fixer ce paramètre pour la comparaison.
# Répéter dans des processus distincts avec --threads=1,2,4,8 selon la machine.
# Pré-requis : standard_means_reference! et standard_means! du fichier fourni.
# Ce test mesure standard_means! entier, y compris affectation cible,
# allocations et copies finales, sans CSPM, intégration ni écritures de simulation.
# Les sources restent figées ; standard_means! remplace intégralement ses sorties.
# Les recettes de variables doivent être en lecture seule.

import LagrangianStochasticConvection as lsc
using LagrangianStochasticConvection
using BenchmarkTools
using Test
using Random
using Statistics
using TOML
using Printf
using LinearAlgebra

const SAMPLES = 15
const SECONDS = 30.0
const CHECK_REPEATS = 3

function fields_spectral(v, grid)
    grid ? (v.mean_eval, v.gradx_eval, v.grady_eval,
            v.hessxx_eval, v.hessxy_eval, v.hessyy_eval) :
           (v.mean_part, v.gradx_part, v.grady_part,
            v.hessxx_part, v.hessxy_part, v.hessyy_part)
end

function serial!(mc, vars, grid)
    lsc.standard_means_reference!(mc, vars...; is_on_grid=grid)
    return nothing
end
function threaded!(mc, vars, grid)
    lsc.standard_means!(mc, vars...; is_on_grid=grid)
    return nothing
end

function verify_spectral!(mc, vars, grid, label)
    serial!(mc, vars, grid)
    expected = map(v -> map(copy, fields_spectral(v, grid)), vars)
    @testset "$label : égalité séquentiel / threads" begin
        for repetition in 1:CHECK_REPEATS
            threaded!(mc, vars, grid)
            for (ivar, (v, reference)) in enumerate(zip(vars, expected))
                @testset "répétition $repetition / variable $ivar $(v.name)" begin
                    for (name, a, b) in zip(
                        (:mean, :gradx, :grady, :hessxx, :hessxy, :hessyy),
                        fields_spectral(v, grid), reference)
                        @testset "$name" begin
                            @test all(isfinite, a)
                            @test all(isfinite, b)
                            # L'ordre des sommes dans chaque cellule n'a pas changé.
                            @test isequal(a, b)
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function bench_pass!(mc, vars; grid=false, label="Spectral", reverse_order=false)
    verify_spectral!(mc, vars, grid, label)
    # Échauffement fait dans verify_spectral!, y compris les recettes de la passe.
    # Aucun setup de restauration : chaque appel reconstruit les sommes R et
    # remplace les six sorties. Il ne change ni particules, ni volumes, ni means.
    a = @benchmarkable serial!($mc, $vars, $grid) evals=1 samples=SAMPLES seconds=SECONDS
    b = @benchmarkable threaded!($mc, $vars, $grid) evals=1 samples=SAMPLES seconds=SECONDS
    if reverse_order
        bt = run(b)
        at = run(a)
    else
        at = run(a)
        bt = run(b)
    end
    ma, mb = median(at), median(bt)
    println("\nMesure : $label ; threads=$(Threads.nthreads())")
    for (name, trial) in (("Séquentiel", at), ("Threads", bt))
        m = median(trial)
        q1, q3 = quantile(trial.times, [0.25, 0.75])
        @printf("  %-11s %.3f ms ; %.2f MiB ; %d allocations ; n=%d\n",
                name, m.time/1e6, m.memory/2.0^20, m.allocs, length(trial.times))
        @printf("              min=%.3f ms ; Q25=%.3f ms ; Q75=%.3f ms\n",
                minimum(trial.times)/1e6, q1/1e6, q3/1e6)
    end
    speedup = ma.time/mb.time
    @printf("  Accélération x%.3f ; temps économisé %.2f %% ; efficacité %.1f %%\n",
            speedup, 100*(1-mb.time/ma.time), 100*speedup/Threads.nthreads())
    if min(length(at.times),length(bt.times)) < 5
        @warn "Peu d'échantillons : augmenter SECONDS pour confirmer le résultat." label
    end
    return (serial_ns=ma.time, threaded_ns=mb.time)
end

function benchmark_state(particles, mean_computer, volume_computer, simu_params)
    rho_from_count = MeanVariable("rho_count", p -> p.mass, simu_params)
    rho = MeanVariable("rho", p -> p.mass, simu_params)
    ux = MeanVariable("ux", p -> p.mass * p.ux, simu_params)
    uy = MeanVariable("uy", p -> p.mass * p.uy, simu_params)
    uz = MeanVariable("uz", p -> p.mass * p.uz, simu_params)
    e = MeanVariable("e", p -> p.mass * p.e, simu_params)
    omega = MeanVariable("omega", p -> p.mass * p.omega, simu_params)
    k = MeanVariable("k", p -> p.mass * 0.5 * ((p.ux - p.means["ux_mean"])^2 + (p.uy - p.means["uy_mean"])^2 + 
            (p.uz - p.means["uz_mean"])^2), simu_params)
    Fex = MeanVariable("Fex", p -> p.mass * (p.ux - p.means["ux_mean"]) * (p.e - p.means["e_mean"]), simu_params)
    Fey = MeanVariable("Fey", p -> p.mass * (p.uy - p.means["uy_mean"]) * (p.e - p.means["e_mean"]), simu_params)
    ke = MeanVariable("ke", p -> p.mass * (p.e - p.means["e_mean"])^2, simu_params)
    Fkin = MeanVariable("fkin", p -> p.mass * 0.5 * (p.ux - p.means["ux_mean"]) * ((p.ux - p.means["ux_mean"])^2 + 
            (p.uy - p.means["uy_mean"])^2 + (p.uz - p.means["uz_mean"])^2), simu_params)
    omega_cond = MeanVariable("omega_cond", p -> ifelse(p.omega > p.means["omega_mean"], p.mass * p.omega, 0.0), simu_params)
    rho_cond = MeanVariable("rho_cond", p -> ifelse(p.omega > p.means["omega_mean"], p.mass, 0.0), simu_params)
    W, X, Y, XX, XY, YY = common_mean_variables(simu_params)
    

    geometry = (W, X, Y, XX, XY, YY)
    # Première mesure : moteur spectral utilisé pour l'estimation des volumes.
    if !simu_params.is_incomp
        set_particle_positions!(volume_computer, particles)
        v = bench_pass!(volume_computer, (rho_from_count,); label="Volumes / densité brute")
    else
        println("Incompressible : pas de reconstruction spectrale des volumes.")
        v = (serial_ns=0.0, threaded_ns=0.0)
    end
    # Fixer une seule fois les volumes et les positions pour toutes les mesures suivantes.
    particle_volumes!(particles, volume_computer, rho_from_count, simu_params)
    set_particle_positions!(mean_computer, particles)
    primary = (rho, ux, uy, uz, e, omega)
    p1 = bench_pass!(mean_computer, (geometry..., primary...);
                     label="Particules / passe 1", reverse_order=true)
    # Sources communes de la passe 2 : reconstruire explicitement la référence,
    # corriger par CSPM puis écrire une fois les moyennes dans les particules.
    serial!(mean_computer, (geometry..., primary...), false)
    corrective_SPH!(mean_computer, geometry..., primary...)
    divide_by_density!(rho, ux, uy, uz, e, omega)
    write_means_to_particles!(particles, primary...)

    secondary = (k, Fex, Fey, ke, omega_cond, rho_cond)
    p2 = bench_pass!(mean_computer, secondary; label="Particules / passe 2")
    g1 = bench_pass!(mean_computer, (geometry..., rho, ux, uy, e, omega);
                     grid=true, label="Grille / passe 1", reverse_order=true)
    g2 = bench_pass!(mean_computer, (k, Fex, ke, Fkin);
                     grid=true, label="Grille / passe 2")

    println("\nBilan indicatif : sommes des médianes du moteur spectral seulement")
    for (label, stages) in (("Sans sortie", (v,p1,p2)),
                            ("Avec sortie", (v,p1,p2,g1,g2)))
        a = sum(t.serial_ns for t in stages)
        b = sum(t.threaded_ns for t in stages)
        @printf("  %-12s %.3f -> %.3f s ; x%.3f\n", label, a/1e9, b/1e9, a/b)
    end
    println("Ces temps excluent CSPM, l'avancement, les fichiers et la préparation des positions sources.")
    println("MiB = allocations cumulées par appel, pas mémoire vive maximale.")
    return nothing
end

function main(config_path; seed=1234)
    isdefined(lsc, :standard_means_reference!) || error(
        "Installer le fichier standard_kernel_scheme.jl fourni, puis relancer Julia.")
    isfile(config_path) || error("Configuration introuvable : $config_path")
    BLAS.set_num_threads(1)
    Random.seed!(seed)
    println("Julia $VERSION ; threads=$(Threads.nthreads()) ; graine=$seed ; BLAS=1")
    println("Pour comparer : même configuration et --gcthreads=1 dans chaque lancement (Julia 1.10).")
    println("Tests d'égalité exacte avant chronométrage, état figé.")
    config = TOML.parsefile(abspath(config_path))
    mktempdir() do temp_dir
        config["paths"]["output_directory"] = joinpath(temp_dir, "simulation")
        temporary_config = joinpath(temp_dir, "config_benchmark.toml")
        open(temporary_config, "w") do io
            TOML.print(io, config)
        end
        particles, mc, vc, params = init_simulation(temporary_config)
        println("N=$(length(particles)) ; grille=$(mc.eval_pos.npos) ; modes=$(mc.nmodes)")
        benchmark_state(particles, mc, vc, params)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    1 <= length(ARGS) <= 2 || error(
        "Usage : julia --project=. --threads=4 --gcthreads=1 test/benchmark_spectral_threads.jl config.toml [graine]")
    main(ARGS[1]; seed=length(ARGS)==2 ? parse(Int,ARGS[2]) : 1234)
end
