# Chronométrage CSPM uniquement, après validation numérique.
# Depuis la racine du projet :
# julia --project=. --threads=1 test/benchmark_mutualisation_CSPM.jl scripts/config_general.toml [graine]
# Prérequis : corrective_SPH_reference! (originale), corrective_SPH! (mutualisée),
# et BenchmarkTools dans l'environnement actif.
# Rétablir les inverses mutualisés avant de mesurer l'optimisation complète.
# Initialisation, sommes spectrales, copies de restauration et sorties exclues
# des temps mesurés. La construction de la géométrie CSPM reste incluse.
# État figé, aucune intégration temporelle. Pas de modification des sorties existantes.

import LagrangianStochasticConvection as lsc
using LagrangianStochasticConvection
using BenchmarkTools
using Statistics
using Random
using TOML
using Printf

const BENCH_SAMPLES = 20
const BENCH_SECONDS = 10.0 # Budget par version et par passe ; au moins un échantillon.

function fields_cspm(v, on_grid)
    if on_grid
        return (v.mean_eval, v.gradx_eval, v.grady_eval,
                v.hessxx_eval, v.hessxy_eval, v.hessyy_eval)
    else
        return (v.mean_part, v.gradx_part, v.grady_part,
                v.hessxx_part, v.hessxy_part, v.hessyy_part)
    end
end

function restore_fields!(variables, raw, on_grid)
    for (v, saved) in zip(variables, raw)
        for (dest, src) in zip(fields_cspm(v, on_grid), saved)
            copyto!(dest, src)
        end
    end
    return nothing
end

function run_old!(mc, geometry, variables, on_grid)
    lsc.corrective_SPH_reference!(mc, geometry..., variables...;
                                 is_on_grid=on_grid)
    return nothing
end

function run_new!(mc, geometry, variables, on_grid)
    lsc.corrective_SPH!(mc, geometry..., variables...; is_on_grid=on_grid)
    return nothing
end

function benchmark_pass!(mc, geometry, variables;
                         on_grid=false, label="CSPM", reverse_order=false)
    # Sauvegarde uniquement les six tableaux actifs, une seule fois.
    raw = map(v -> map(copy, fields_cspm(v, on_grid)), variables)
    all(v -> all(a -> all(isfinite, a), fields_cspm(v, on_grid)), variables) ||
        error("Entrées non finies pour $label")

    # Échauffement : compiler les deux versions sur les mêmes données.
    run_old!(mc, geometry, variables, on_grid)
    all(v -> all(a -> all(isfinite, a), fields_cspm(v, on_grid)), variables) ||
        error("Résultats de référence non finis pour $label")
    restore_fields!(variables, raw, on_grid)
    run_new!(mc, geometry, variables, on_grid)
    all(v -> all(a -> all(isfinite, a), fields_cspm(v, on_grid)), variables) ||
        error("Résultats mutualisés non finis pour $label")
    restore_fields!(variables, raw, on_grid)

    # setup est exécuté hors chronométrage avant chaque échantillon.
    # evals=1 interdit de corriger plusieurs fois les mêmes données par échantillon.
    old_bench = @benchmarkable run_old!($mc, $geometry, $variables, $on_grid) setup=(restore_fields!($variables, $raw, $on_grid)) evals=1 samples=BENCH_SAMPLES seconds=BENCH_SECONDS
    new_bench = @benchmarkable run_new!($mc, $geometry, $variables, $on_grid) setup=(restore_fields!($variables, $raw, $on_grid)) evals=1 samples=BENCH_SAMPLES seconds=BENCH_SECONDS

    println("\nMesure : $label")
    if reverse_order
        new_trial = run(new_bench)
        old_trial = run(old_bench)
    else
        old_trial = run(old_bench)
        new_trial = run(new_bench)
    end
    restore_fields!(variables, raw, on_grid)

    a, b = median(old_trial), median(new_trial)
    @printf("  Ancien : %10.3f ms ; %9.3f MiB ; %d allocations ; n=%d\n",
            a.time / 1e6, a.memory / 2.0^20, a.allocs, length(old_trial.times))
    @printf("  Nouveau: %10.3f ms ; %9.3f MiB ; %d allocations ; n=%d\n",
            b.time / 1e6, b.memory / 2.0^20, b.allocs, length(new_trial.times))
    @printf("  Accélération : x%.3f ; temps économisé : %.2f %%\n",
            a.time / b.time, 100 * (1 - b.time / a.time))
    # Minimum et quartiles aident à repérer des mesures instables.
    for (name, trial) in (("Ancien", old_trial), ("Nouveau", new_trial))
        q1, q3 = quantile(trial.times, [0.25, 0.75])
        @printf("  %s : min=%.3f ms ; Q25=%.3f ms ; Q75=%.3f ms\n",
                name, minimum(trial.times) / 1e6, q1 / 1e6, q3 / 1e6)
    end
    if min(length(old_trial.times), length(new_trial.times)) < 5
        @warn "Peu d'échantillons : augmenter BENCH_SECONDS avant de conclure." label
    end
    return (old_ns=a.time, new_ns=b.time)
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
    primary = (rho, ux, uy, uz, e, omega)
    secondary = (k, Fex, Fey, ke, omega_cond, rho_cond)
    particle_volumes!(particles, volume_computer, rho_from_count, simu_params)
    set_particle_positions!(mean_computer, particles)

    standard_means!(mean_computer, geometry..., primary...)
    p1 = benchmark_pass!(mean_computer, geometry, primary;
                         label="Particules / passe 1")

    # Préparer une fois les sources de la seconde passe, via la référence.
    # benchmark_pass! a restauré les tableaux bruts avant de rendre la main.
    run_old!(mean_computer, geometry, primary, false)
    divide_by_density!(rho, ux, uy, uz, e, omega)
    write_means_to_particles!(particles, primary...)

    standard_means!(mean_computer, secondary...)
    p2 = benchmark_pass!(mean_computer, geometry, secondary;
                         label="Particules / passe 2", reverse_order=true)

    grid_primary = (rho, ux, uy, e, omega)
    standard_means!(mean_computer, geometry..., grid_primary...; is_on_grid=true)
    g1 = benchmark_pass!(mean_computer, geometry, grid_primary;
                         on_grid=true, label="Grille / passe 1")

    grid_secondary = (k, Fex, ke, Fkin)
    standard_means!(mean_computer, grid_secondary...; is_on_grid=true)
    g2 = benchmark_pass!(mean_computer, geometry, grid_secondary;
                         on_grid=true, label="Grille / passe 2", reverse_order=true)

    println("\nBilan indicatif : sommes des temps médians, CSPM seulement")
    for (label, passes) in (("Pas sans sortie", (p1, p2)),
                            ("Pas avec sortie", (p1, p2, g1, g2)))
        old_ns = sum(p.old_ns for p in passes)
        new_ns = sum(p.new_ns for p in passes)
        @printf("  %-18s : %.3f -> %.3f ms ; x%.3f\n",
                label, old_ns / 1e6, new_ns / 1e6, old_ns / new_ns)
    end
    println("Ces sommes excluent les autres étapes du pas et les écritures de fichiers.")
    println("MiB = volume alloué par appel ; ce n'est pas la mémoire vive maximale.")
    return nothing
end

function main(config_path::String; seed::Int=1234)
    isdefined(lsc, :corrective_SPH_reference!) || error(
        "Conserver l'ancien algorithme sous le nom corrective_SPH_reference!.")
    isfile(config_path) || error("Configuration introuvable : $config_path")
    Random.seed!(seed)
    println("Julia $VERSION ; threads=$(Threads.nthreads()) ; graine=$seed")
    println("État figé ; chronométrage CSPM seul, après compilation.")
    println("Utiliser corrective_SPH! avec inverses mutualisés pour mesurer le gain complet.")
    println("Exécuter auparavant le test de non-régression numérique.")

    # init_simulation copie normalement la configuration dans un dossier de sortie.
    # Rediriger ces effets vers un dossier temporaire. Les chemins relatifs
    # d'initial_directory restent relatifs au répertoire courant, comme dans le code.
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
        "Usage : julia --project=. --threads=1 test/benchmark_mutualisation_CSPM.jl config.toml [graine]")
    seed = length(ARGS) == 2 ? parse(Int, ARGS[2]) : 1234
    main(ARGS[1]; seed=seed)
end
