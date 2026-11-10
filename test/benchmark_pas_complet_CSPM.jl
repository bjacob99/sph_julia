# Benchmark du pas complet du modèle complex_model fourni.
# julia --project=. --threads=1 test/benchmark_pas_complet_CSPM.jl scripts/config_general.toml [graine]
# Ancien : lsc.corrective_SPH_reference! ; nouveau : lsc.corrective_SPH!.
# Rétablir les inverses mutualisés avant la mesure. Validation numérique préalable.
# Chaque échantillon repart d'une copie indépendante du même état et du même RNG.
# Copies, initialisation et compilation sont hors du temps mesuré.
# Avec sorties : reconstruction sur grille ET écritures texte incluses.
# Les sorties sont temporaires, sous pwd(), pour employer le système de fichiers
# du répertoire de lancement. Pas de fsync : latence applicative, avec cache OS.
# Mémoire nécessaire : état initial + copie de travail + temporaires du calcul.
# Les coefficients physiques ci-dessous sont repris du fichier fourni par l'utilisateur.
# Si le modèle physique change, actualiser ces coefficients également.

import LagrangianStochasticConvection as lsc
using LagrangianStochasticConvection
using BenchmarkTools
using Statistics
using Random
using TOML
using Printf

const SAMPLES = 10
const SECONDS = 60.0 # Budget par essai ; peut être dépassé par un échantillon long.
const PROFILE_REPEATS = 3

function pressure(particle::Particle)
    return particle.means["rho_mean"] * particle.means["e_mean"]
end

function pressure_gradient(particle::Particle)
    dp_dx = particle.means["rho_gradx"] * particle.means["e_mean"] + particle.means["rho_mean"] * particle.means["e_gradx"]
    dp_dy = particle.means["rho_grady"] * particle.means["e_mean"] + particle.means["rho_mean"] * particle.means["e_grady"]
    return dp_dx, dp_dy
end

function shear_tensor_contraction(particle::Particle)
    shear = 4.0 * particle.means["ux_gradx"]^2
    shear += 2.0 * particle.means["ux_grady"]^2
    shear += 4.0 * particle.means["ux_grady"] * particle.means["uy_gradx"]
    shear += 2.0 * particle.means["uy_gradx"]^2
    shear += 4.0 * particle.means["uy_grady"]^2
    shear *= 0.25
    return shear
end

function Somega(particle::Particle, kwparams::NamedTuple)
    p_mean = pressure(particle)
    dp_dx, dp_dy = pressure_gradient(particle)
    P_over_epsilon = (dp_dx * particle.means["Fex_mean"] + dp_dy * particle.means["Fey_mean"])
    P_over_epsilon /= (p_mean * particle.means["omega_mean"] * particle.means["k_mean"] * kwparams.gamma1 * (kwparams.npoly - 1.0))
    P_over_epsilon += 0.5 * kwparams.Cmu * shear_tensor_contraction(particle) / particle.means["omega_mean"]^2
    return (kwparams.Ceps2 - 1.0) - (kwparams.Ceps1 - 1)*P_over_epsilon
end

function inverse_C(particle::Particle)
    k = particle.means["k_mean"]
    ke = particle.means["ke_mean"]
    invC = [1.0/ke   0.0        0.0        0.0  ;
            0.0      3.0/2.0/k  0.0        0.0  ;
            0.0      0.0        3.0/2.0/k  0.0  ;
            0.0      0.0        0.0        3.0/2.0/k ]
    return invC
end

function drift_ux(particle::Particle, kwparams::NamedTuple)
    p_mean = pressure(particle)
    dp_dx, _ = pressure_gradient(particle)
    k = particle.means["k_mean"]
    invC21 = 0.0
    invC22 = 3.0/2.0/k
    invC23 = 0.0
    invC24 = 0.0
    dux = particle.ux - particle.means["ux_mean"]
    duy = particle.uy - particle.means["uy_mean"]
    duz = particle.uz - particle.means["uz_mean"]
    de = particle.e - particle.means["e_mean"]
    domega = particle.omega - particle.means["omega_mean"]
    G_0x = -dp_dx / (kwparams.gamma1 * particle.means["rho_mean"]) - kwparams.gravity
    G_xx = - (0.5 + 0.75*kwparams.C0) * particle.means["omega_mean"]
    G_xx -= 0.5 * kwparams.C0 * domega * k * invC22
    G_xy = - 0.5 * kwparams.C0 * domega * k * invC23
    G_xz = - 0.5 * kwparams.C0 * domega * k * invC24
    G_ex = dp_dx / (p_mean * (kwparams.npoly - 1.0) * kwparams.gamma1)
    G_ex -= 0.5 * kwparams.C0 * domega * k * invC21 / (kwparams.gamma1*(kwparams.gamma1 - 1.0))
    return G_0x + G_xx*dux + G_xy*duy + G_xz*duz + G_ex*de
end

function drift_uy(particle::Particle, kwparams::NamedTuple)
    p_mean = pressure(particle)
    _, dp_dy = pressure_gradient(particle)
    k = particle.means["k_mean"]
    invC31 = 0.0
    invC32 = 0.0
    invC33 = 3.0/2.0/k
    invC34 = 0.0
    dux = particle.ux - particle.means["ux_mean"]
    duy = particle.uy - particle.means["uy_mean"]
    duz = particle.uz - particle.means["uz_mean"]
    de = particle.e - particle.means["e_mean"]
    domega = particle.omega - particle.means["omega_mean"]
    G_0y = -dp_dy / (kwparams.gamma1 * particle.means["rho_mean"])
    G_yx = - 0.5 * kwparams.C0 * domega * k * invC32
    G_yy = - (0.5 + 0.75*kwparams.C0) * particle.means["omega_mean"]
    G_yy -= 0.5 * kwparams.C0 * domega * k * invC33
    G_yz = - 0.5 * kwparams.C0 * domega * k * invC34
    G_ey = dp_dy / (p_mean * (kwparams.npoly - 1.0) * kwparams.gamma1)
    G_ey -= 0.5 * kwparams.C0 * domega * k * invC31 / (kwparams.gamma1*(kwparams.gamma1 - 1.0))
    return G_0y + G_yx*dux + G_yy*duy + G_yz*duz + G_ey*de
end

function drift_uz(particle::Particle, kwparams::NamedTuple)
    k = particle.means["k_mean"]
    invC41 = 0.0
    invC42 = 0.0
    invC43 = 0.0
    invC44 = 3.0/2.0/k
    dux = particle.ux - particle.means["ux_mean"]
    duy = particle.uy - particle.means["uy_mean"]
    duz = particle.uz - particle.means["uz_mean"]
    de = particle.e - particle.means["e_mean"]
    domega = particle.omega - particle.means["omega_mean"]
    G_zx = - 0.5 * kwparams.C0 * domega * k * invC42
    G_zy = - 0.5 * kwparams.C0 * domega * k * invC43
    G_zz = - (0.5 + 0.75*kwparams.C0) * particle.means["omega_mean"]
    G_zz -= 0.5 * kwparams.C0 * domega * k * invC44
    G_ez = - 0.5 * kwparams.C0 * domega * k * invC41 / (kwparams.gamma1*(kwparams.gamma1 - 1.0))
    return G_zx*dux + G_zy*duy + G_zz*duz + G_ez*de
end

function drift_energy(particle::Particle, kwparams::NamedTuple)
    k = particle.means["k_mean"]
    invC11 = 1.0/particle.means["ke_mean"]
    invC12 = 0.0
    invC13 = 0.0
    invC14 = 0.0
    div_u = particle.means["ux_gradx"] + particle.means["uy_grady"]
    div_Fe = particle.means["Fex_gradx"] + particle.means["Fey_grady"]
    Fe_dot_grad_e = particle.means["Fex_mean"]*particle.means["e_gradx"] + particle.means["Fey_mean"]*particle.means["e_grady"]
    dux = particle.ux - particle.means["ux_mean"]
    duy = particle.uy - particle.means["uy_mean"]
    duz = particle.uz - particle.means["uz_mean"]
    de = particle.e - particle.means["e_mean"]
    domega = particle.omega - particle.means["omega_mean"]
    K_0 = (kwparams.gamma1 - 1.0)/(kwparams.npoly - 1.0) * (div_Fe - Fe_dot_grad_e / particle.means["e_mean"])
    K_0 -= (kwparams.gamma1 - 1.0) * particle.means["e_mean"] * div_u
    K_0 += kwparams.gamma1 * (kwparams.gamma1 - 1.0) * particle.means["omega_mean"] * k
    K_x = - 0.5 * kwparams.C1 * domega * particle.means["ke_mean"] * invC12 * (kwparams.gamma1*(kwparams.gamma1 - 1.0))
    K_y = - 0.5 * kwparams.C1 * domega * particle.means["ke_mean"] * invC13 * (kwparams.gamma1*(kwparams.gamma1 - 1.0))
    K_z = - 0.5 * kwparams.C1 * domega * particle.means["ke_mean"] * invC14 * (kwparams.gamma1*(kwparams.gamma1 - 1.0))
    K_e = - kwparams.gamma1 * (kwparams.gamma1 - 1.0) * particle.means["omega_mean"] * k / (particle.means["e_mean"] * (kwparams.npoly - 1.0))
    K_e -= 0.5 * kwparams.C1 * particle.means["omega_mean"]
    K_e -= (kwparams.gamma1 - 1.0) * div_u
    K_e -= 0.5 * kwparams.C1 * domega * particle.means["ke_mean"] * invC11
    a_e = K_0 + K_x*dux + K_y*duy + K_z*duz + K_e*de
    # if (kwparams.hx <= particle.x) && (particle.x <= kwparams.Lx - kwparams.hx)
    #     a_e += kwparams.kappa * particle.means["e_laplacian"]
    # end
    if (particle.x <= kwparams.Lx - kwparams.hx)
        a_e += kwparams.kappa * particle.means["e_laplacian"]
    end
    return a_e
end

function drift_omega(particle::Particle, kwparams::NamedTuple)
    a_omega = -particle.means["omega_cond_mean"] * (particle.omega - particle.means["omega_mean"])
    a_omega -= particle.means["omega_mean"] * particle.omega * Somega(particle, kwparams)
    return a_omega
end

function diff_u(particle::Particle, kwparams::NamedTuple)
    k = max(particle.means["k_mean"], 0.0)
    return sqrt(kwparams.C0 * particle.omega * k)
end

function diff_energy(particle::Particle, kwparams::NamedTuple)
    ke = max(particle.means["ke_mean"], 0.0)
    return sqrt(kwparams.C1 * particle.omega * ke)
end

function diff_omega(particle::Particle, kwparams::NamedTuple)
    omega_mean = max(particle.means["omega_mean"], 0.0)
    omega_cond_mean = max(particle.means["omega_cond_mean"], 0.0)
    return sqrt(2.0 * kwparams.C2 * omega_mean * omega_cond_mean * particle.omega)
end

function extra_diff_omega(particle::Particle, kwparams::NamedTuple)
    omega_mean = max(particle.means["omega_mean"], 0.0)
    omega_cond_mean = max(particle.means["omega_cond_mean"], 0.0)
    return 2.0 * kwparams.C2 * omega_mean * omega_cond_mean
end


function make_state(config_path)
    particles, mean_computer, volume_computer, simu_params = init_simulation(config_path)
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
    
    return (; particles, mean_computer, volume_computer, simu_params, rho_from_count, rho, ux, uy, uz, e, omega, k, Fex, Fey, ke, Fkin, omega_cond, rho_cond, W, X, Y, XX, XY, YY)
end

# Désactivé pour le benchmark global. Activé seulement pour le relevé par étape.
measure_stage!(f, ::Nothing, key) = f()
function measure_stage!(f, times::Dict{Symbol,Float64}, key)
    start = time_ns()
    result = f()
    times[key] = (time_ns() - start) / 1e9
    return result
end

function step!(state, corrector!::F, with_output::Bool; times=nothing) where {F}
    (; particles, mean_computer, volume_computer, simu_params, rho_from_count, rho, ux, uy, uz, e, omega, k, Fex, Fey, ke, Fkin, omega_cond, rho_cond, W, X, Y, XX, XY, YY) = state
    npart_to_save = min(10000, length(particles))
    it = 1 # Noms de fichiers constants, réécrits comme le checkpoint courant.
    measure_stage!(times, :volumes) do
        particle_volumes!(particles, volume_computer, rho_from_count, simu_params)
    end
    measure_stage!(times, :positions) do
        set_particle_positions!(mean_computer, particles)
    end
    measure_stage!(times, :spectral1) do
        standard_means!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, uz, e, omega)
    end
    measure_stage!(times, :cspm1) do
        corrector!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, uz, e, omega)
    end
    measure_stage!(times, :favre1) do
        divide_by_density!(rho, ux, uy, uz, e, omega)
        write_means_to_particles!(particles, rho, ux, uy, uz, e, omega)
    end
    measure_stage!(times, :spectral2) do
        standard_means!(mean_computer, k, Fex, Fey, ke, omega_cond, rho_cond)
    end
    measure_stage!(times, :cspm2) do
        corrector!(mean_computer, W, X, Y, XX, XY, YY, k, Fex, Fey, ke, omega_cond, rho_cond)
    end
    measure_stage!(times, :favre2) do
        divide_by_density!(rho, k, Fex, Fey, ke)
        divide_by_density!(rho_cond, omega_cond)
        write_means_to_particles!(particles, k, Fex, Fey, ke, omega_cond, rho_cond)
    end
    if with_output
        measure_stage!(times, :grid_spectral1) do
            standard_means!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, e, omega; is_on_grid=true)
        end
    end
    if with_output
        measure_stage!(times, :grid_cspm1) do
            corrector!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, e, omega; is_on_grid=true)
        end
    end
    if with_output
        measure_stage!(times, :grid_favre1) do
            divide_by_density!(rho, ux, uy, e, omega; is_on_grid=true)
        end
    end
    if with_output
        measure_stage!(times, :grid_spectral2) do
            standard_means!(mean_computer, k, Fex, ke, Fkin; is_on_grid=true)
        end
    end
    if with_output
        measure_stage!(times, :grid_cspm2) do
            corrector!(mean_computer, W, X, Y, XX, XY, YY, k, Fex, ke, Fkin; is_on_grid=true)
        end
    end
    if with_output
        measure_stage!(times, :grid_favre2) do
            divide_by_density!(rho, k, Fex, ke, Fkin; is_on_grid=true)
        end
    end
    if with_output
        measure_stage!(times, :io) do
            save_mean_output(it, simu_params.save_directory, "mean", mean_computer, rho, ux, uy, e, omega, k, Fex, ke, Fkin)
            save_mean_output(it, simu_params.save_directory, "gradx", mean_computer, e)
            save_particle_output(it, simu_params.save_directory, particles[1:npart_to_save])
            save_particle_output(-1, simu_params.save_directory, particles)
        end
    end
    measure_stage!(times, :advance) do
        bad = advance_flow!(particles, drift_ux, drift_uy, drift_uz, drift_energy, drift_omega,
                            diff_u, diff_u, diff_u, diff_energy, diff_omega, extra_diff_omega, simu_params)
        bad
    end
    return nothing
end

const STAGES = (
    (:volumes, "Volumes"),
    (:positions, "Positions et cellules"),
    (:spectral1, "Spectral particules 1"),
    (:cspm1, "CSPM particules 1"),
    (:favre1, "Favre et transfert 1"),
    (:spectral2, "Spectral particules 2"),
    (:cspm2, "CSPM particules 2"),
    (:favre2, "Favre et transfert 2"),
    (:grid_spectral1, "Spectral grille 1"),
    (:grid_cspm1, "CSPM grille 1"),
    (:grid_favre1, "Favre grille 1"),
    (:grid_spectral2, "Spectral grille 2"),
    (:grid_cspm2, "CSPM grille 2"),
    (:grid_favre2, "Favre grille 2"),
    (:io, "Écritures texte"),
    (:advance, "Avancement et contrôles"),
)

function check_state(state)
    # Hors mesure : un benchmark de calcul non fini n'est pas exploitable.
    all(p -> all(isfinite, (p.x, p.y, p.ux, p.uy, p.uz, p.e, p.omega)) &&
             all(isfinite, values(p.means)), state.particles) ||
        error("État non fini après le pas : examiner la configuration et la simulation avant de chronométrer.")
end

function warmup(initial, corrector!, with_output, seed)
    work = deepcopy(initial)
    Random.seed!(seed)
    step!(work, corrector!, with_output)
    check_state(work)
    return nothing
end

function print_trial(name, trial)
    m = median(trial)
    q1, q3 = quantile(trial.times, [0.25, 0.75])
    @printf("  %-8s : %.3f s ; %.2f MiB ; %d allocations ; n=%d\n",
            name, m.time/1e9, m.memory/2.0^20, m.allocs, length(trial.times))
    @printf("             min=%.3f s ; Q25=%.3f s ; Q75=%.3f s\n",
            minimum(trial.times)/1e9, q1/1e9, q3/1e9)
    return m.time/1e9
end

function benchmark_pair(initial, with_output, seed)
    label = with_output ? "AVEC sorties (grille + fichiers)" : "SANS sorties"
    println("\nÉchauffement : $label")
    warmup(initial, lsc.corrective_SPH_reference!, with_output, seed)
    warmup(initial, lsc.corrective_SPH!, with_output, seed)

    # deepcopy porte sur TOUT l'état d'un seul coup afin de conserver les liens
    # entre particules, cellules et MeanComputer au sein de la copie.
    # GC.gc() après la copie limite l'imputation du nettoyage de setup au pas.
    # Ces opérations sont hors chronométrage. evals=1 est indispensable.
    old_bench = @benchmarkable step!(work, lsc.corrective_SPH_reference!, $with_output) setup=(work=deepcopy($initial); Random.seed!($seed); GC.gc()) evals=1 samples=SAMPLES seconds=SECONDS
    new_bench = @benchmarkable step!(work, lsc.corrective_SPH!, $with_output) setup=(work=deepcopy($initial); Random.seed!($seed); GC.gc()) evals=1 samples=SAMPLES seconds=SECONDS
    println("Mesure du pas complet : $label")
    if with_output
        new_trial = run(new_bench)
        old_trial = run(old_bench)
    else
        old_trial = run(old_bench)
        new_trial = run(new_bench)
    end
    a = print_trial("Ancien", old_trial)
    b = print_trial("Nouveau", new_trial)
    @printf("  Accélération x%.3f ; %.2f %% de temps économisé ; %.3f s gagnées/pas\n",
            a/b, 100*(1-b/a), a-b)
    if min(length(old_trial.times), length(new_trial.times)) < 5
        @warn "Moins de cinq échantillons : augmenter SECONDS avant de conclure."
    end
    return (old=a, new=b)
end

function profile_once(initial, corrector!, with_output, seed)
    work = deepcopy(initial)
    times = Dict{Symbol,Float64}()
    Random.seed!(seed)
    GC.gc()
    step!(work, corrector!, with_output; times=times)
    check_state(work)
    return times
end

function profile_pair(initial, with_output, seed)
    # Échauffer également la variante instrumentée, dont le type times diffère.
    profile_once(initial, lsc.corrective_SPH_reference!, with_output, seed)
    profile_once(initial, lsc.corrective_SPH!, with_output, seed)
    olds = Dict{Symbol,Float64}[]
    news = Dict{Symbol,Float64}[]
    for repetition in 1:PROFILE_REPEATS
        if isodd(repetition)
            push!(olds, profile_once(initial, lsc.corrective_SPH_reference!, with_output, seed))
            push!(news, profile_once(initial, lsc.corrective_SPH!, with_output, seed))
        else
            push!(news, profile_once(initial, lsc.corrective_SPH!, with_output, seed))
            push!(olds, profile_once(initial, lsc.corrective_SPH_reference!, with_output, seed))
        end
    end
    active = filter(item -> haskey(news[1], item[1]), STAGES)
    new_total = sum(median([t[key] for t in news]) for (key, _) in active)
    println("\nDécomposition indicative : médianes de $PROFILE_REPEATS pas instrumentés")
    println("Étape                           Ancien ms  Nouveau ms  Part nouveau")
    for (key, label) in active
        a = median([t[key] for t in olds])
        b = median([t[key] for t in news])
        @printf("%-30s %10.2f %11.2f %11.1f %%\n", label, 1e3*a, 1e3*b, 100*b/new_total)
    end
    println("Les parts sont calculées sur la somme des médianes des étapes.")
    println("Utiliser le benchmark global pour le gain total ; ce relevé localise les coûts.")
    return nothing
end

function main(config_path::String; seed::Int=1234)
    isdefined(lsc, :corrective_SPH_reference!) || error("corrective_SPH_reference! absente.")
    isfile(config_path) || error("Configuration introuvable : $config_path")
    println("Julia $VERSION ; threads=$(Threads.nthreads()) ; graine=$seed")
    println("Même état initial et mêmes tirages à chaque échantillon ; aucune compilation mesurée.")
    println("Les sorties de benchmark sont temporaires sous : $(pwd())")
    config = TOML.parsefile(abspath(config_path))
    mktempdir(pwd(); prefix="benchmark_cspm_") do temp_dir
        config["paths"]["output_directory"] = joinpath(temp_dir, "simulation")
        temporary_config = joinpath(temp_dir, "config_benchmark.toml")
        open(temporary_config, "w") do io
            TOML.print(io, config)
        end
        Random.seed!(seed)
        initial = make_state(temporary_config)
        println("N=$(length(initial.particles)) ; grille=$(initial.mean_computer.eval_pos.npos)")
        no_output = benchmark_pair(initial, false, seed)
        with_output = benchmark_pair(initial, true, seed)
        frequency = initial.simu_params.freq_out
        if frequency > 0
            a = ((frequency-1)*no_output.old + with_output.old)/frequency
            b = ((frequency-1)*no_output.new + with_output.new)/frequency
            @printf("\nEstimation moyenne avec freq_out=%d : %.3f -> %.3f s/pas ; x%.3f\n",
                    frequency, a, b, a/b)
            println("Estimation à partir de l'état figé, pas une mesure d'une longue trajectoire.")
        end
        # Avec sorties pour exposer tous les postes, y compris la grille et l'I/O.
        profile_pair(initial, true, seed)
    end
    println("\nMiB = octets alloués par pas, pas mémoire vive maximale.")
    println("Le coût des fichiers dépend du stockage et de son cache ; aucune garantie de flush disque.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    1 <= length(ARGS) <= 2 || error("Usage : julia --project=. --threads=1 test/benchmark_pas_complet_CSPM.jl config.toml [graine]")
    main(ARGS[1]; seed=length(ARGS)==2 ? parse(Int,ARGS[2]) : 1234)
end
