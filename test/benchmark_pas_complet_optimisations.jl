# Benchmark COMPLET : A original, B CSPM mutualisé, C CSPM + cellules parallèles.
# julia --project=. --threads=8 --gcthreads=1 test/benchmark_pas_complet_optimisations.jl scripts/config_general.toml [graine]
# Nécessite les quatre fonctions du module : standard_means_reference!,
# standard_means!, corrective_SPH_reference!, corrective_SPH!.
# Pas de modification du module : injection des deux algorithmes dans le benchmark.
# Le calcul des volumes utilise AUSSI le moteur sélectionné (piège de comparaison évité).
# Même état initial, mêmes tirages, copies et compilation hors mesure.
# Les coefficients physiques sont ceux du fichier complex_model fourni ; les adapter
# si la simulation de production emploie un autre modèle.
# Les cas A/B sont séquentiels pour le spectral, dans le même processus multithread que C.
# Ce benchmark ne remplace pas la validation numérique préalable.
# Sorties temporaires sous pwd(), sans fsync ; mesures sensibles au cache du stockage.

import LagrangianStochasticConvection as lsc
using LagrangianStochasticConvection
using BenchmarkTools
using Statistics
using Random
using TOML
using Printf
using LinearAlgebra

const SAMPLES = 5
const ROUNDS = 2
const SECONDS = 30.0 # Budget par essai ; peut être dépassé par un échantillon long.
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

function benchmark_volumes!(particles::Vector{Particle}, volume_computer::MeanComputer, rho_mean::MeanVariable, kwparams::NamedTuple, spectral!::S) where {S}
    is_incomp = kwparams.is_incomp
    Lx = kwparams.Lx
    hx = volume_computer.hx
    if is_incomp
        npart = kwparams.nparticles
        Ly = kwparams.Ly
    end

    if is_incomp
        for particle in particles
            particle.volume = Lx * Ly / npart
        end
    else
        set_particle_positions!(volume_computer, particles)
        spectral!(volume_computer, rho_mean)
        write_means_to_particles!(particles, rho_mean)
        for particle in particles
            vcorr_bottom = lsc.volume_correction(particle.x, hx)
            vcorr_top = lsc.volume_correction(Lx - particle.x, hx)
            particle.volume = particle.mass / particle.means["$(rho_mean.name)_mean"] * vcorr_bottom * vcorr_top
        end
    end
end

function step!(state, spectral!::S, corrector!::F, with_output::Bool; times=nothing) where {S,F}
    (; particles, mean_computer, volume_computer, simu_params, rho_from_count, rho, ux, uy, uz, e, omega, k, Fex, Fey, ke, Fkin, omega_cond, rho_cond, W, X, Y, XX, XY, YY) = state
    npart_to_save = min(10000, length(particles))
    it = 1 # Noms de fichiers constants, réécrits comme le checkpoint courant.
    measure_stage!(times, :volumes) do
        benchmark_volumes!(particles, volume_computer, rho_from_count, simu_params, spectral!)
    end
    measure_stage!(times, :positions) do
        set_particle_positions!(mean_computer, particles)
    end
    measure_stage!(times, :spectral1) do
        spectral!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, uz, e, omega)
    end
    measure_stage!(times, :cspm1) do
        corrector!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, uz, e, omega)
    end
    measure_stage!(times, :favre1) do
        divide_by_density!(rho, ux, uy, uz, e, omega)
        write_means_to_particles!(particles, rho, ux, uy, uz, e, omega)
    end
    measure_stage!(times, :spectral2) do
        spectral!(mean_computer, k, Fex, Fey, ke, omega_cond, rho_cond)
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
            spectral!(mean_computer, W, X, Y, XX, XY, YY, rho, ux, uy, e, omega; is_on_grid=true)
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
            spectral!(mean_computer, k, Fex, ke, Fkin; is_on_grid=true)
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


function variants()
    return (
        (name="A Original", spectral=lsc.standard_means_reference!, corrector=lsc.corrective_SPH_reference!),
        (name="B CSPM seul", spectral=lsc.standard_means_reference!, corrector=lsc.corrective_SPH!),
        (name="C CSPM + threads", spectral=lsc.standard_means!, corrector=lsc.corrective_SPH!),
    )
end

function warmup(initial, mode, with_output, seed)
    work = deepcopy(initial)
    Random.seed!(seed)
    step!(work, mode.spectral, mode.corrector, with_output)
    check_state(work)
    return nothing
end

function benchmark_one(initial, mode, with_output, seed)
    spectral! = mode.spectral
    corrector! = mode.corrector
    # Une copie unique de l'état entier conserve les alias internes entre cellules
    # et particules. GC et copie restent hors mesure ; une évaluation par échantillon.
    bench = @benchmarkable step!(work, $spectral!, $corrector!, $with_output) setup=(work=deepcopy($initial); Random.seed!($seed); GC.gc()) evals=1 samples=SAMPLES seconds=SECONDS
    return run(bench)
end

function benchmark_case(initial, with_output, seed)
    modes = variants()
    label = with_output ? "AVEC sorties (grille et fichiers)" : "SANS sorties"
    println("\nÉchauffement : $label")
    for mode in modes
        warmup(initial, mode, with_output, seed)
    end
    results = [Any[] for _ in modes]
    # Inverser l'ordre au deuxième tour limite le biais d'ordre systématique.
    for round in 1:ROUNDS
        order = isodd(round) ? (1,2,3) : (3,2,1)
        for i in order
            println("Mesure $label / tour $round / $(modes[i].name)")
            push!(results[i], benchmark_one(initial, modes[i], with_output, seed))
        end
    end
    times = Float64[]
    println("\nBILAN : $label")
    for (mode, trials) in zip(modes, results)
        samples = vcat([trial.times for trial in trials]...)
        seconds = median(samples)/1e9
        push!(times, seconds)
        memory = median([median(trial).memory for trial in trials])/2.0^20
        allocs = median([median(trial).allocs for trial in trials])
        q1,q3 = quantile(samples,[0.25,0.75])/1e9
        @printf("  %-18s %.3f s ; %.2f MiB ; %.0f allocations ; n=%d\n",
                mode.name, seconds, memory, allocs, length(samples))
        @printf("                     min=%.3f s ; Q25=%.3f s ; Q75=%.3f s\n",
                minimum(samples)/1e9, q1,q3)
        length(samples)<5 && @warn "Trop peu d'échantillons : augmenter SECONDS."
    end
    for (label,i,j) in (("CSPM seul A -> B",1,2),
                        ("Cellules B -> C",2,3),
                        ("GAIN COMBINE A -> C",1,3))
        @printf("  %-23s x%.3f ; %.2f %% économisés ; %.3f s gagnées/pas\n",
                label,times[i]/times[j],100*(1-times[j]/times[i]),times[i]-times[j])
    end
    return times
end

function profile_once(initial, mode, seed)
    work=deepcopy(initial)
    times=Dict{Symbol,Float64}()
    Random.seed!(seed)
    GC.gc()
    step!(work,mode.spectral,mode.corrector,true; times=times)
    check_state(work)
    return times
end

function profile_all(initial, seed)
    modes=variants()
    for mode in modes
        profile_once(initial,mode,seed) # Compiler le chemin instrumenté également.
    end
    measurements=[Dict{Symbol,Float64}[] for _ in modes]
    for repetition in 1:PROFILE_REPEATS
        # Rotation A/B/C, B/C/A, C/A/B.
        for offset in 0:2
            i=mod1(repetition+offset,3)
            push!(measurements[i],profile_once(initial,modes[i],seed))
        end
    end
    optimized_total=sum(median([t[key] for t in measurements[3]]) for (key,_) in STAGES)
    println("\nDÉCOMPOSITION INDICATIVE : pas avec sorties, médianes de $PROFILE_REPEATS répétitions")
    println("Étape                            A ms        B ms        C ms     Part C")
    for (key,label) in STAGES
        values=[median([t[key] for t in group]) for group in measurements]
        @printf("%-29s %10.2f %11.2f %11.2f %8.1f %%\n",
                label,1000*values[1],1000*values[2],1000*values[3],100*values[3]/optimized_total)
    end
    println("La part C est calculée sur la somme des médianes des étapes.")
    println("Utiliser le benchmark global pour le gain total, pas cette somme.")
end

function main(config_path::String; seed::Int=1234)
    for name in (:standard_means_reference!, :standard_means!,
                 :corrective_SPH_reference!, :corrective_SPH!)
        isdefined(lsc,name) || error("Fonction absente du module : $name")
    end
    isfile(config_path) || error("Configuration introuvable : $config_path")
    BLAS.set_num_threads(1)
    println("Julia $VERSION ; threads=$(Threads.nthreads()) ; graine=$seed ; BLAS=1")
    println("A = spectral séquentiel + CSPM original")
    println("B = spectral séquentiel + CSPM mutualisé")
    println("C = spectral multithread + CSPM mutualisé")
    println("Même sélection spectrale pour les volumes et toutes les passes.")
    println("L'ordonnanceur de C est celui installé dans standard_means! (:dynamic ou :greedy).")
    println("Conserver --gcthreads=1 dans les lancements comparés.")
    println("Les sorties temporaires utilisent le stockage sous $(pwd()).")
    config=TOML.parsefile(abspath(config_path))
    mktempdir(pwd();prefix="benchmark_combine_") do temp_dir
        config["paths"]["output_directory"]=joinpath(temp_dir,"simulation")
        temporary_config=joinpath(temp_dir,"config_benchmark.toml")
        open(temporary_config,"w") do io
            TOML.print(io,config)
        end
        Random.seed!(seed)
        initial=make_state(temporary_config)
        println("N=$(length(initial.particles)) ; grille=$(initial.mean_computer.eval_pos.npos)")
        without=benchmark_case(initial,false,seed)
        with=benchmark_case(initial,true,seed)
        frequency=initial.simu_params.freq_out
        if frequency>0
            average=((frequency-1).*without .+ with)./frequency
            println("\nESTIMATION PONDÉRÉE avec freq_out=$frequency (état figé)")
            for (mode,t) in zip(variants(),average)
                @printf("  %-18s %.3f s/pas\n",mode.name,t)
            end
            @printf("  Gain combiné A -> C : x%.3f ; %.2f %% de temps économisé\n",
                    average[1]/average[3],100*(1-average[3]/average[1]))
            println("Ce calcul pondéré n'est pas une mesure de longue trajectoire.")
        end
        profile_all(initial,seed)
    end
    println("\nMiB = volume alloué par pas, pas mémoire vive maximale.")
    println("La mesure avec fichiers inclut les écritures applicatives, sans flush forcé du cache disque.")
    return nothing
end

if abspath(PROGRAM_FILE)==@__FILE__
    1<=length(ARGS)<=2 || error("Usage : julia --project=. --threads=8 --gcthreads=1 test/benchmark_pas_complet_optimisations.jl config.toml [graine]")
    main(ARGS[1];seed=length(ARGS)==2 ? parse(Int,ARGS[2]) : 1234)
end
