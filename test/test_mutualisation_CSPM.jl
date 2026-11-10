# Validation numérique de la mutualisation CSPM, sans chronométrage.
# Depuis la racine du projet :
# julia --project=. --threads=1 test_mutualisation_CSPM.jl config.toml [graine]
#
# Préconditions dans LagrangianStochasticConvection :
#   corrective_SPH_reference! : ancien algorithme, conservé indépendamment ;
#   corrective_SPH!           : nouvel algorithme mutualisé.
# Ce test ne prouve pas la justesse physique : il vérifie la non-régression.
# Conserver également les tests analytiques linéaire/cosinus existants.
# Les paramètres N, noyaux et grille sont ceux de la configuration.
# Un seul état est testé ; aucune intégration temporelle ni sortie de simulation.

import LagrangianStochasticConvection as lsc
using LagrangianStochasticConvection
using Test
using Random
using TOML
using Printf

const CSPM_RTOL = 1e-9
const CSPM_ATOL = 1e-12
const COMPONENTS = (:mean, :gradx, :grady, :hessxx, :hessxy, :hessyy)

# Accès local : ne dépend pas de l'ajout de cspm_fields dans la bibliothèque.
function fields_cspm(v, on_grid)
    suffix = on_grid ? "_eval" : "_part"
    return map(c -> getfield(v, Symbol(string(c), suffix)), COMPONENTS)
end

function compare_cspm!(mc, geometry, raw_variables;
                       on_grid=false, label="CSPM")
                       
    reference = deepcopy(raw_variables)
    optimized = deepcopy(raw_variables)
    geometry_ref = deepcopy(geometry)
    geometry_new = deepcopy(geometry)

    lsc.corrective_SPH_reference!(mc, geometry_ref..., reference...;
                                  is_on_grid=on_grid)
    lsc.corrective_SPH!(mc, geometry_new..., optimized...;
                        is_on_grid=on_grid)

    @testset "$label" begin
        # Les moments géométriques doivent rester bruts pour la seconde passe.
        @testset "Géométrie inchangée" begin
            for (original, ref, new) in zip(geometry, geometry_ref, geometry_new)
                for grid in (false, true)
                    for (a, b, c) in zip(fields_cspm(original, grid),
                                         fields_cspm(ref, grid),
                                         fields_cspm(new, grid))
                        @test isequal(a, b)
                        @test isequal(a, c)
                    end
                end
            end
        end
        for (raw, ref, new) in zip(raw_variables, reference, optimized)
            @testset "$(ref.name)" begin
                for (component, a, b) in zip(COMPONENTS,
                                             fields_cspm(ref, on_grid),
                                             fields_cspm(new, on_grid))
                    @testset "$component" begin
                        @test size(a) == size(b)
                        @test !isempty(a)
                        finite = all(isfinite, a) && all(isfinite, b)
                        @test finite
                        if size(a) == size(b) && !isempty(a) && finite
                            # Critère point par point, y compris près des zéros.
                            # ratio <= 1 signifie que tous les points passent.
                            max_abs = 0.0
                            max_ratio = -Inf
                            worst = firstindex(a)
                            for i in eachindex(a, b)
                                delta = abs(a[i] - b[i])
                                tol = CSPM_ATOL + CSPM_RTOL * max(abs(a[i]), abs(b[i]))
                                ratio = delta / tol
                                max_abs = max(max_abs, delta)
                                if ratio > max_ratio
                                    max_ratio, worst = ratio, i
                                end
                            end
                            @printf("%-28s %-12s %-7s max_abs=%10.3e ratio=%10.3e index=%d\n",
                                    label, ref.name, string(component),
                                    max_abs, max_ratio, worst)
                            if max_ratio > 1.0
                                pos = on_grid ? mc.eval_pos : mc.part_pos
                                @printf("  Point en échec : x=%.16e y=%.16e ref=%.16e new=%.16e\n",
                                        pos.x[worst], pos.y[worst], a[worst], b[worst])
                            end
                            @test max_ratio <= 1.0
                        end
                    end
                end
                # Une correction sur la grille ne doit pas toucher les particules,
                # et réciproquement.
                for (a, b, c) in zip(fields_cspm(raw, !on_grid),
                                     fields_cspm(ref, !on_grid),
                                     fields_cspm(new, !on_grid))
                    @test isequal(a, b)
                    @test isequal(a, c)
                end
            end
        end
    end
    # La référence seule fournit les champs utilisés pour préparer la passe 2.
    # Cela isole CSPM : une éventuelle différence n'est pas réinjectée dans
    # les quantités sources avant leur comparaison.
    return reference
end

function verify_state(particles, mean_computer, volume_computer, simu_params)
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

    @testset "Mutualisation CSPM : état figé" begin
        # 1. Première passe aux positions des particules.
        standard_means!(mean_computer, geometry..., primary...)
        primary_ref = compare_cspm!(mean_computer, geometry, primary;
                                   label="Particules / passe 1")
        # Division de Favre seulement APRÈS comparaison CSPM.
        divide_by_density!(primary_ref[1], primary_ref[2:end]...)
        write_means_to_particles!(particles, primary_ref...)

        # 2. Fluctuations calculées à partir des mêmes moyennes de référence.
        standard_means!(mean_computer, secondary...)
        compare_cspm!(mean_computer, geometry, secondary;
                      label="Particules / passe 2")

        # 3. Première passe sur la grille, comme dans complex_model.jl.
        grid_primary = (rho, ux, uy, e, omega)
        standard_means!(mean_computer, geometry..., grid_primary...;
                         is_on_grid=true)
        compare_cspm!(mean_computer, geometry, grid_primary;
                      on_grid=true, label="Grille / passe 1")

        # 4. Diagnostics sur grille. Les recettes lisent toujours les moyennes
        # aux particules écrites après la première comparaison.
        grid_secondary = (k, Fex, ke, Fkin)
        standard_means!(mean_computer, grid_secondary...; is_on_grid=true)
        compare_cspm!(mean_computer, geometry, grid_secondary;
                      on_grid=true, label="Grille / passe 2")
    end
    return nothing
end

function main(config_path::String; seed::Int=1234)
    isdefined(lsc, :corrective_SPH_reference!) || error(
        "Fonction lsc.corrective_SPH_reference! absente : conserver l'ancien " *
        "algorithme sous ce nom avant d'exécuter ce test.")
    isfile(config_path) || error("Configuration introuvable : $config_path")
    Random.seed!(seed)
    println("Graine : $seed ; rtol=$CSPM_RTOL ; atol=$CSPM_ATOL")
    println("Vérifier que corrective_SPH! contient bien la version mutualisée.")

    # read_config_file/init_simulation créent normalement le dossier de sortie
    # et y copient la configuration. Rediriger ces effets dans un dossier
    # temporaire, afin de préserver les résultats de simulation existants.
    # initial_directory garde la convention du code : chemin relatif au cwd.
    config = TOML.parsefile(abspath(config_path))
    mktempdir() do temp_dir
        config["paths"]["output_directory"] = joinpath(temp_dir, "simulation")
        temporary_config = joinpath(temp_dir, "config_test.toml")
        open(temporary_config, "w") do io
            TOML.print(io, config)
        end
        particles, mc, vc, params = init_simulation(temporary_config)
        println("Particules : $(length(particles)) ; points de grille : $(mc.eval_pos.npos)")
        verify_state(particles, mc, vc, params)
    end
    println("Validation terminée : toutes les comparaisons ont réussi.")
    println("Ce résultat concerne cet état ; aucun gain de temps n'a été mesuré.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    1 <= length(ARGS) <= 2 || error(
        "Usage : julia --project=. --threads=1 test_mutualisation_CSPM.jl config.toml [graine]")
    seed = length(ARGS) == 2 ? parse(Int, ARGS[2]) : 1234
    main(ARGS[1]; seed=seed)
end
