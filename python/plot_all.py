#!/usr/bin/env python

import os
import argparse
from lagrangian_post_treatment.simulation import Simulation
from lagrangian_post_treatment.plot_handler import PlotHandler
from progress.bar import ChargingBar

def parse_arguments():
    parser = argparse.ArgumentParser(description=
    """
    Plot output of Lagrangian stochastic simulation
    """
    )
    parser.add_argument('path', type=str, help=
    """
    Full path to root directory of the simulation
    """
    )
    parser.add_argument('--plumes', action='store_true', help=
    """
    Add plots related to up- and down-wards plume distribution 
    """
    )
    parser.add_argument('--short', action='store_true', help=
    """
    Shorten the animations by only plotting one every ten frames
    """
    )
    parser.add_argument('--init_time', type=float, help=
    """
    Start of steady state, in seconds. If not provided, the steady state will be detected automatically
    """
    )
    parser.add_argument('--max_diffusion_time', type=float, help=
    """
    Maximum time increment used to measure diffusion coefficients. Default to 300 seconds
    """
    )
    parser.add_argument('--essential', action='store_true', help=
    """
    Only plot the essential variables (density, velocity, internal energy)
    """
    )
    parser.add_argument('--flux_up_down', action='store_true', help=
    """
    Plot flux budget for upflows and downflows separately, instead of full flux budget
    """
    )
    args = parser.parse_args()
    path = args.path
    is_plotting_plumes = args.plumes
    is_shortening_movie = args.short
    init_time = args.init_time
    max_diffusion_time = args.max_diffusion_time
    essential = args.essential
    up_down = args.flux_up_down
    assert os.path.exists(path), "{:s} is not a valid path".format(path)
    assert os.path.exists(os.path.join(path, "raw_output")), "no 'raw_output' subdirectory found in {:s}".format(path)
    assert os.path.exists(os.path.join(path, "config.toml")), "no configuration file found in {:s}".format(path)
    return path, is_plotting_plumes, is_shortening_movie, init_time, max_diffusion_time, essential, up_down

if __name__ == '__main__':
    directory, is_plotting_plumes, is_shortening_movie, init_time, max_diffusion_time, essential, up_down = parse_arguments()
    simulation = Simulation(directory, init_time, max_diffusion_time)
    plotter = PlotHandler(simulation)
    if essential:
        mean_variables = ['density', 'velocity_x', 'velocity_y', 'internal_energy']
        particle_variables = ['ux', 'uy', 'e']
    else:
        mean_variables = ['density', 'velocity_x', 'internal_energy', 'turbulent_frequency', 'kinetic_energy',
                            'convective_flux', 'energy_variance', 'kinetic_flux', 'radiative_flux']
        particle_variables = ['ux', 'uy', 'e', 'omega']
    with ChargingBar("Plotting mean flow variables...", max=len(mean_variables)) as bar:
        for var in mean_variables:
            plotter.plot_time_evolution(var)
            plotter.plot_vertical_profile(var)
            plotter.animate_1dmovie(var, short=is_shortening_movie)
            plotter.animate_2dmovie(var, short=is_shortening_movie)
            if is_plotting_plumes:
                plotter.plot_plume_profiles(var)
            bar.next()
    with ChargingBar("Plotting particle variables...", max=len(particle_variables)) as bar:
        for var in particle_variables:
            plotter.animate_particle_movie(var, short=is_shortening_movie)
            if is_plotting_plumes:
                plotter.plot_pdf(var)
            bar.next()
    if not essential:
        plotter.plot_flux_budget(up_down)
        plotter.plot_vertical_profile('nabla')
        plotter.plot_particle_dispersion()
    if is_plotting_plumes:
        plotter.plot_kurtosis_skewness()