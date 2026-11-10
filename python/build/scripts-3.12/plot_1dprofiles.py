#!python

import os
import argparse
from lagrangian_post_treatment.simulation import Simulation
from lagrangian_post_treatment.plot_handler import PlotHandler

def parse_arguments():
    parser = argparse.ArgumentParser(description=
    """
    Plots the vertical profiles of a Lagrangian stochastic simulation and saves the plots
    """)
    parser.add_argument('path', type=str, help=
    """
    Full path to root directory of the simulation
    """)
    parser.add_argument('-v', '--variable', nargs='+', type=str, help=
    """
    List of mean variable to plot (there must be at least one)
    """)
    args = parser.parse_args()
    path = args.path
    vars = args.variable
    assert os.path.exists(path), "{:s} is not a valid path".format(path)
    assert os.path.exists(os.path.join(path, "raw_output")), "no 'raw_output' subdirectory found in {:s}".format(path)
    assert os.path.exists(os.path.join(path, "config.toml")), "no configuration file found in {:s}".format(path)
    return path, vars

if __name__ == '__main__':
    directory, variables = parse_arguments()
    simulation = Simulation(directory)
    plotter = PlotHandler(simulation)
    for var in variables:
        plotter.plot_vertical_profile(var)
        plotter.animate_1dmovie(var)
    plotter.plot_flux_budget()