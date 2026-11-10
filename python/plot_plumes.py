#!/usr/bin/env python

import os
import argparse
from lagrangian_post_treatment.simulation import Simulation
from lagrangian_post_treatment.plot_handler import PlotHandler

def parse_arguments():
    parser = argparse.ArgumentParser(description=
    """
    Plot vertical profiles for upwards and downwards plumes separately, and save the plots
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
        plotter.plot_plume_profiles(var)
    for var in ['ux', 'uy', 'e', 'omega']:
        plotter.plot_pdf(var)
