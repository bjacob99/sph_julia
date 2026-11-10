#!/usr/bin/env python

import os
import argparse
from lagrangian_post_treatment.simulation import Simulation
from lagrangian_post_treatment.plot_handler import PlotHandler

def parse_arguments():
    parser = argparse.ArgumentParser(description=
    """
    Plot particle movies, and save the animations
    """)
    parser.add_argument('path', type=str, help=
    """
    Full path to root directory of the simulation
    """)
    args = parser.parse_args()
    path = args.path
    assert os.path.exists(path), "{:s} is not a valid path".format(path)
    assert os.path.exists(os.path.join(path, "raw_output")), "no 'raw_output' subdirectory found in {:s}".format(path)
    assert os.path.exists(os.path.join(path, "config.toml")), "no configuration file found in {:s}".format(path)
    return path

if __name__ == '__main__':
    directory = parse_arguments()
    simulation = Simulation(directory)
    plotter = PlotHandler(simulation)
    for var in ['ux', 'uy', 'e', 'omega']:
        plotter.animate_particle_movie(var)