#!python

import os
import argparse
from matplotlib.pyplot import *
from lagrangian_post_treatment.simulation import Simulation
from lagrangian_post_treatment.plot_handler import PlotHandler
from progress.bar import ChargingBar

def parse_arguments():
    parser = argparse.ArgumentParser(description=
    """
    Plot comparison of Lagrangian stochastic simulations
    """)
    parser.add_argument('path', type=str, help=
    """
    Full path to root directory of the simulations. Each folder in 'path' should contain one simulation
    """)
    args = parser.parse_args()
    path = args.path
    assert os.path.exists(path), "{:s} is not a valid path".format(path)
    dirs = list_simulation_paths(path)
    assert len(dirs) != 0, "No simulation found in {:s}".format(path)
    return path, dirs

def list_simulation_paths(path):
    subdirs = os.listdir(path=path)
    dirs = []
    for subdir in subdirs:
        if os.path.exists(os.path.join(path, subdir, "raw_output")):
            dirs.append(os.path.join(path, subdir))
    return dirs

if __name__ == '__main__':
    save_path, simu_paths = parse_arguments()
    fig, ax = subplots()
    for path in simu_paths:
        simulation = Simulation(path, None, None)
        k = simulation.time_evolution("kinetic_energy")
        ax.plot(simulation.time, k, label="{:d} x {:d} cells".format(simulation.ncell_v, simulation.ncell_h))
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(os.path.join(save_path, "kinetic_energy"))