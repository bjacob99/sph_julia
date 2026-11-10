#!/usr/bin/env python

import argparse
import os
import numpy as np
from matplotlib.pyplot import *
from scipy.optimize import curve_fit
from lagrangian_post_treatment.reader import Reader

rcParams.update({'text.usetex':True,
                 'font.family':'serif',
                 'font.serif':['Palatino'],
                 'figure.figsize':(15.0, 10.0),
                 'font.size':30.0,
                 'lines.linewidth': 3,
                 'lines.markersize':10,
                 'animation.writer':'ffmpeg'})
cycle = rcParams['axes.prop_cycle'].by_key()['color'] + ['black', 'gold', 'chartreuse', 'cyan', 'fuchsia']
ioff()

DEFAULT_DURATION_CHUNK = 500.0

class ParticleSimulation:

    def __init__(self, dirs, dt_chunk):
        if isinstance(dirs, list):
            self.dirs = [os.path.normpath(dir) for dir in dirs]
        else:
            self.dirs = [os.path.normpath(dirs)]
        self.readers = [Reader(dir) for dir in self.dirs]
        first_reader = self.readers[0]
        self.time = first_reader.time
        self.Lx = first_reader.Lx
        self.Ly = first_reader.Ly
        if len(self.readers) > 1:
            for reader in self.readers[1:]:
                local_time = reader.time + self.time[-1]
                self.time = np.concatenate((self.time, local_time))
        self.nchunk = int((self.time[-1] - self.time[0]) // dt_chunk) + 1
        self.dt_chunk = dt_chunk

    def get_particle_property(self, kw):
        field = self.readers[0].get_particle_property(kw)
        if len(self.readers) > 1:
            for reader in self.readers[1:]:
                local_field = reader.get_particle_property(kw)
                field = np.concatenate((field, local_field), axis=0)
        return field

    def cut_time_series(self, x, ichunk):
        assert (0 <= ichunk) and (ichunk < self.nchunk), "Requesting invalid chunk of simulation (must be positive and smaller than {:d}, got {:d})".format(self.nchunk, ichunk)
        start_time = ichunk * self.dt_chunk
        end_time = (ichunk+1) * self.dt_chunk
        start_index = np.argmin(np.abs(self.time - start_time))
        end_index = np.argmin(np.abs(self.time - end_time))
        return x[start_index:end_index]
    
    def get_particle_subtraj(self, id, ichunk):
        max_id = self.readers[0].nparticles_output - 1
        assert (id <= max_id), "Requesting invalid particle id (maximum is {:d}, got {:d})".format(max_id, id)
        x = self.get_particle_property("x")[:,id]
        y = self.get_particle_property("y")[:,id]
        xtraj = self.cut_time_series(x, ichunk)
        ytraj = self.cut_time_series(y, ichunk)
        return xtraj, ytraj
    
    def cut_valid_subtraj(self, xtraj, ytraj):
        index_end = 0
        stop = False
        while not stop:
            if index_end == len(xtraj):
                stop = True
            elif (xtraj[index_end] < 0.05*self.Lx) or (xtraj[index_end] > 0.95*self.Lx):
                stop = True
            elif (ytraj[index_end] < 0.05*self.Ly) or (ytraj[index_end] > 0.95*self.Ly):
                stop = True
            else:
                index_end += 1
        return xtraj[:index_end], ytraj[:index_end]
    
    def get_all_travel_distances(self, ichunk):
        ntime = int(self.dt_chunk // (self.time[1] - self.time[0]))
        distances = np.zeros(ntime)
        count = np.zeros(ntime, dtype='int')
        for id in range(self.readers[0].nparticles_output):
            x, y = self.get_particle_subtraj(id, ichunk)
            x, y = self.cut_valid_subtraj(x, y)
            ndist = len(x)
            if (0 < ndist) and (ndist <= ntime):
                distances[:ndist] += np.sqrt((x-x[0])**2 + (y-y[0])**2)
                count[:ndist] += 1
        distances = remove_trailing_zeros(distances)
        count = remove_trailing_zeros(count)
        t = self.time[:len(distances)]
        distances /= count
        return t, distances

def parse_arguments():
    parser = argparse.ArgumentParser(description=
    """
    Save all plots for a Lagrangian stochastic simulation.
    """)
    parser.add_argument('path', type=str, help=
    """
    Full path to root directory of the simulation
    """)
    parser.add_argument('-d', type=float, default=DEFAULT_DURATION_CHUNK)
    args = parser.parse_args()
    path = args.path
    dt_chunk = args.d
    assert os.path.exists(path), "{:s} is not a valid path".format(path)
    assert os.path.exists(os.path.join(path, "raw_output")), "no raw_output subdirectory found in {:s}".format(path)
    assert os.path.exists(os.path.join(path, "config.toml")), "no configuration file found in {:s}".format(path)
    assert dt_chunk > 0.0, "dt_chunk must be positive"
    return path, dt_chunk

def remove_trailing_zeros(x):
    i = len(x) - 1
    while (i != 0) and (x[i] == 0.0):
        i -= 1
    return x[:i]

def fdata_travel_distance(t, B, t0):
    res = np.where(t < t0, B*t**2/(2.0*t0), B*t - B*t0/2.0)
    return res

def fit_travel_distance(t, r):
    t0_init = np.max(t) / 4.0
    B_init = np.max(r)**2 / np.max(t)
    p_init = [B_init, t0_init]
    p, _ = curve_fit(fdata_travel_distance, t, r**2, p_init)
    return p[0], p[1]

if __name__ == "__main__":
    path, dt_chunk = parse_arguments()
    simulation = ParticleSimulation(path, dt_chunk)
    for ichunk in range(simulation.nchunk):
        t, r = simulation.get_all_travel_distances(ichunk)
        B, t0 = fit_travel_distance(t, r)
        print("Drift velocity = {:.0f} m/s, turbulent diffusivity = {:.0f} km2/s".format(np.sqrt(B/(2.0*t0)), B/1e6))
        plot(t[1:], fdata_travel_distance(t[1:], B, t0) / 1e6)
        scatter(t[1:], r[1:]**2 / 1e6, marker='.')
        show()
