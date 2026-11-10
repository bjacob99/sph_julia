#!/usr/bin/env python

import os
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.pyplot import *
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
style.use('seaborn-v0_8')
prop_cycle = rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
rcParams.update({'text.usetex': True,
                 'mathtext.default': 'regular',
                 'font.family': 'serif',
                 'font.serif': ['Palatino'],
                 'font.size': 10.0,
                 'lines.linewidth': 2,
                 'lines.markersize': 6,
                 'image.cmap': 'copper',
                 'animation.writer': 'ffmpeg'})

def create_dir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    else:
        pass
    return None

def fdata_travel_distance(t, B, t0):
    res = np.where(t < t0, B*t**2/(2.0*t0), B*t - B*t0/2.0)
    return res

def fit_travel_distance(t, r):
    t0_init = np.max(t) / 4.0
    B_init = np.max(r)**2 / np.max(t)
    p_init = [B_init, t0_init]
    p, _ = curve_fit(fdata_travel_distance, t, r**2, p_init)
    return p[0], p[1]

class PlotHandler:

    def __init__(self, simulation):
        # Simulation attributes
        self.simulation = simulation
        self.time = simulation.time
        self.x = simulation.x
        self.y = simulation.y
        self.simulation()
        self.field, self.is_log = simulation.get_field('density')
        self.xpart = simulation.get_particle_property('x')
        self.ypart = simulation.get_particle_property('y')
        self.particle_property = simulation.get_particle_property('ux')
        # Initialising all figures
        self.fig_time, self.ax_time = subplots()
        self.fig_1dprofile, self.ax_1dprofile = subplots()
        self.fig_flux, self.ax_flux = subplots()
        self.fig_1dmovie, self.ax_1dmovie = subplots()
        self.line_1dmovie = None
        self.fig_2dmovie, self.ax_2dmovie = subplots()
        self.map_2dmovie = None
        div = make_axes_locatable(self.ax_2dmovie)
        self.ax_colorbar_2dmovie = div.append_axes("right", "5%", "5%")
        self.fig_2dmovie.add_axes(self.ax_colorbar_2dmovie)
        self.fig_particle_movie, self.ax_particle_movie = subplots()
        self.dots_particle_movie = None
        div = make_axes_locatable(self.ax_particle_movie)
        self.ax_colorbar_particle_movie = div.append_axes("right", "5%", "5%")
        self.fig_particle_movie.add_axes(self.ax_colorbar_particle_movie)
        self.fig_plume_profiles, self.ax_plume_profiles = subplots()
        self.fig_pdf, self.ax_pdf = subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                                             squeeze=True, gridspec_kw={'wspace':0.0, 'hspace':0.0})
        self.fig_bimodal, self.ax_bimodal = subplots()
        self.fig_dispersion, self.ax_dispersion = subplots()
        # Creating figure directories
        create_dir(os.path.join(simulation.directory, "figures"))
        for subdir in ['evolution', 'vertical_profiles', 'plume_statistics',
                       '2dmovies', '1dmovies', 'particle_movies', 'dispersion', 'flux_budget']:
            create_dir(os.path.join(simulation.directory, 'figures', subdir))

    """
    Setting attributes
    """

    def set_field(self, kw):
        self.field, self.is_log = self.simulation.get_field(kw)
        return None
    
    def set_particle_property(self, kw):
        self.particle_property = self.simulation.get_particle_property(kw)
        return None
    
    """
    Plot titles
    """

    def title_time(self, kw):
        directory = os.path.join(self.simulation.directory, "figures", "evolution")
        plot_title = "evolution_{:s}".format(kw)
        return os.path.join(directory, plot_title)

    def title_1dprofile(self, kw):
        directory = os.path.join(self.simulation.directory, "figures", "vertical_profiles")
        plot_title = "profile_{:s}".format(kw)
        return os.path.join(directory, plot_title)
    
    def title_flux(self, flux_type):
        directory = os.path.join(self.simulation.directory, "figures", "flux_budget")
        plot_title = "flux_budget_{:s}".format(flux_type)
        return os.path.join(directory, plot_title)
    
    def title_plume_profiles(self, kw):
        directory = os.path.join(self.simulation.directory, "figures", "plume_statistics")
        plot_title = "profile_{:s}".format(kw)
        return os.path.join(directory, plot_title)
    
    def title_pdf(self, kw):
        directory = os.path.join(self.simulation.directory, "figures", "plume_statistics")
        plot_title = "distribution_{:s}".format(kw)
        return os.path.join(directory, plot_title)

    def title_bimodal(self):
        directory = os.path.join(self.simulation.directory, "figures", "plume_statistics")
        plot_title = "bimodal_parameters"
        return os.path.join(directory, plot_title)
    
    def title_dispersion(self):
        directory = os.path.join(self.simulation.directory, "figures", "dispersion")
        plot_title = "travel_distance"
        return os.path.join(directory, plot_title)

    def title_2dmovie(self, kw):
        directory = os.path.join(self.simulation.directory, "figures", "2dmovies")
        plot_title = "2D_{:s}.mp4".format(kw)
        return os.path.join(directory, plot_title)
    
    def title_1dmovie(self, kw):
        directory = os.path.join(self.simulation.directory, "figures", "1dmovies")
        plot_title = "1D_{:s}.mp4".format(kw)
        return os.path.join(directory, plot_title)

    def title_particle_movie(self, kw):
        directory = os.path.join(self.simulation.directory, "figures", "particle_movies")
        plot_title = "particles_{:s}.mp4".format(kw)
        return os.path.join(directory, plot_title)

    """
    Stationary plots
    """

    def plot_time_evolution(self, kw):
        self.set_field(kw)
        time_data = self.simulation.time_evolution(kw)
        self.ax_time.clear()
        self.ax_time.plot(self.time, time_data)
        if self.is_log:
            self.ax_time.set_yscale('log')
        else:
            pass
        self.ax_time.vlines(self.time[self.simulation.it_start_steady],
                            self.ax_time.get_ylim()[0],
                            self.ax_time.get_ylim()[1],
                            linestyle='--',
                            color='black')
        self.ax_time.set_xlabel("t [s]")
        self.ax_time.set_ylabel(kw)
        self.fig_time.savefig(self.title_time(kw), dpi=500, bbox_inches="tight")
        return None

    def plot_vertical_profile(self, kw):
        self.set_field(kw)
        field_profile = self.simulation.vertical_profile(kw)
        self.ax_1dprofile.clear()
        self.ax_1dprofile.plot(self.x*1e-6, field_profile)
        if self.is_log:
            self.ax_1dprofile.set_yscale("log")
        else:
            pass
        self.ax_1dprofile.set_xlabel("x [Mm]")
        self.ax_1dprofile.set_ylabel(kw)
        self.fig_1dprofile.savefig(self.title_1dprofile(kw), dpi=500, bbox_inches="tight")
        return None
    
    def plot_flux_budget(self, up_down):
        if up_down:
            # Getting fluxes
            conv_flux_1_u, conv_flux_2_u, conv_flux_3_u, conv_flux_4_u, \
                conv_flux_1_d, conv_flux_2_d, conv_flux_3_d, conv_flux_4_d \
                = self.simulation.get_convective_flux(True)
            conv_flux_u = conv_flux_1_u + conv_flux_2_u + conv_flux_3_u + conv_flux_4_u
            conv_flux_d = conv_flux_1_d + conv_flux_2_d + conv_flux_3_d + conv_flux_4_d
            kin_flux_1_u, kin_flux_2_u, kin_flux_3_u, \
                kin_flux_1_d, kin_flux_2_d, kin_flux_3_d \
                = self.simulation.get_kinetic_flux(True)
            kin_flux_u = kin_flux_1_u + kin_flux_2_u + kin_flux_3_u
            kin_flux_d = kin_flux_1_d + kin_flux_2_d + kin_flux_3_d
            rad_flux_u, rad_flux_d = self.simulation.get_radiative_flux(True)
            tot_flux_u = conv_flux_u + kin_flux_u + rad_flux_u
            tot_flux_d = conv_flux_d + kin_flux_d + rad_flux_d
            # Plotting convective flux
            self.ax_flux.clear()
            self.ax_flux.plot(self.x*1e-6, conv_flux_u, linestyle="-", label="Convective flux", linewidth=3, color="black")
            self.ax_flux.plot(self.x*1e-6, conv_flux_1_u, linestyle='--', label=r"$\Gamma_1 \overline{\rho} \widetilde{u''e''}$")
            self.ax_flux.plot(self.x*1e-6, conv_flux_2_u, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} \widetilde{e}$")
            self.ax_flux.plot(self.x*1e-6, conv_flux_3_u, linestyle='--', label=r"$\widetilde{u_x} \overline{p}$")
            self.ax_flux.plot(self.x*1e-6, conv_flux_4_u, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} \Psi$")
            self.ax_flux.legend()
            self.ax_flux.set_xlabel("x [Mm]")
            self.ax_flux.set_ylabel("Vertical energy flux")
            self.fig_flux.savefig(self.title_flux('convective_up'), dpi=500, bbox_inches="tight")
            self.ax_flux.clear()
            self.ax_flux.plot(self.x*1e-6, conv_flux_d, linestyle="-", label="Convective flux", linewidth=3, color="black")
            self.ax_flux.plot(self.x*1e-6, conv_flux_1_d, linestyle='--', label=r"$\Gamma_1 \overline{\rho} \widetilde{u''e''}$")
            self.ax_flux.plot(self.x*1e-6, conv_flux_2_d, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} \widetilde{e}$")
            self.ax_flux.plot(self.x*1e-6, conv_flux_3_d, linestyle='--', label=r"$\widetilde{u_x} \overline{p}$")
            self.ax_flux.plot(self.x*1e-6, conv_flux_4_d, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} \Psi$")
            self.ax_flux.legend()
            self.ax_flux.set_xlabel("x [Mm]")
            self.ax_flux.set_ylabel("Vertical energy flux")
            self.fig_flux.savefig(self.title_flux('convective_down'), dpi=500, bbox_inches="tight")
            self.ax_flux.clear()
            self.ax_flux.plot(self.x*1e-6, conv_flux_d + conv_flux_u, linestyle="-", label="Convective flux", linewidth=3, color="black")
            self.ax_flux.plot(self.x*1e-6, conv_flux_1_d + conv_flux_1_u, linestyle='--', label=r"$\Gamma_1 \overline{\rho} \widetilde{u''e''}$")
            self.ax_flux.plot(self.x*1e-6, conv_flux_2_d + conv_flux_2_u, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} \widetilde{e}$")
            self.ax_flux.plot(self.x*1e-6, conv_flux_3_d + conv_flux_3_u, linestyle='--', label=r"$\widetilde{u_x} \overline{p}$")
            self.ax_flux.plot(self.x*1e-6, conv_flux_4_d + conv_flux_4_u, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} \Psi$")
            self.ax_flux.legend()
            self.ax_flux.set_xlabel("x [Mm]")
            self.ax_flux.set_ylabel("Vertical energy flux")
            self.fig_flux.savefig(self.title_flux('convective'), dpi=500, bbox_inches="tight")
            # Plotting kinetic flux
            self.ax_flux.clear()
            self.ax_flux.plot(self.x*1e-6, kin_flux_u, linestyle="-", label="Kinetic flux", linewidth=3, color="black")
            self.ax_flux.plot(self.x*1e-6, kin_flux_1_u, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x''u_i''^2}$")
            self.ax_flux.plot(self.x*1e-6, kin_flux_2_u, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} k$")
            self.ax_flux.plot(self.x*1e-6, kin_flux_3_u, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} \left(\widetilde{u_x}^2 + \widetilde{u_y}^2 \right)/2$")
            self.ax_flux.legend()
            self.ax_flux.set_xlabel("x [Mm]")
            self.ax_flux.set_ylabel("Vertical energy flux")
            self.fig_flux.savefig(self.title_flux('kinetic_up'), dpi=500, bbox_inches="tight")
            self.ax_flux.clear()
            self.ax_flux.plot(self.x*1e-6, kin_flux_d, linestyle="-", label="Kinetic flux", linewidth=3, color="black")
            self.ax_flux.plot(self.x*1e-6, kin_flux_1_d, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x''u_i''^2}$")
            self.ax_flux.plot(self.x*1e-6, kin_flux_2_d, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} k$")
            self.ax_flux.plot(self.x*1e-6, kin_flux_3_d, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} \left(\widetilde{u_x}^2 + \widetilde{u_y}^2 \right)/2$")
            self.ax_flux.legend()
            self.ax_flux.set_xlabel("x [Mm]")
            self.ax_flux.set_ylabel("Vertical energy flux")
            self.fig_flux.savefig(self.title_flux('kinetic_down'), dpi=500, bbox_inches="tight")
            self.ax_flux.clear()
            self.ax_flux.plot(self.x*1e-6, kin_flux_u + kin_flux_d, linestyle="-", label="Kinetic flux", linewidth=3, color="black")
            self.ax_flux.plot(self.x*1e-6, kin_flux_1_u + kin_flux_1_d, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x''u_i''^2}$")
            self.ax_flux.plot(self.x*1e-6, kin_flux_2_u + kin_flux_2_d, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} k$")
            self.ax_flux.plot(self.x*1e-6, kin_flux_3_u + kin_flux_3_d, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} \left(\widetilde{u_x}^2 + \widetilde{u_y}^2 \right)/2$")
            self.ax_flux.legend()
            self.ax_flux.set_xlabel("x [Mm]")
            self.ax_flux.set_ylabel("Vertical energy flux")
            self.fig_flux.savefig(self.title_flux('kinetic'), dpi=500, bbox_inches="tight")
            # # Plotting total flux budget
            self.ax_flux.clear()
            self.ax_flux.plot(self.x*1e-6, tot_flux_u, linestyle="-", label="Total flux", linewidth=3, color="black")
            self.ax_flux.plot(self.x*1e-6, conv_flux_u, linestyle="--", label="Convective flux")
            self.ax_flux.plot(self.x*1e-6, rad_flux_u, linestyle="--", label="Radiative flux")
            self.ax_flux.plot(self.x*1e-6, kin_flux_u, linestyle="--", label="Kinetic energy flux")
            self.ax_flux.legend()
            self.ax_flux.set_xlabel("x [Mm]")
            self.ax_flux.set_ylabel("Vertical energy flux")
            self.fig_flux.savefig(self.title_flux('total_up'), dpi=500, bbox_inches="tight")
            self.ax_flux.clear()
            self.ax_flux.plot(self.x*1e-6, tot_flux_d, linestyle="-", label="Total flux", linewidth=3, color="black")
            self.ax_flux.plot(self.x*1e-6, conv_flux_d, linestyle="--", label="Convective flux")
            self.ax_flux.plot(self.x*1e-6, rad_flux_d, linestyle="--", label="Radiative flux")
            self.ax_flux.plot(self.x*1e-6, kin_flux_d, linestyle="--", label="Kinetic energy flux")
            self.ax_flux.legend()
            self.ax_flux.set_xlabel("x [Mm]")
            self.ax_flux.set_ylabel("Vertical energy flux")
            self.fig_flux.savefig(self.title_flux('total_down'), dpi=500, bbox_inches="tight")
            self.ax_flux.clear()
            self.ax_flux.plot(self.x*1e-6, tot_flux_u + tot_flux_d, linestyle="-", label="Total flux", linewidth=3, color="black")
            self.ax_flux.plot(self.x*1e-6, conv_flux_u + conv_flux_d, linestyle="--", label="Convective flux")
            self.ax_flux.plot(self.x*1e-6, rad_flux_u + rad_flux_d, linestyle="--", label="Radiative flux")
            self.ax_flux.plot(self.x*1e-6, kin_flux_u + kin_flux_d, linestyle="--", label="Kinetic energy flux")
            self.ax_flux.legend()
            self.ax_flux.set_xlabel("x [Mm]")
            self.ax_flux.set_ylabel("Vertical energy flux")
            self.fig_flux.savefig(self.title_flux('total'), dpi=500, bbox_inches="tight")
        else:
            # Getting fluxes
            conv_flux_1, conv_flux_2, conv_flux_3, conv_flux_4 = self.simulation.get_convective_flux(False)
            conv_flux = conv_flux_1 + conv_flux_2 + conv_flux_3 + conv_flux_4
            kin_flux_1, kin_flux_2, kin_flux_3 = self.simulation.get_kinetic_flux(False)
            kin_flux = kin_flux_1 + kin_flux_2 + kin_flux_3
            rad_flux = self.simulation.get_radiative_flux(False)
            tot_flux = conv_flux + kin_flux + rad_flux
            # Plotting convective flux
            self.ax_flux.clear()
            self.ax_flux.plot(self.x*1e-6, conv_flux, linestyle="-", label="Convective flux", linewidth=3, color="black")
            self.ax_flux.plot(self.x*1e-6, conv_flux_1, linestyle='--', label=r"$\Gamma_1 \overline{\rho} \widetilde{u''e''}$")
            self.ax_flux.plot(self.x*1e-6, conv_flux_2, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} \widetilde{e}$")
            self.ax_flux.plot(self.x*1e-6, conv_flux_3, linestyle='--', label=r"$\widetilde{u_x} \overline{p}$")
            self.ax_flux.plot(self.x*1e-6, conv_flux_4, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} \Psi$")
            self.ax_flux.legend()
            self.ax_flux.set_xlabel("x [Mm]")
            self.ax_flux.set_ylabel("Vertical energy flux")
            self.fig_flux.savefig(self.title_flux('convective'), dpi=500, bbox_inches="tight")
            # Plotting kinetic flux
            self.ax_flux.clear()
            self.ax_flux.plot(self.x*1e-6, kin_flux, linestyle="-", label="Kinetic flux", linewidth=3, color="black")
            self.ax_flux.plot(self.x*1e-6, kin_flux_1, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x''u_i''^2}$")
            self.ax_flux.plot(self.x*1e-6, kin_flux_2, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} k$")
            self.ax_flux.plot(self.x*1e-6, kin_flux_3, linestyle='--', label=r"$\overline{\rho} \widetilde{u_x} \left(\widetilde{u_x}^2 + \widetilde{u_y}^2 \right)/2$")
            self.ax_flux.legend()
            self.ax_flux.set_xlabel("x [Mm]")
            self.ax_flux.set_ylabel("Vertical energy flux")
            self.fig_flux.savefig(self.title_flux('kinetic'), dpi=500, bbox_inches="tight")
            # # Plotting total flux budget
            self.ax_flux.clear()
            self.ax_flux.plot(self.x*1e-6, tot_flux, linestyle="-", label="Total flux", linewidth=3, color="black")
            self.ax_flux.plot(self.x*1e-6, conv_flux, linestyle="--", label="Convective flux")
            self.ax_flux.plot(self.x*1e-6, rad_flux, linestyle="--", label="Radiative flux")
            self.ax_flux.plot(self.x*1e-6, kin_flux, linestyle="--", label="Kinetic energy flux")
            self.ax_flux.legend()
            self.ax_flux.set_xlabel("x [Mm]")
            self.ax_flux.set_ylabel("Vertical energy flux")
            self.fig_flux.savefig(self.title_flux('total'), dpi=500, bbox_inches="tight")
        return None
    
    def plot_plume_profiles(self, kw):
        self.set_field(kw)
        profile_up, profile_down = self.simulation.plume_profiles(kw)
        self.ax_plume_profiles.clear()
        self.ax_plume_profiles.plot(self.x*1e-6, profile_up, label="Upward plumes")
        self.ax_plume_profiles.plot(self.x*1e-6, profile_down, label="Downward plumes")
        if self.is_log:
            self.ax_plume_profiles.set_yscale("log")
        else:
            pass
        self.ax_plume_profiles.legend()
        self.ax_plume_profiles.set_xlabel("x [Mm]")
        self.ax_plume_profiles.set_ylabel(kw)
        self.fig_plume_profiles.tight_layout()
        self.fig_plume_profiles.savefig(self.title_plume_profiles(kw))
        return None
    
    def plot_pdf(self, kw, x1=0.15, x2=0.35, x3=0.65, x4=0.85, width=0.01):
        variable_up, variable_down = self.simulation.plume_distributions(kw)
        x_up, x_down = self.simulation.plume_distributions('x')
        center_slice = [c * self.simulation.unit_length for c in [x1, x2, x3, x4]]
        width_slice = width * self.simulation.unit_length
        for islice, xslice in enumerate(center_slice):
            nrow = islice // 2
            ncol = islice % 2
            ax = self.ax_pdf[nrow,ncol]
            ax.clear()
            mask_up = (xslice - width_slice/2.0 <= x_up) * (x_up <= xslice + width_slice/2.0)
            mask_down = (xslice - width_slice/2.0 <= x_down) * (x_down <= xslice + width_slice/2.0)
            variable_slice_up = variable_up[mask_up]
            variable_slice_down = variable_down[mask_down]
            variable_slice = np.concatenate((variable_slice_up, variable_slice_down))
            ax.hist(variable_slice, label="Full distribution", bins=100, color=colors[0])
            ax.hist(variable_slice_up, label="Upwards plumes", bins=100, alpha=0.8, color=colors[1])
            ax.hist(variable_slice_down, label="Downward plumes", bins=100, alpha=0.8, color=colors[2])
            ax.vlines([np.mean(variable_slice), np.mean(variable_slice_up), np.mean(variable_slice_down)],
                      ax.get_ylim()[0],
                      ax.get_ylim()[1],
                      linestyle="--",
                      color=[colors[0], colors[1], colors[2]],
                      alpha=1.0)
        self.ax_pdf[0,0].legend()
        self.fig_pdf.savefig(self.title_pdf(kw), dpi=500, bbox_inches="tight")
        return None
    
    def plot_kurtosis_skewness(self):
        x, kurtosis, skewness = self.simulation.velocity_bimodal_parameters()
        self.ax_bimodal.clear()
        self.ax_bimodal.plot(x*1e-6, skewness, label="Skewness")
        self.ax_bimodal.plot(x*1e-6, kurtosis, label="Kurtosis")
        self.ax_bimodal.legend()
        self.ax_bimodal.set_xlabel("x [Mm]")
        self.fig_bimodal.savefig(self.title_bimodal(), dpi=500, bbox_inches="tight")
        return None

    def plot_particle_dispersion(self):
        self.simulation.set_start_steady_state()
        self.ax_dispersion.clear()
        mean_nu_turb = 0.0
        mean_drift_velocity = 0.0
        for idiff in range(self.simulation.number_diffusion_coefs):
            t, r = self.simulation.travel_distances(idiff)
            nu_turb, tau_cutoff = fit_travel_distance(t, r)
            mean_nu_turb += nu_turb / self.simulation.number_diffusion_coefs
            mean_drift_velocity += np.sqrt(nu_turb/(2.0*tau_cutoff)) / self.simulation.number_diffusion_coefs
            self.ax_dispersion.plot(t[1:], fdata_travel_distance(t[1:], nu_turb, tau_cutoff) / 1e6)
            self.ax_dispersion.scatter(t[1:], r[1:]**2 / 1e6, marker='.', label="Time series no. {:d}".format(idiff+1))
        self.ax_dispersion.set_xlabel(r"$\tau$ [s]")
        self.ax_dispersion.set_ylabel(r"$r(\tau)^2$ [km$^2$]")
        self.ax_dispersion.legend()
        title = r"$u_\mathrm{{drift}}$ = {:.0f} m/s, $\nu_\mathrm{{turb}}$ = {:.0f} km$^2$/s"
        self.ax_dispersion.set_title(title.format(mean_drift_velocity, mean_nu_turb / 1e6))
        self.fig_dispersion.savefig(self.title_dispersion(), dpi=500, bbox_inches="tight")
        return None

    """
    Animations
    """

    def init_2dmovie(self, kw):
        self.set_field(kw)
        self.ax_2dmovie.clear()
        if self.is_log:
            min_val = np.percentile(self.field[self.field > 0.0], 0.1)
            max_val = np.percentile(self.field[self.field > 0.0], 99.9)
        else:
            min_val = np.percentile(self.field, 0.1)
            max_val = np.percentile(self.field, 99.9)
        if min_val == max_val:
            min_val *= 0.9
            max_val *= 1.1
        if self.is_log:
            norm = LogNorm(vmin=min_val, vmax=max_val)
        else:
            norm = Normalize(vmin=min_val, vmax=max_val)
        dummy = np.ones((len(self.x), len(self.y)))
        self.map_2dmovie = self.ax_2dmovie.imshow(
                dummy,
                extent=(np.min(self.y)*1e-6, np.max(self.y)*1e-6, np.min(self.x)*1e-6, np.max(self.x)*1e-6),
                aspect="auto",
                origin="lower",
                norm=norm
                )
        self.map_2dmovie.set_clim(min_val, max_val)
        self.fig_2dmovie.colorbar(self.map_2dmovie, cax=self.ax_colorbar_2dmovie)
        self.ax_2dmovie.set_title(r"$t = {:.4f} s$".format(self.time[0]))
        self.ax_2dmovie.set_xlabel("x [Mm]")
        self.ax_2dmovie.set_ylabel("y [Mm]")
        return None
    
    def update_2dmovie(self, frame):
        data = self.field[frame,:,:]
        self.ax_2dmovie.set_title(r"$t = {:.4f} s$".format(self.time[frame]))
        self.map_2dmovie.set_data(data)
        return self.map_2dmovie,

    def animate_2dmovie(self, kw, short=False):
        self.init_2dmovie(kw)
        if short:
            frames = range(len(self.time))[::10]
        else:
            frames = range(len(self.time))
        ani = FuncAnimation(fig=self.fig_2dmovie,
                            func=self.update_2dmovie,
                            frames=frames,
                            interval=5,
                            blit=True)
        ani.save(self.title_2dmovie(kw), fps=10)
        return None

    def init_1dmovie(self, kw):
        self.set_field(kw)
        self.ax_1dmovie.clear()
        if self.is_log:
            min_val = np.percentile(self.field[self.field > 0.0], 0.1)
            max_val = np.percentile(self.field[self.field > 0.0], 99.9)
        else:
            min_val = np.percentile(self.field[~np.isnan(self.field)], 0.1)
            max_val = np.percentile(self.field[~np.isnan(self.field)], 99.9)
        if min_val == max_val:
            min_val *= 0.9
            max_val *= 1.1
        self.ax_1dmovie.set_ylim(min_val, max_val)
        if self.is_log:
            self.ax_1dmovie.set_yscale("log")
        else:
            pass
        dummy = np.ones(len(self.x))
        self.line_1dmovie, = self.ax_1dmovie.plot(self.x*1e-6, dummy, linewidth=3)
        self.ax_1dmovie.set_title(r"$t = {:.4f} s$".format(self.time[0]))
        self.ax_1dmovie.set_xlabel("x [Mm]")
        self.ax_1dmovie.set_ylabel(kw)
        return None
    
    def update_1dmovie(self, frame):
        data = np.mean(self.field[frame,:,:], axis=1)
        self.ax_1dmovie.set_title(r"$t = {:.4f} s$".format(self.time[frame]))
        self.line_1dmovie.set_ydata(data)
        return self.line_1dmovie,

    def animate_1dmovie(self, kw, short=False):
        self.init_1dmovie(kw)
        if short:
            frames = range(len(self.time))[::10]
        else:
            frames = range(len(self.time))
        ani = FuncAnimation(fig=self.fig_1dmovie,
                            func=self.update_1dmovie,
                            frames=frames,
                            interval=5,
                            blit=True)
        ani.save(self.title_1dmovie(kw), fps=10)
        return None
    
    def init_particle_movie(self, kw):
        self.set_particle_property(kw)
        self.ax_particle_movie.clear()
        min_val = np.percentile(self.particle_property, 0.1)
        max_val = np.percentile(self.particle_property, 99.9)
        if min_val==max_val:
            min_val *= 0.9
            max_val *= 1.1
        dummy = np.ones(self.xpart.shape[1])
        self.dots_particle_movie = self.ax_particle_movie.scatter(self.ypart[0,:]*1e-6,
                                                                  self.xpart[0,:]*1e-6,
                                                                  c=dummy,
                                                                  marker="o",
                                                                  s=10)
        self.dots_particle_movie.set_clim(min_val, max_val)
        self.fig_particle_movie.colorbar(self.dots_particle_movie, cax=self.ax_colorbar_particle_movie)
        self.ax_particle_movie.set_title(r"$t = {:.4f} s$".format(self.time[0]))
        self.ax_particle_movie.set_xlabel("x [Mm]")
        self.ax_particle_movie.set_ylabel("y [Mm]")
        return None
    
    def update_particle_movie(self, frame):
        offsets = np.array([[self.ypart[frame,i]*1e-6, self.xpart[frame,i]*1e-6] for i in range(self.xpart.shape[1])])
        self.dots_particle_movie.set_offsets(offsets)
        self.dots_particle_movie.set_array(self.particle_property[frame,:])
        self.ax_particle_movie.set_title(r"$t = {:.4f} s$".format(self.time[frame]))
        return self.dots_particle_movie,

    def animate_particle_movie(self, kw, short=False):
        self.init_particle_movie(kw)
        if short:
            frames = range(len(self.time))[::10]
        else:
            frames = range(len(self.time))
        ani = FuncAnimation(fig=self.fig_particle_movie,
                            func=self.update_particle_movie,
                            frames=frames,
                            interval=5,
                            blit=True)
        ani.save(self.title_particle_movie(kw), fps=10)
        return None
