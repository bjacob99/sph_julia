#!/usr/bin/env python

import numpy as np
import os
import re
import toml
from scipy.interpolate import RectBivariateSpline
from progress.bar import ChargingBar

GRAVITY = 275.0 # Gravitational acceleration at the Sun's surface, in m/s^2
SOUND_SPEED = 5e3 # Sound speed at the Sun's surface, in m/s
DENSITY_REF = 3e-4 # Typical gas density at the Sun's photosphere, in kg/m^3
DEFAULT_MAX_TIME_DIFFUSION = 300.0 # Default maximum time increment for measurement of diffusion coefficients, in s

"""
Redimensioning
"""

def redim_length(gravity_adim):
    return gravity_adim * SOUND_SPEED**2 / GRAVITY

def redim_time(gravity_adim):
    return gravity_adim * SOUND_SPEED / GRAVITY

def redim_energy(gamma1):
    return SOUND_SPEED**2 / gamma1 / (gamma1 - 1.0)

"""
Read parameters from configuration file
"""

def read_time_step(filename):
    with open(filename, 'r') as f:
        config = toml.load(f)
    return config["time"]["time_step"]

def read_vertical_size(filename):
    return 1.0

def read_horizontal_size(filename):
    with open(filename, 'r') as f:
        config = toml.load(f)
    return config["spatial"]["aspect_ratio"]

def read_number_particles(filename):
    with open(filename, 'r') as f:
        config = toml.load(f)
    return config["particles"]["number_particles"]

def read_first_adiabatic_exponent(filename):
    with open(filename, 'r') as f:
        config = toml.load(f)
    return config["physics"]["first_adiabatic_index"]

def read_gravity(filename):
    with open(filename, 'r') as f:
        config = toml.load(f)
    return config["physics"]["gravity"]

def read_radiative_diffusivity(filename):
    with open(filename, 'r') as f:
        config = toml.load(f)
    return config["physics"]["radiative_diffusivity"]

def read_cell_number(filename):
    with open(filename, 'r') as f:
        config = toml.load(f)
    return config["spatial"]["vertical_cells"], config["spatial"]["horizontal_cells"]

"""
Getting all simulation output files
"""

def find_all_time_indices(directory):
    pattern = r"mean_(\d+)\.dat"
    all_files = os.listdir(os.path.join(directory, "raw_output"))
    list_time_indices = []
    for f in all_files:
        if re.match(pattern, f):
            list_time_indices.append(re.search(pattern, f).group(1))
    return list_time_indices

def get_time_axis(list_time_indices, dt):
    its = np.array(sorted([int(it) for it in list_time_indices]))
    return dt * its

def get_all_mean_files(directory, list_time_indices):
    list_time_indices_integer = [int(str_int) for str_int in list_time_indices]
    filenames = ["mean_{:s}.dat".format(str_int) for str_int in list_time_indices]
    _, filenames_sorted = zip(*sorted(zip(list_time_indices_integer, filenames)))
    return [os.path.join(directory, "raw_output", filename) for filename in filenames_sorted]

def get_all_gradient_files(directory, list_time_indices):
    list_time_indices_integer = [int(str_int) for str_int in list_time_indices]
    filenames = ["gradx_{:s}.dat".format(str_int) for str_int in list_time_indices]
    _, filenames_sorted = zip(*sorted(zip(list_time_indices_integer, filenames)))
    return [os.path.join(directory, "raw_output", filename) for filename in filenames_sorted]

def get_all_particle_files(directory, list_time_indices):
    list_time_indices_integer = [int(str_int) for str_int in list_time_indices]
    filenames = ["particles_{:s}.dat".format(str_int) for str_int in list_time_indices]
    _, filenames_sorted = zip(*sorted(zip(list_time_indices_integer, filenames)))
    return [os.path.join(directory, "raw_output", filename) for filename in filenames_sorted]

"""
Bi-modal gaussian parameters retrieval
"""

def kurtosis_skewness(mu1, mu2, mu, sigma1, sigma2):
    p = (mu - mu2) / (mu1 - mu2)
    delta1 = mu1 - mu
    delta2 = mu2 - mu
    sigma = np.sqrt(p*(sigma1**2 + delta1**2) + (1-p)*(sigma2**2 + delta2**2))
    mu3 = p*(3.0*delta1*sigma1**2 + delta1**3) + (1-p)*(3.0*delta2*sigma2**2 + delta2**3)
    mu4 = p*(6.0*delta1**2*sigma1**2 + delta1**4) + (1-p)*(6.0*delta2**2*sigma2**2 + delta2**4)
    skewness = mu3 / sigma**3
    kurtosis = mu4 / sigma**4
    return kurtosis, skewness

"""
Miscalleneous
"""

def create_dir(str):
    if not os.path.isdir(str):
        os.mkdir(str)
    else:
        pass
    return None

def get_grid_coordinates(filename):
    var_names = get_variable_names(filename)
    ivar_x = var_names.index('x')
    ivar_y = var_names.index('y')
    x_all, y_all = np.loadtxt(filename, unpack=True, usecols=(ivar_x, ivar_y))
    x_unique = np.unique(x_all)
    y_unique = np.unique(y_all)
    return x_unique, y_unique

def get_nrows(filename):
    data = np.loadtxt(filename)
    return data.shape[0]

def get_variable_names(filename):
    with open(filename, 'r') as f:
        line = f.readline()
    names = line.split()
    return names[1:]

def find_start_steady_state(temps, variable):
    nt = len(temps)
    if nt <= 100:
        it_start = 0
    else:
        nt_average = nt // 10
        window = np.ones(nt_average) / nt_average
        temps_average = np.convolve(temps, window, mode='valid')
        variable_average = np.transpose(np.array([np.convolve(variable[:,ix], window, mode='valid') for ix in range(variable.shape[1])]))
        variable_steady = variable_average[-1,:]
        it = len(temps_average) - 1
        stop = False
        while (not stop) and it > 0:
            diff_var = np.mean(np.abs((variable_average[it,:] - variable_steady) / variable_steady))
            if diff_var >= 0.25:
                stop = True
            else:
                it -= 1
        it_start = np.argmin(np.abs(temps - temps_average[it]))
    return it_start

def remove_trailing_zeros(x):
    i = len(x) - 1
    while (i != 0) and (x[i] == 0.0):
        i -= 1
    return x[:i]

"""
Simulation class
"""

class Simulation:

    def __init__(self, relative_directory, init_time, max_time_diffusion):
        directory = os.path.abspath(relative_directory)
        config_file = os.path.join(directory, "config.toml")
        dt = read_time_step(config_file)
        Lx = read_vertical_size(config_file)
        Ly = read_horizontal_size(config_file)
        nparticles = read_number_particles(config_file)
        gamma1 = read_first_adiabatic_exponent(config_file)
        grav = read_gravity(config_file)
        kappa = read_radiative_diffusivity(config_file)
        ncell_v, ncell_h = read_cell_number(config_file)
        time_indices = find_all_time_indices(directory)
        temps = get_time_axis(time_indices, dt)
        mean_files = get_all_mean_files(directory, time_indices)
        grad_files = get_all_gradient_files(directory, time_indices)
        particle_files = get_all_particle_files(directory, time_indices)
        nparticles_output = get_nrows(particle_files[0])
        var_names = get_variable_names(mean_files[0])
        var_names_grad = get_variable_names(grad_files[0])
        x, y = get_grid_coordinates(mean_files[0])
        unit_length = redim_length(grav)
        unit_time = redim_time(grav)
        unit_energy = redim_energy(gamma1)
        # Initialisation of mean variable arrays
        size_mean_variable = (len(temps), len(x), len(y))
        self.density = np.empty(size_mean_variable)
        self.velocity_x = np.empty(size_mean_variable)
        self.velocity_y = np.empty(size_mean_variable)
        self.energy = np.empty(size_mean_variable)
        self.omega = np.empty(size_mean_variable)
        self.kinetic_energy = np.empty(size_mean_variable)
        self.convective_flux = np.empty(size_mean_variable)
        self.energy_variance = np.empty(size_mean_variable)
        self.radiative_flux = np.empty(size_mean_variable)
        self.kinetic_flux = np.empty(size_mean_variable)
        # Initialisation of particles arrays
        size_particle_properties = (len(temps), nparticles_output)
        self.x_part = np.empty(size_particle_properties)
        self.y_part = np.empty(size_particle_properties)
        self.ux_part = np.empty(size_particle_properties)
        self.uy_part = np.empty(size_particle_properties)
        self.uz_part = np.empty(size_particle_properties)
        self.e_part = np.empty(size_particle_properties)
        self.omega_part = np.empty(size_particle_properties)
        # Other attributes
        self.directory = directory
        self.dt = dt * unit_time
        self.nparticles = nparticles
        self.var_names = var_names
        self.var_names_grad = var_names_grad
        self.x = x * unit_length
        self.neval_x = len(x)
        self.y = y * unit_length
        self.neval_y = len(y)
        self.time = temps * unit_time
        self.ntime = len(temps)
        self.Lx = Lx * unit_length
        self.Ly = Ly * unit_length
        self.nparticles_output = nparticles_output
        self.gamma1 = gamma1
        self.grav = GRAVITY
        self.kappa = kappa * unit_length * SOUND_SPEED
        self.ncell_v, self.ncell_h = ncell_v, ncell_h
        self.mean_files = mean_files
        self.grad_files = grad_files
        self.particle_files = particle_files
        self.is_read = False
        self.unit_length = unit_length
        self.unit_time = unit_time
        self.unit_energy = unit_energy
        if init_time:
            if init_time >= np.max(self.time):
                self.it_start_steady = None
            else:
                self.it_start_steady = np.argmin(np.abs(self.time - init_time))
        else:
            self.it_start_steady = None
        if max_time_diffusion:
            if (max_time_diffusion < 0.0) or (max_time_diffusion >= self.time[-1]):
                self.max_time_diffusion = DEFAULT_MAX_TIME_DIFFUSION
            else:
                self.max_time_diffusion = max_time_diffusion
        else:
            self.max_time_diffusion = DEFAULT_MAX_TIME_DIFFUSION
        self.number_diffusion_coefs = None

    def __call__(self, only_particles=False):
        if not self.is_read:
            print("Reading output files from directory:", self.directory, "...")
            if not only_particles:
                self.read_all_mean_files()
                self.read_all_grad_files()
            self.read_all_particle_files()
            self.is_read = True
        else:
            pass
        return None
    
    """
    Reading data
    """

    def read_all_mean_files(self):
        with ChargingBar("Reading mean files...", max=self.ntime) as bar:
            for it, mean_file in zip(range(self.ntime), self.mean_files):
                means = np.loadtxt(mean_file, unpack=True)
                self.density[it,:,:] = self.read_column(means, 'rho') * DENSITY_REF
                self.velocity_x[it,:,:] = self.read_column(means, 'ux') * SOUND_SPEED
                self.velocity_y[it,:,:] = self.read_column(means, 'uy') * SOUND_SPEED
                self.energy[it,:,:] = self.read_column(means, 'e') * self.unit_energy
                self.omega[it,:,:] = self.read_column(means, 'omega') / self.unit_time
                self.kinetic_energy[it,:,:] = self.read_column(means, 'k') * SOUND_SPEED**2
                self.convective_flux[it,:,:] = self.read_column(means, 'Fex') * SOUND_SPEED * self.unit_energy
                self.energy_variance[it,:,:] = self.read_column(means, 'ke') * self.unit_energy**2
                self.kinetic_flux[it,:,:] = self.read_column(means, 'fkin') * SOUND_SPEED**3
                bar.next()
        return None
    
    def read_all_grad_files(self):
        with ChargingBar("Reading gradient files...", max=self.ntime) as bar:
            for it, grad_file in zip(range(self.ntime), self.grad_files):
                grad = np.loadtxt(grad_file, unpack=True)
                grad_e = self.read_column(grad, 'e', is_grad=True) * self.unit_energy / self.unit_length
                self.radiative_flux[it,:,:] = -self.kappa * grad_e
                bar.next()
        return None
    
    def read_all_particle_files(self):
        with ChargingBar("Reading particle files...", max=self.ntime) as bar:
            for it, particle_file in zip(range(self.ntime), self.particle_files):
                x, y, ux, uy, uz, e, omega = np.loadtxt(particle_file, unpack=True)
                self.x_part[it,:] = x * self.unit_length
                self.y_part[it,:] = y * self.unit_length
                self.ux_part[it,:] = ux * SOUND_SPEED
                self.uy_part[it,:] = uy * SOUND_SPEED
                self.uz_part[it,:] = uz * SOUND_SPEED
                self.e_part[it,:] = e * self.unit_energy
                self.omega_part[it,:] = omega / self.unit_time
                bar.next()
        return None
    
    def read_column(self, array, name, is_grad=False):
        if is_grad:
            namelist = self.var_names_grad
        else:
            namelist = self.var_names
        if name in namelist:
            return np.reshape(array[namelist.index(name)], (self.neval_x, self.neval_y), order='F')
        else:
            return np.ones((self.neval_x, self.neval_y))

    def set_start_steady_state(self):
        if self.it_start_steady is None:
            it_start = 0
            for kw in ["kinetic_energy", "energy_variance", "convective_flux"]:
                field, _ = self.get_field(kw)
                mean_field = np.mean(field, axis=2)
                it = find_start_steady_state(self.time, mean_field)
                if it > it_start:
                    it_start = it
            self.it_start_steady = it_start
            start = self.time[self.it_start_steady]
            end = self.time[-1]
            self.number_diffusion_coefs = int((end - start) // self.max_time_diffusion) + 1
        else:
            pass
        return None
    
    """
    Getting attributes
    """

    def get_field(self, kw):
        self()
        match kw:
            case "density":
                field = self.density
                is_log = False
            case "velocity_x":
                field = self.velocity_x
                is_log = False
            case "velocity_y":
                field = self.velocity_y
                is_log = False
            case "internal_energy":
                field = self.energy
                is_log = False
            case "turbulent_frequency":
                field = self.omega
                is_log = False
            case "gas_pressure":
                field = (self.gamma1-1.0) * self.get_field("density")[0] * self.get_field("internal_energy")[0]
                is_log = False
            case "gravity_potential":
                _, x, _ = np.meshgrid(self.time, self.x, self.y, indexing='ij')
                field = -self.grav * x
                is_log = False
            case "kinetic_energy":
                field = self.kinetic_energy
                is_log = True
            case "energy_variance":
                field = self.energy_variance
                is_log = True
            case "convective_flux":
                field = self.convective_flux
                is_log = False
            case "radiative_flux":
                field = self.radiative_flux
                is_log = False
            case "kinetic_flux":
                field = self.kinetic_flux
                is_log = False
            case "nabla":
                rho_profile = self.vertical_profile("density")
                energy_profile = self.vertical_profile("internal_energy")
                pgas_profile = rho_profile * energy_profile
                drho_dx = np.gradient(rho_profile, self.x)
                de_dx = np.gradient(energy_profile, self.x)
                dp_dx = energy_profile * drho_dx + rho_profile * de_dx
                nabla_profile = (de_dx / energy_profile) / (dp_dx / pgas_profile)
                _, field, _ = np.meshgrid(self.time, nabla_profile, self.neval_y, indexing='ij')
                is_log = False
            case _:
                raise ValueError("Keyword is wrong (got {:s})".format(kw))
        return field, is_log
    
    def get_particle_property(self, kw):
        self(only_particles=True)
        match kw:
            case "x":
                field = self.x_part
            case "y":
                field = self.y_part
            case "ux":
                field = self.ux_part
            case "uy":
                field = self.uy_part
            case "uz":
                field = self.uz_part
            case "e":
                field = self.e_part
            case "omega":
                field = self.omega_part
            case _:
                raise ValueError("Keyword is wrong (got {:s})".format(kw))
        return field
    
    def get_convective_flux(self, up_down):
        if up_down:
            rho_u, rho_d = self.plume_profiles("density")
            ux_u, ux_d = self.plume_profiles("velocity_x")
            e_u, e_d = self.plume_profiles("internal_energy")
            pgas_u, pgas_d = self.plume_profiles("gas_pressure")
            psi_u, psi_d = self.plume_profiles("gravity_potential")
            fex_u, fex_d = self.plume_profiles("convective_flux")
            conv_flux_1_u = rho_u * fex_u * self.gamma1
            conv_flux_2_u = rho_u * ux_u * e_u
            conv_flux_3_u = ux_u * pgas_u
            conv_flux_4_u = rho_u * ux_u * psi_u
            conv_flux_1_d = rho_d * fex_d * self.gamma1
            conv_flux_2_d = rho_d * ux_d * e_d
            conv_flux_3_d = ux_d * pgas_d
            conv_flux_4_d = rho_d * ux_d * psi_d
            return conv_flux_1_u, conv_flux_2_u, conv_flux_3_u, conv_flux_4_u, \
                conv_flux_1_d, conv_flux_2_d, conv_flux_3_d, conv_flux_4_d
        else:
            rho, _ = self.get_field("density")
            ux, _ = self.get_field("velocity_x")
            e, _ = self.get_field("internal_energy")
            pgas, _ = self.get_field("gas_pressure")
            psi, _ = self.get_field("gravity_potential")
            fex, _ = self.get_field("convective_flux")
            conv_flux_1 = self.compute_vertical_profile(rho * fex * self.gamma1)
            conv_flux_2 = self.compute_vertical_profile(rho * ux * e)
            conv_flux_3 = self.compute_vertical_profile(ux * pgas)
            conv_flux_4 = self.compute_vertical_profile(rho * ux * psi)
            return conv_flux_1, conv_flux_2, conv_flux_3, conv_flux_4

    def get_kinetic_flux(self, up_down):
        if up_down:
            rho_u, rho_d = self.plume_profiles("density")
            ux_u, ux_d = self.plume_profiles("velocity_x")
            uy_u, uy_d = self.plume_profiles("velocity_y")
            k_u, k_d = self.plume_profiles("kinetic_energy")
            fkin_u, fkin_d = self.plume_profiles("kinetic_flux")
            kin_flux_1_u = rho_u * fkin_u
            kin_flux_2_u = 5.0/3.0 * rho_u * ux_u * k_u
            kin_flux_3_u = rho_u * ux_u * 0.5*(ux_u**2 + uy_u**2)
            kin_flux_1_d = rho_d * fkin_d
            kin_flux_2_d = 5.0/3.0 * rho_d * ux_d * k_d
            kin_flux_3_d = rho_d * ux_d * 0.5*(ux_d**2 + uy_d**2)
            return kin_flux_1_u, kin_flux_2_u, kin_flux_3_u, kin_flux_1_d, kin_flux_2_d, kin_flux_3_d
        else:
            rho, _ = self.get_field("density")
            ux, _ = self.get_field("velocity_x")
            uy, _ = self.get_field("velocity_y")
            k, _ = self.get_field("kinetic_energy")
            fkin, _ = self.get_field("kinetic_flux")
            kin_flux_1 = self.compute_vertical_profile(rho * fkin)
            kin_flux_2 = self.compute_vertical_profile(5.0/3.0 * rho * ux * k)
            kin_flux_3 = self.compute_vertical_profile(rho * ux * 0.5*(ux**2 + uy**2))
            return kin_flux_1, kin_flux_2, kin_flux_3
    
    def get_radiative_flux(self, up_down):
        if up_down:
            rad_flux_u, rad_flux_d = self.plume_profiles("radiative_flux")
            rho_u, rho_d = self.plume_profiles("density")
            return rad_flux_u * rho_u, rad_flux_d * rho_d
        else:
            return self.vertical_profile('radiative_flux') / 20000.0

    def uxmean_interpolation(self):
        ux_steady_map = self.steady_map("velocity_x")
        return RectBivariateSpline(self.x, self.y, ux_steady_map, bbox=[0.0, self.Lx, 0.0, self.Ly])
    
    """
    Operations on the mean flow variables
    """

    def vertical_profile(self, kw):
        self()
        self.set_start_steady_state()
        field, _ = self.get_field(kw)
        return np.mean(field[self.it_start_steady:,:,:], axis=(0,2))
    
    def compute_vertical_profile(self, data):
        self()
        self.set_start_steady_state()
        return np.mean(data[self.it_start_steady:,:,:], axis=(0,2))

    def steady_map(self, kw):
        self()
        self.set_start_steady_state()
        field, _ = self.get_field(kw)
        return np.mean(field[self.it_start_steady:,:,:], axis=0)

    def time_evolution(self, kw):
        self()
        self.set_start_steady_state()
        field, _ = self.get_field(kw)
        return np.mean(field, axis=(1,2))
    
    def plume_profiles(self, kw):
        self()
        mapping = self.steady_map(kw)
        mapping_ux = self.steady_map("velocity_x")
        mean_up, mean_down = np.empty((2, len(self.x)))
        for ix in range(len(self.x)):
            mask_up = mapping_ux[ix,:] >= 0.0
            mask_down = mapping_ux[ix,:] < 0.0
            mean_up[ix] = np.mean(mapping[ix,mask_up])
            mean_down[ix] = np.mean(mapping[ix,mask_down])
        return mean_up, mean_down
    
    def plume_distributions(self, kw):
        self(only_particles=True)
        uxmean_func = self.uxmean_interpolation()
        variable = (self.get_particle_property(kw)[self.it_start_steady:,:]).flatten()
        xpart = (self.get_particle_property('x')[self.it_start_steady:,:]).flatten()
        ypart = (self.get_particle_property('y')[self.it_start_steady:,:]).flatten()
        is_upwards = np.ones(len(xpart), dtype="bool")
        for i in range(len(xpart)):
            if uxmean_func(xpart[i], ypart[i]) < 0.0:
                is_upwards[i] = False
        return variable[is_upwards], variable[np.logical_not(is_upwards)]

    def velocity_bimodal_parameters(self, width=0.01):
        ux_up, ux_down = self.plume_distributions('ux')
        x_up, x_down = self.plume_distributions('x')
        center_slice = np.arange(width/2.0, 1.0-width/2.0, step=width) * self.unit_length
        width_slice = width * self.unit_length
        kurtosis, skewness = np.zeros((2, len(center_slice)))
        for islice, xslice in enumerate(center_slice):
            mask_up = (xslice - width_slice/2.0 <= x_up) * (x_up <= xslice + width_slice/2.0)
            mask_down = (xslice - width_slice/2.0 <= x_down) * (x_down <= xslice + width_slice/2.0)
            ux_slice_up = ux_up[mask_up]
            ux_slice_down = ux_down[mask_down]
            ux_slice = np.concatenate((ux_slice_up, ux_slice_down))
            ux_mean_up = np.mean(ux_slice_up)
            ux_std_up = np.std(ux_slice_up)
            ux_mean_down = np.mean(ux_slice_down)
            ux_std_down = np.std(ux_slice_down)
            ux_mean = np.mean(ux_slice)
            kurtosis[islice], skewness[islice] = kurtosis_skewness(ux_mean_up, ux_mean_down, ux_mean, ux_std_up, ux_std_down)
        return center_slice, kurtosis, skewness
    
    """
    Diffusion coefficients measurement
    """

    def cut_time_series(self, data, idiff):
        error_message = "Requesting invalid chunk of simulation (must be positive and smaller than {:d}, got {:d})"
        assert (0 <= idiff) and (idiff < self.number_diffusion_coefs), error_message.format(self.number_diffusion_coefs, idiff)
        start_time = self.time[self.it_start_steady] + idiff * self.max_time_diffusion
        end_time = self.time[self.it_start_steady] + (idiff+1) * self.max_time_diffusion
        start_index = np.argmin(np.abs(self.time - start_time))
        end_index = np.argmin(np.abs(self.time - end_time))
        return data[start_index:end_index]
    
    def particle_subtraj(self, ipart, idiff):
        max_ipart = self.nparticles_output - 1
        assert (ipart <= max_ipart), "Requesting invalid particle id (maximum is {:d}, got {:d})".format(max_ipart, ipart)
        x = self.get_particle_property("x")[:,ipart]
        y = self.get_particle_property("y")[:,ipart]
        xtraj = self.cut_time_series(x, idiff)
        ytraj = self.cut_time_series(y, idiff)
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
    
    def travel_distances(self, idiff):
        self.set_start_steady_state()
        ntime = int(self.max_time_diffusion // self.dt)
        distances = np.zeros(ntime)
        count = np.zeros(ntime, dtype='int')
        for ipart in range(self.nparticles_output):
            x, y = self.particle_subtraj(ipart, idiff)
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