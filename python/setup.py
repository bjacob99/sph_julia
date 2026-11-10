#!/usr/bin/env python3

from setuptools import setup

setup(name='lagrangian_post_treatment',
      version='1.0',
      description='Post-treatment routines for the Lagrangian Stochastic Convection code',
      author='Jordan Philidet',
      author_email='jordan.philidet@obspm.fr',
      scripts=['plot_1dprofiles.py', 'plot_2dmovies.py', 'plot_plumes.py', 'compare_simulations.py',
               'plot_particles.py', 'compare_simulations.py', 'plot_all.py'],
      package_dir={
          'lagrangian_post_treatment': 'src/lagrangian_post_treatment'},
      packages=['lagrangian_post_treatment'])
