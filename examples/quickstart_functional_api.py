#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quickstart guide functional api
===============================

This example is designed as quickstart guide for the functional api of
:py:mod:`pylawr`.
"""
# %%
# Introduction
# ------------
# In the following, we will use the functional api to process netCDF data from
# the ``HHG`` X-band weather radar. These steps are almost the same as for the
# low-level api.

# %%
# The following processing steps are:
# 1. Read in data with pre-defined data handler for netcdf data
# 2. Remove noise 3. Filter out clutter
# 4. Interpolate missing values
# 5. Plot the data

import pylawr.functions.input as input_funcs
import pylawr.functions.transform as transform_funcs
import pylawr.functions.plot as plot_funcs

from pylawr.grid import PolarGrid

# %%
# 1. Read in data
# ---------------
# To read in the ascii data, we will use a pre-defined function to read in
# ``HHG`` X-band weather radar netCDF data. This functions takes a file name
# and returns read reflectivity. For this, we need to open a file handler.
# We need to set the grid manually, because we cannot extract the grid from
# given netCDF data at the moment.

file_path = '../tests/data/lawr_l0_example.nc'
reflectivity, _ = input_funcs.read_lawr_nc_level0(file_path)
grid = PolarGrid()
reflectivity = reflectivity.lawr.set_grid_coordinates(grid)

# %%
print(reflectivity)

# %%
print(grid)

# %%
# 2. Remove noise
# ---------------
# As next step, we want to remove existing noise from reflectivity.
# For this, we will use ``remove_noise`` from ``pylawr.functions.transform``,
# which fits a ``NoiseRemover`` and uses this remover to subtract the noise.

reflectivity, noise_remover = transform_funcs.remove_noise(reflectivity)

print(reflectivity)
print('Determined noiselevel: {0:.3e}'.format(noise_remover.noiselevel))

# %%
# We notice that the logarithmic reflectivity was translated into a linear
# reflectivity with our noise remover. Thus, we need to retranslate our
# linear reflectivity to a logarithmic reflectivity.

reflectivity = reflectivity.lawr.to_dbz()

# %%
# 3. Filter out clutter
# ---------------------
# After we substracted a determined noise level from our reflectivity field,
# we want to remove clutter. Based on multiple application gradient-based
# filters andoptical filters, ``remove_clutter`` removes clutter from our
# reflectivity.

reflectivity, cluttermap = transform_funcs.remove_clutter_lawr(reflectivity)

print(reflectivity)
print(cluttermap)

# %%
# 4. Interpolate missing values
# -----------------------------
# We can see that there are missing values within our reflectivity array,
# which need to be interpolated. For this, we will use ``interpolate_missing``
# from our filter functions. This remaps existing values to missing values
# such that no missing value remains. If a criterion is exceeded, some
# missing values are filled with no rain to avoid interpolation artifacts.

reflectivity, remapper = transform_funcs.interpolate_missing(reflectivity)

print(reflectivity)
print(remapper)

# %%
# 5. Plot the data
# ----------------
# The missing values are interpolated with a nearest neighbor interpolation,
# where the five nearest neighbors are averaged. After we interpolated our
# data, we can plot the data.

# %%
# To plot the data, we need to translate our reflectivity to rain rates.
# This translation is given by a Z-R relationship, where the scale and
# ploynomial degree factor can be set. All values without rain are set
# to NaN, because they will be plotted transparent.

rain_rate = reflectivity.lawr.to_z().lawr.zr_convert(a=256, b=1.42)
rain_rate = rain_rate.where(rain_rate >= 0.1)
rain_rate = rain_rate.lawr.set_grid_coordinates(grid)

# %%
# After we translated the rain rate, we can use this array as input array
# to ``plot_rain_clutter`` function to plot this rain rate together with
# our cluttermap.
plotter = plot_funcs.plot_rain_clutter(rain_rate,
                                       cluttermap=cluttermap,
                                       plot_path='/tmp/quickstart_functional_api.png')
plotter.show()

# %%
# This creates a default plot with an information header, a map with the
# rainfall rates and a corresponding colorbar.

# %%
# After we plotted the rain rate and cluttermap, we finished this quickstart
# tutorial. If you want to get a deeper insight into this radar software,
# you can also check-out the quickstart tutorial for our low-level api.
