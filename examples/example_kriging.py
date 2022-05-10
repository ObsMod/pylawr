#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kriging example
===============

This example compares nearest neighbor interpolation with
kriging interpolation.
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# Import packages
# ---------------
from pylawr.remap import kernel as k_ops
from pylawr.remap import OrdinaryKriging, NearestNeighbor
from pylawr.plot.layer import ColorbarLayer
from pylawr.grid import PolarGrid

import pylawr.functions.input as input_funcs
import pylawr.functions.transform as transform_funcs
import pylawr.functions.fit as fit_funcs
import pylawr.functions.plot as plot_funcs

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# %%
rnd = np.random.RandomState(42)

# %%
file_path = '../tests/data/example_lawr.nc'
lawr_raw, _ = input_funcs.read_lawr_nc_new(file_path)
lawr_grid = PolarGrid()
lawr_raw = lawr_raw.lawr.set_grid_coordinates(lawr_grid)
lawr_denoised, noise_remover = transform_funcs.remove_noise(lawr_raw)
lawr_denoised = lawr_denoised.lawr.to_dbz()
lawr_denoised = lawr_denoised.lawr.set_grid_coordinates(lawr_grid)

# Remove clutter
lawr_decluttered, _ = transform_funcs.remove_clutter_lawr(
    lawr_denoised
)
lawr_decluttered, _ = transform_funcs.remove_clutter_lawr(
    lawr_decluttered
)

# %%
lawr_decluttered[:, 200:205, 160:180] = np.nan
lawr_decluttered = lawr_decluttered.lawr.set_grid_coordinates(lawr_grid)

# %%
plotter = plot_funcs.plot_reflectivity(
    lawr_decluttered, plot_path='/tmp/lawr.png', title='Clutter holes',
)
plotter.show()

# %%
# Nearest neighbor interpolation
# ------------------------------
nearest_neighbor = NearestNeighbor(10, max_dist=1500)
refl_nearest, _ = transform_funcs.interpolate_missing(
    lawr_decluttered, remapper=nearest_neighbor
)

# %%
plotter = plot_funcs.plot_reflectivity(
    refl_nearest, plot_path='/tmp/lawr.png',
    title='{0:s}'.format(str(nearest_neighbor)),
)
plotter.show()

# %%
# Ordinary kriging with RBF kernel
# --------------------------------
# First step: define kernel and kriging instance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
rbf_data_kernel = k_ops.gaussian_rbf(length_scale=14000, stddev=5.1)
noise_kernel = k_ops.WhiteNoise(rbf_data_kernel.placeholders[0], noise_level=5)
rbf_kernel = rbf_data_kernel + noise_kernel

kriging_rbf = OrdinaryKriging(rbf_kernel, n_neighbors=10)

# %%
# Second step: build variogram and plot kernel function into variogram
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin_mean, bin_vario = fit_funcs.sample_variogram(lawr_decluttered)

# %%
# Parameter tuning
# ^^^^^^^^^^^^^^^^

# Observation noise / nugget
noise_kernel.noise_level = 0.3
# Covariance / sill
rbf_kernel.params[0].value = np.sqrt(55)
# Length scale / range / decorrelation length
rbf_kernel.params[1].value = 2900
# Maximum distance (1.5 * decorrelation)
kriging_rbf.max_dist = 1.5 * rbf_kernel.params[1].value

# %%
dist_linspace = np.linspace(0, 40000, 10000)
kernel_semivar = rbf_kernel.diag(
    np.zeros_like(dist_linspace)) - rbf_data_kernel.diag(dist_linspace)

# %%
fig, ax = plt.subplots()
ax.plot(dist_linspace, kernel_semivar, c='k', label='Fitted RBF kernel')
ax.scatter(bin_mean, bin_vario, marker='x', label='Data')
ax.set_xlim(0, 20000)
ax.set_ylabel('Semivariance (dBZ^2)')
ax.set_xlabel('Distance (m)')
ax.legend()
plt.show()

# %%
# Third step: use kriging instance to interpolate missing values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
refl_rbf, kriging_rbf = transform_funcs.interpolate_missing(
    lawr_decluttered, remapper=kriging_rbf, trunc_radius=1.5
)

# %%
rbf_plotter = plot_funcs.plot_reflectivity(
    refl_rbf, plot_path='/tmp/lawr.png',
    title='Kriging(RBF kernel)',
)
rbf_plotter.show()

# %%
# Plot interpolation standard deviation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# As maximum value for colorbar is 2 dBZ set, because all interpolated values
# above this threshold were not interpolated.
stddev = np.sqrt(kriging_rbf.covariance)

# %%
stddev_latlon = kriging_rbf._grid_out.get_lat_lon()
refl_stddev = lawr_decluttered.where(
    lawr_decluttered>9999).lawr.set_grid_coordinates(lawr_grid)

# %%
stddev_plotter = plot_funcs.plot_reflectivity(
    refl_stddev, plot_path='/tmp/lawr.png',
    title='StdDev Kriging(RBF kernel)',
)
scatter_points = stddev_plotter.subplots['map'].ax.scatter(
    stddev_latlon['lon'], stddev_latlon['lat'], c=stddev,
    cmap='OrRd', vmin=0, vmax=2, s=100, marker='h',
    transform=ccrs.PlateCarree(),
)
stddev_plotter.subplots['map'].layers[0].plot_store = scatter_points
stddev_plotter.subplots['colorbar'].swap_layer(
    ColorbarLayer(stddev_plotter.subplots['map'].layers[0]),
    stddev_plotter.subplots['colorbar'].layers[0]
)
stddev_plotter.show()

# %%
# Reflectivity without clutter detection
# --------------------------------------
plotter = plot_funcs.plot_reflectivity(
    lawr_denoised, plot_path='/tmp/lawr.png',
    title='Without cluttermap',
)
plotter.show()
