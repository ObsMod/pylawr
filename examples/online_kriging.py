#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example for adaptive kriging with particle filter
=================================================

"""

# sphinx_gallery_thumbnail_number = 2

# %%
# Import packages
# ---------------
from pylawr.remap import kernel as k_ops
from pylawr.remap import OrdinaryKriging
from pylawr.plot.layer import ColorbarLayer
from pylawr.grid import PolarGrid

import pylawr.functions.input as input_funcs
import pylawr.functions.transform as filter_funcs
import pylawr.functions.plot as plot_funcs
import pylawr.functions.fit as fit_funcs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import cartopy.crs as ccrs

# %%
rnd = np.random.RandomState(42)

# %%
# Read and preprocess data
# ------------------------
file_path = '../tests/data/example_lawr.nc'
reflectivity, _ = input_funcs.read_lawr_nc_new(file_path)
grid = PolarGrid()
reflectivity = reflectivity.isel(time=[0, ])
reflectivity = reflectivity.lawr.set_grid_coordinates(grid)
reflectivity, noise_remover = filter_funcs.remove_noise(reflectivity)
reflectivity = reflectivity.lawr.to_dbz()

# %%
refl_holes, cluttermap = filter_funcs.remove_clutter_lawr(reflectivity)
refl_holes = refl_holes.where(refl_holes>5)
refl_holes[:, 205:210, 160:180] = np.nan
refl_holes = refl_holes.lawr.set_grid_coordinates(grid)

# %%
# Define kernel
# -------------
rbf_data_kernel = k_ops.gaussian_rbf(length_scale=5000, stddev=5)
noise_kernel = k_ops.WhiteNoise(rbf_data_kernel.placeholders[0], noise_level=1,
                                constant=False)
rbf_kernel = rbf_data_kernel + noise_kernel

kriging_rbf = OrdinaryKriging(rbf_kernel, n_neighbors=10)

fitted_kriging, particle_filter = fit_funcs.fit_kriging(
    refl_holes, iterations=50, kriging=kriging_rbf, ens_size=500,
    ens_threshold=150
)
print(fitted_kriging.kernel)


# %%
# Plot semivariogram with mean kernel
# -----------------------------------

dist_linspace = np.linspace(0, 20000, 10000)

kernel_semivar = fitted_kriging.kernel.variogram(dist_linspace)

bin_mean, bin_vario = fit_funcs.sample_variogram(refl_holes, samples=10000,
                                                 bins=100)

# %%
fig, ax = plt.subplots()
ax.plot(dist_linspace, kernel_semivar, c='k', label='Fitted RBF kernel')
ax.scatter(bin_mean, bin_vario, marker='x', label='Data')
ax.set_ylim(0, 100)
ax.set_xlim(0, 20000)
ax.set_ylabel('Semivariance (dBZ^2)')
ax.set_xlabel('Distance (m)')
ax.legend()
plt.show()

# %%
# Plot distribution of parameters
# -------------------------------
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, ax = plt.subplots()
param_hist = np.array(particle_filter.param_hist)
iterations = np.arange(param_hist.shape[0])
for mem in range(param_hist.shape[1]):
    ax.plot(iterations, param_hist[:, mem, 1], color=colors[0], alpha=0.01)
ax.set_xlim(0, np.max(iterations))
ax.set_ylim(0, 20000)
ax.set_ylabel('Length scale (m)')
ax.set_xlabel('Iteration')
plt.show()

# %%
fig, ax = plt.subplots()
scatter_color = np.array(matplotlib.colors.to_rgba(colors[0]))
scatter_color = np.repeat(scatter_color[None, :], repeats=500, axis=0)
scatter_color[:, -1] = (particle_filter.weight_hist[-1] /
                        particle_filter.weight_hist[-1].max() * 0.75)
_ = ax.scatter(particle_filter.param_hist[-1][:, 1],
               particle_filter.param_hist[-1][:, 0], c=scatter_color)
ax.set_xlabel('Length scale (m)')
ax.set_ylabel('Sill (dBZ)')
plt.show()


# %%
# Reflectivity plotting without interpolation
# -------------------------------------------

hole_plotter = plot_funcs.plot_reflectivity(
    refl_holes, plot_path='/tmp/lawr.png',
    title='Hole',
)
hole_plotter.show()


# %%
# Reflectivity plotting with tuned kriging interpolation
# ------------------------------------------------------

refl_rbf, fitted_kriging = filter_funcs.interpolate_missing(
    refl_holes, remapper=fitted_kriging
)

rbf_plotter = plot_funcs.plot_reflectivity(
    refl_rbf, plot_path='/tmp/lawr.png',
    title='Kriging(RBF kernel)',
)
rbf_plotter.show()

# %%
stddev = np.sqrt(fitted_kriging.covariance)
stddev_latlon = fitted_kriging._grid_out.get_lat_lon()
refl_stddev = refl_holes.where(refl_holes > 9999)
refl_stddev = refl_stddev.lawr.set_grid_coordinates(grid)
stddev_plotter = plot_funcs.plot_reflectivity(
    refl_stddev, plot_path='/tmp/lawr.png',
    title='StdDev Kriging(RBF kernel)',
)
scatter_points = stddev_plotter.subplots['map'].ax.scatter(
    stddev_latlon['lon'], stddev_latlon['lat'], c=stddev,
    cmap='OrRd', vmin=0, s=10, marker='h',
    transform=ccrs.PlateCarree(),
)
stddev_plotter.subplots['map'].layers[0].plot_store = scatter_points
stddev_plotter.subplots['colorbar'].swap_layer(
    ColorbarLayer(stddev_plotter.subplots['map'].layers[0]),
    stddev_plotter.subplots['colorbar'].layers[0]
)
stddev_plotter.show()

