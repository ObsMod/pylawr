#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quickstart guide low-level api
==============================

This notebook is designed as quickstart guide for the low-level api
of :py:mod:`pylawr`.
"""

# %%
# Introduction
# ------------
# In the following, we will use the low-level api to process data from the
# ``HHG`` X-band weather radar.

# %%
# The following processing steps are:
# 1. Read in data with pre-defined data handler for data,
# 2. Define a corresponding grid which represents the position of the data,
# 3. Remove noise,
# 4. Filter out clutter,
# 5. Interpolate missing values,
# 6. Remap the data to a new grid, and
# 7. Plot the data.

# %%
# Import packages
# ---------------
# Package including the LawrHandler
import pylawr.datahandler
# Package including all grids
import pylawr.grid
# For filters
import pylawr.transform
# For beamexpansion tag
import pylawr.transform.spatial.beamexpansion
# For interpolation
import pylawr.remap
# For plotting
import pylawr.plot
import pylawr.plot.layer

# For filters from wradlib
import wradlib

# For array operations
import numpy as np

# For plotting purpose
import matplotlib.colors as mpl_colors

# For georeferencing within plotting
import cartopy.crs as ccrs

# %%
# 1. Read in data
# ---------------
# To read in the ascii data, we will use a pre-defined ascii data handler.
# For examples to read in other file types, please take a look to other
# examples.This data handler will get an opened file handler such that
# we have to open the file beforehand.

file_path = '../tests/data/lawr_data.txt'
opened_file = open(file_path, mode='r')

# %%
# Different data handlers are available in the ``datahandler`` subpackage of
# :py:mod:`pylawr`. To read in ascii files generated with Universität
# Hamburg weather radar, we will use ``LawrHandler``

data_handler = pylawr.datahandler.LawrHandler(opened_file)

# %%
# The next step is to decode the data. Every data handler has a unified
# method (``get_reflectivity``) to get the reflectivity as xarray.DataArray
# in ``dBZ`` from the opened file.

refl_raw = data_handler.get_reflectivity()
print(refl_raw)

# %%
# As we can see, the data handler decodes the data into three dimensions.
# The time is inferred from ``datestr`` within the metadata of the file.
# The data handler couldn't decode the azimuth and range from this
# ascii-file and we have to set them later with a grid. The metadata within
# the file are set as attributes of this data array.

# %%
# As last step in this read in procedure, we need to close the opened file.
# For the opened text file, this doesn't matter but for binary files, like
# netCDF or HDF5-files, this is a crucial step because else the file will
# be corrupted.

opened_file.close()

# %%
# 2. Define a grid
# ----------------
# The next large step is to define a grid, which is used as georeference of
# the data. Some data handlers can decode the grid from the metadata of the
# opened file. Within the ascii-file there is not enough information to
# decode the grid and we have to define it manually.

# %%
# Almost every weather radar works in polar coordinates. For this,
# we can use ``PolarGrid``. The center of this grid describes the origin
# of this grid. For ``PolarGrid``, this is the position of the radar
# in (center=(latitude, longitude, height)). We further need to define the
# beam elevation (beam_ele) in degrees from the earth,
# number of ranges (nr_ranges), the range resolution in meters (range_res)
# and the number of azimuth angles (nr_azi). We further can set an offset
# for the azimuth angles (azi_offset, in degrees) and the
# ranges (range_offset, in meters) if the radar is rotated or
# valid only after some meters.

center = (53.56796, 9.97451, 95)
grid = pylawr.grid.PolarGrid(
    center=center,
    beam_ele=3,
    nr_ranges=333,
    range_res=60,
    range_offset=0,
    nr_azi=360,
    azi_offset=0
)

# %%
# We now defined a grid to the corresponding data. We can now set the
# coordinates of this grid to our data with
# ``lawr.set_grid_coordinates(grid)``.

refl_raw = refl_raw.lawr.set_grid_coordinates(grid)
print(refl_raw)

# %%
# 3. Remove noise
# ---------------
# In this package, noise is defined as the 10. percentile of the raw
# uncorrected reflectivity.

# %%
# Caused by beam expansion the radar beam losses its intensity through the
# ranges. In the raw ASCII-data, this effect is already corrected.
# The data array has a tagging system, where tags can be added and
# removed from the data array. The next step is to tag the array that it
# is already corrected. This tag is necessary for the beam expansion
# filter to determine the direction of the correction.
# To add a tag to the DataArray, we will use ``lawr.add_tag(TAG_NAME)``.

refl_raw.lawr.add_tag(
    pylawr.transform.spatial.beamexpansion.TAG_BEAM_EXPANSION_CORR
)
print(refl_raw)

# %%
# In the output, we can see that the beam expansion tag is added to the
# DataArray as attribute. These tags can be used to track the processing
# steps of this DataArray.

# %%
# The noise is calculated and removed in the raw, uncorrected data and we
# need to revert the beam expansion correction. For this, we will use the
# ``BeamExpansion`` filter, which determines the direction of the correction
# automatically based on set tags.

beam_expansion = pylawr.transform.spatial.BeamExpansion()

# %%
# Every filter within ``pylawr.filter`` has a ``transform`` method, which
# applies the filter to given DataArray. Normally, the transform method
# returns a data array as first value.

refl_uncorr = beam_expansion.transform(refl_raw.lawr.to_z())
print(refl_uncorr)

# %%
# After, we uncorrected the beam expansion, we can remove noise from our
# radar field. The noise is removed with a ``pylawr.filter.NoiseRemover``.
# A noise remover needs an appropriate noise level. For this, the noise
# remover can be fitted. In current implementation, the noise level is a
# running median of the last ten determined noise levels. For this single
# test case, the noise level equals the determined noise level from given
# reflectivity.

noise_remover = pylawr.transform.temporal.NoiseRemover()
print('Before fit: {0:.3E}'.format(noise_remover.noiselevel))
noise_remover.fit(refl_uncorr)
print('After fit: {0:.3E}'.format(noise_remover.noiselevel))

# %%
# After, we fitted the noise remover, we can use it to remove the noise from
# given reflectivity field. After we removed the noise, we need to recorrect
# the beam expansion. Finally, we translate our radar field into dbZ
# with `field.lawr.to_dbz()`.

refl_wo_noise_uncorr = noise_remover.transform(refl_uncorr)
refl_wo_noise = beam_expansion.transform(refl_wo_noise_uncorr).lawr.to_dbz()

# %%
# 4. Filter out clutter
# ---------------------
# After we removed noise from our radar field, we can filter out clutter.
# For this we use techniques from computer vision to identify measured points,
# where the values and/or gradients are jumping. These points are removed
# by the TDBZ and SPIN filter and replaced with a NaN value. To further
# filter out clutter, we also use another clutter filter from `wradlib`
# called "Gabella filter". This should show how a clutter filter outside
# of pylawr can be used to filter clutter.

# %%
# In our package, clutter filters create a map, which will be translated to
# a binary clutter map via thresholding. The created cluttermaps can be
# combined such that different combinations of cluttermaps are possible.

# %%
# As first step, we need to initialize the clutter filters, here the
# TDBZ and SPIN filter.

tdbz_filter = pylawr.transform.filter.TDBZFilter()
spin_filter = pylawr.transform.filter.SPINFilter()

# %%
# Every clutter filter has `create_cluttermap` as method, which
# creates a `ClutterMap` object, where the the cluttermap is saved as array.

tdbz_cmap = tdbz_filter.create_cluttermap(refl_wo_noise)
spin_cmap = spin_filter.create_cluttermap(refl_wo_noise)

# %%
# As previously written, we further want to use a clutter filter, which
# is not within our package. For this example, we use the "Gabella filter"
# from `wradlib`. This filter returns a boolean clutter map, which is then
# converted into a `ClutterMap` object. For this, the clutter map needs to
# be translated into an integer array and need to have three dimensions
# (time, grid_dim_1, grid_dim_2). After, we converted the clutter map into
# a `ClutterMap` object, we can use this object as native clutter map.

gabella_arr = wradlib.clutter.filter_gabella(
    refl_wo_noise.values[0], wsize=7, thrsnorain=0.,
    tr1=6., n_p=12, tr2=1.3, rm_nans=False,
    radial=False, cartesian=False
)[np.newaxis, ...]
gabella_cmap = pylawr.transform.filter.ClutterMap('Gabella',
                                                  gabella_arr.astype(int))

# %%
# Cluttermaps can be combined via `append()`. This method appends a given
# cluttermap to existing cluttermaps. To get a clean cluttermap, we
# initialize an empty `ClutterMap` object, where then the other cluttermaps
# are appended. The fuzzy threshold can be set to get a fuzzy
# identification of clutter.

cluttermap = pylawr.transform.filter.ClutterMap('ClutterMap',
                                                fuzzy_threshold=0.)
cluttermap.append(tdbz_cmap)
cluttermap.append(spin_cmap)
cluttermap.append(gabella_cmap)

# %%
# After we appended all clutter maps, we can transform a given radar
# field with this clutter map. This will set all clutter pixels to NaN.

refl_filtered = cluttermap.transform(refl_wo_noise)

# %%
# 5. Interpolate missing values
# -----------------------------
# After, we filtered out clutter values, we need to interpolate missing
# values. To interpolate missing values, we can use an ``Interpolator``,
# defined in ``pylawr.filter``. This interpolator uses a given remapping
# algorithm to interpolate missing values. If no remapping algorithm is set,
# the interpolator uses a single nearest neighbor interpolation.

# %%
# In this example, we use a nearest neighbor interpolation with five
# neighbors, which are then averaged with a median. After we defined
# the algorithm, we initialize an `Interpolator`.

int_remapper = pylawr.remap.NearestNeighbor(5, max_dist=500)
int_transformer = pylawr.transform.spatial.Interpolator(algorithm=int_remapper)

# %%
# The interpolator is then used to fill missing values via
# ``Interpolator.transform()``. This method refits given remapper and
# interpolates from existing to missing grid points. This interpolated
# result is then used to fill missing values. This step may take several
# seconds, because the remapping algorithm is refitted.

refl_interpolated = int_transformer.transform(refl_filtered)

# %%
# 6. Remap the data to a new grid
# -------------------------------
# After we interpolated missing values, we can remap our data to a new grid,
# e.g. a cartesian grid. This is needed to combine different radar fields
# and for nowcasting purpose. In this example, we will used it to
# visualize the remapping effects.

# %%
# As first step for remapping, we need to initialize a remapping object,
# here nearest neighbour with six averaged neighbors.

grid_remapper = pylawr.remap.NearestNeighbor(1, max_dist=300)

# %%
# This remapping algorithm need to be fitted to translate given radar
# field with given old grid to a target grid with ``fit()``. As target grid,
# a cartesian grid is used here. This fit procedure is necessary because
# the finding of the nearest neighbors is the most time-expensive step
# within the fitting algorithms.

target_grid = pylawr.grid.CartesianGrid(start=-25000)
grid_remapper.fit(grid, target_grid)

# %%
# After we found the nearest neighbors with ``fit()``, we can use ``remap()``
# to remap given radar field from old grid to a new target grid.

refl_remapped = grid_remapper.remap(refl_interpolated)
print(refl_remapped)

# %%
# This remapping is the last processing step within this example.
# As last step, we will plot this processed data.

# %%
# 7. Plot the data
# ----------------
# Our plotting system is based on an object-oriented approach and
# matplotlib. As main plotting tool, we introduced a ``Plotter``,
# which is the main plot and can be split in several subplots. This
# main plot has a given grid (e.g. 14x14 grid points) and we can define
# different grid slices, which are translate into subplots. The main
# plotting logic is hidden into plotting layers. They can be added to a
# subplot and are then plotted on this specified subplot.

# %%
# As first step, we will translate our logarithmic reflectivity into a
# rain rate. For this, we use a z-r relationship with ``zr_convert``.
# There, the scale (a) and polynomial grade factor (b) can be specified.
# After this conversion, we set all values below 0.1 mm/h to
# NaN for plotting purpose.

rain_rate = refl_remapped.lawr.to_z().lawr.zr_convert(a=256, b=1.42)
rain_rate = rain_rate.where(rain_rate>=0.1, np.nan)

# %%
# We need to initialize a plotter and we need to specify the grid size for
# this plotter. In this example, we will use a 14x14 grid. We further can
# specify the names and grid slices for different subplots (here
# `map` and `colorbar`).

default_gridspec_slices = {
    'map': [
        slice(None, None),
        slice(12),
    ],
    'colorbar': [
        slice(None, None),
        slice(12, None),
    ],
}
plotter = pylawr.plot.Plotter(
    grid_size=(14, 14), grid_slices=default_gridspec_slices,
    backend_name='agg', figsize=(13, 9)
)

# %%
# Within pylawr, we implemented different colormaps, which are used
# for our plots. Here, we will use a pre-specified rain colormap and set
# all NaN values to transparent. We further specify a logarithmic colorscale.

cmap = pylawr.plot.available_cmaps['rain']
cmap.set_bad((0, 0, 0, 0))

norm = mpl_colors.LogNorm(vmin=0.1, vmax=200)

# %%
# In our next steps, we define our plotting layers.

# %%
# To plot a background map, we can use a `BackgroundLayer`, which will plot
# an openstreetmap as background (here we will use ``cartopy`` implicit).
# The zoom level of this openstreetmap layer can be set via `resolution`.

# l_bg = pylawr.plot.layer.BackgroundLayer(resolution=10)

# %%
# As next layer, we define a ``RadarFieldLayer``, which is used to plot a
# given array and grid on a subplot. Here, we define our rain rate as
# plotting array with our defined cartesian grid as underlying grid.
# The previously defined colormap and colorscale are used for this
# radar layer.

l_radar = pylawr.plot.layer.RadarFieldLayer(radar_field=rain_rate,
                                            grid=target_grid, zorder=1)
l_radar['cmap'] = cmap
l_radar['norm'] = norm

# %%
# As additional information, we want to add a colorbar. For this, we can
# use ``ColorbarLayer``, we will plot a colorbar for given radar field.
# The settings of this colorbar are changed after plotting,
# which will create this colorbar.

l_cbar = pylawr.plot.layer.ColorbarLayer(l_radar)

# %%
# These three layers are added to specified subplots. These subplots are
# automatically generated based on the names of the slices
# (here, `map` and `colorbar`). The background and radar field are added
# to the `map` subplot, while the colorbar is added to `colorbar` subplot.

# plotter.add_layer('map', l_bg)
plotter.add_layer('map', l_radar)
plotter.add_layer('colorbar', l_cbar)

# %%
# Our map subplot should be a georeferenced subplot. For this, we define
# a cartopy projection (here a rotated pole).

rotated_pole = ccrs.RotatedPole(-170.415, 36.063)

# %%
# All subplots are saved in the plotter under ``Plotter.subplots`` as
# dictionary. The projection of the ``map`` subplot is then set to
# the previously defined projection.

map_subplot = plotter.subplots.get('map')
map_subplot.projection = rotated_pole

# %%
# We need to specify the extent of this mapping subplot. We can set the
# extent as dictionary with latitude and longitude values. If
# auto_extent of this subplot is set to true, the extent will be
# extended during plotting to preserve the geometry of this subplot.

map_subplot.auto_extent = True
map_subplot.extent = dict(
    lon_min=9.6,
    lon_max=10.3,
    lat_min=53.35,
    lat_max=53.76
)

# %%
# The trigger the plotting commands, we have to call ``Plotter.plot()``.
# This method creates the figure and subplots and plot the layers
# on created subplots.

plotter.plot()

# %%
# After, we plotted our layers, subplots and plotter, we can
# manipulate the figure, axes etc. Here, we manipulate the ticks
# of the colorbar.

l_cbar.colorbar.set_ticks([0.1, 0.5, 1, 2, 5, 10, 100])
_ = l_cbar.colorbar.ax.set_yticklabels(['0.1', '0.5', '1',
                                        '2', '5', '10', '>100'])

# %%
# After ``plot()`` was called, the plotter has all methods from the
# underlying figure. Then, ``Plotter.savefig()`` can be used to save the
# plotted figure to a specified path.

plotter.savefig('/tmp/quickstart_low_level.png')
plotter.show()

# %%
# We can see the effect of remapping in our saved plot. The
# remapping decreased the resolution to 100 meters from 60 meters
# and interpolated values, where no values are available. This is caused
# by nearest neighbor remapping. A more advanced algorithm, e.g.
# Kriging, can be used to avoid these artifacts.

# %%
# After we saved the figure, this quickstart example for the low-level
# api is finished and showed that different components of :py:mod:`pylawr`
# can be used to read-in, process and plot radar data. Further examples
# show how the plotter can be used for comparison plots and how
# different subpackages can interact.