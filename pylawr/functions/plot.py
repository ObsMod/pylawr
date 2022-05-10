#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
from collections import OrderedDict

# External modules
from matplotlib import colors
from matplotlib import cm
import cartopy.crs as ccrs
from cartopy import geodesic
from cartopy.feature import ShapelyFeature
import shapely.geometry as sgeom
import xarray

# Internal modules
from pylawr.plot.layer.colorbar import ColorbarLayer
from pylawr.plot.layer.lawr_header import LawrHeaderLayer
from pylawr.plot.layer.radarfield import RadarFieldLayer
from pylawr.plot.plotter import Plotter
from pylawr.plot.cmaps import available_cmaps
from pylawr.utilities import log_decorator
from pylawr.field import get_verified_grid

logger = logging.getLogger(__name__)


def create_default_plotter(grid_extent=None):
    """
    Create a default plotter with a header, map and colorbar subplot.

    Parameters
    ----------
    grid_extent : dict
        Extent of the map subplot

    Returns
    -------
    plotter : :py:class:`pylawr.plot.plotter.Plotter` or None, optional
        The initialized plotter with default grid slices and set projection for
        map subplot.
    clutter : bool, optional
        If a cluttermap should be plotted
    """
    default_gridspec_slices = {
        'header': [
            slice(2),
            slice(None),
        ],
        'map': [
            slice(2, None),
            slice(12),
        ],
        'colorbar': [
            slice(2, None),
            slice(12, None),
        ],
    }
    plotter = Plotter(grid_size=(14, 14), grid_slices=default_gridspec_slices,
                      backend_name='agg', figsize=(13, 9))

    map_subplot = plotter.subplots.get('map')
    rotated_pole = ccrs.RotatedPole(-170.415, 36.063)
    map_subplot.projection = rotated_pole
    map_subplot.auto_extent = True
    if isinstance(grid_extent, dict):
        map_subplot.extent = grid_extent
    else:
        map_subplot.extent = dict(
            lon_min=9.6,
            lon_max=10.3,
            lat_min=53.35,
            lat_max=53.76
        )
    return plotter


@log_decorator(logger)
def plot_rain_rate(rate, plot_path, title='Hamburg X-Band Radar',
                   radar='HHG', grid_extent=None, plotter=None):
    """
    This function can be used to plot a rain rate. The figure will have a given
    title, an information box on the top-left with extracted date from given
    rain rate. Another information box on top-right shows a given radar name and
    what variable is plotted. The main subplot displays the rain rate as
    geo-referenced map with given grid and extent.

    Parameters
    ----------
    rate : :py:class:`xarray.DataArray`
        The rain field with r as unit.
    plot_path : str
        This path is used to save the figure.
    title : str
        Title of this figure
    radar : str
        Radar name
    grid_extent : dict
        Extent of the map subplot
    plotter: :py:class:`pylawr.plot.plotter.Plotter` or None
        The plotter to use. If none plotter is given a default plot is created.
    """
    # rain layer
    grid = get_verified_grid(rate)
    rain_layer = RadarFieldLayer(radar_field=rate, zorder=1)
    rain_layer['cmap'] = available_cmaps['rain']
    rain_layer['cmap'].set_bad((0, 0, 0, 0))
    rain_layer['norm'] = colors.LogNorm(vmin=0.1, vmax=200)

    # header layer
    hl = LawrHeaderLayer()
    plot_date = rate.indexes['time'][-1].tz_localize('UTC')
    current_date = plot_date.tz_convert('Europe/Berlin').strftime('%Y-%m-%d')
    current_time = plot_date.tz_convert('Europe/Berlin').strftime('%H:%M:%S')
    hl.left = OrderedDict(Datum=current_date, Zeit=current_time)
    hl.title = title
    hl.right = OrderedDict(Radar=radar, Parameter='Regenrate [mm/h]')

    # if a plotter is given the dynamic layers will be replaced
    if not plotter:
        plotter = create_default_plotter(grid_extent=grid_extent)

        cl = ColorbarLayer(rain_layer)

        plotter.add_layer('map', rain_layer)
        plotter.add_layer('colorbar', cl)
        plotter.add_layer('header', hl)
        plotter.plot()

        cl.colorbar.set_ticks([0.1, 0.5, 1, 2, 5, 10, 100])
        cl.colorbar.ax.set_yticklabels(
            ['0.1', '0.5', '1', '2', '5', '10', '>100'])

    else:
        try:
            plotter.swap_layer(rain_layer, layer_num=("map", 1))
        except IndexError:
            plotter.add_layer('map', rain_layer)

        try:
            plotter.swap_layer(hl, layer_num=("header", 0))
        except IndexError:
            plotter.add_layer('header', hl)

        if not plotter._subplots['colorbar'].layers:
            cl = ColorbarLayer(rain_layer)
            plotter.add_layer('colorbar', cl)
            cl.colorbar.set_ticks([0.1, 0.5, 1, 2, 5, 10, 100])
            cl.colorbar.ax.set_yticklabels(
                ['0.1', '0.5', '1', '2', '5', '10', '>100'])

    plotter.savefig(plot_path)
    return plotter


@log_decorator(logger)
def plot_reflectivity(refl, plot_path, title='Hamburg X-Band Radar',
                      cmap=available_cmaps['reflectivity'],
                      vmin=0, vmax=70,
                      radar='HHG', grid_extent=None,
                      add_circle=(9.97451, 53.56796, 20 * 1e3)):
    """
    This function can be used to plot a reflectivity in dBZ. The figure will
    have a given title, an information box on the top-left with extracted date
    from given reflectivity. Another information box on top-right shows a given
    radar name and what variable is plotted. The main subplot displays the
    reflectivity as geo-referenced map with given grid and extent. The color
    of the map is freely selectable.

    Parameters
    ----------
    refl : :py:class:`xarray.DataArray`
        The reflectivity with dBZ as unit.
    plot_path : str
        This path is used to save the figure.
    title : str
        Title of this figure
    radar : str
        Radar name
    grid_extent : dict
        Extent of the map subplot
    add_circle : tuple or None
        The circle specifications: `longitude_center`, `latitude_center`,
        `radius`. If `None` there will be no circle.
    cmap : py:class: 'matplotlib.colors.Colormap'
        The colormap of the plot
    vmin : float
        Normalization of the colormap
    vmax : float
        Normalization of the colormap
    """
    grid = get_verified_grid(refl)
    plotter = create_default_plotter(grid_extent=grid_extent)

    refl_layer = RadarFieldLayer(radar_field=refl, grid=grid)

    refl_layer['cmap'] = cmap
    refl_layer['cmap'].set_under(alpha=0)
    refl_layer['norm'] = colors.Normalize(vmin, vmax)

    cl = ColorbarLayer(refl_layer)

    hl = LawrHeaderLayer()

    plot_date = refl.indexes['time'][-1].tz_localize('UTC')
    current_date = plot_date.tz_convert('Europe/Berlin').strftime('%Y-%m-%d')
    current_time = plot_date.tz_convert('Europe/Berlin').strftime('%H:%M:%S')
    hl.left = OrderedDict(Datum=current_date, Zeit=current_time)
    hl.title = title
    hl.right = OrderedDict(Radar=radar, Parameter='Reflektivität [dBZ]')

    plotter.add_layer('map', refl_layer)
    plotter.add_layer('colorbar', cl)
    plotter.add_layer('header', hl)

    plotter.plot()

    if add_circle:
        geod = geodesic.Geodesic()
        circle = geod.circle(add_circle[0], add_circle[1], add_circle[2])
        geom = [sgeom.Polygon(circle)]
        feature = ShapelyFeature(geom, refl_layer._transform)
        plotter.subplots['map'].ax.add_feature(feature, facecolor='none',
                                               edgecolor='black',
                                               linestyle=':')

    plotter.savefig(plot_path)
    return plotter


@log_decorator(logger)
def plot_rain_clutter(rate, cluttermap, plot_path,
                      title='Hamburg X-Band Radar', radar='HHG',
                      grid_extent=None):
    """
    This function can be used to plot a rain rate. The figure will have a given
    title, an information box on the top-left with extracted date from given
    rain rate. Another information box on top-right shows a given radar name and
    what variable is plotted. The main subplot displays the rain rate as
    geo-referenced map with given grid and extent. On top of the rain rate is a
    given cluttermap displayed.

    Parameters
    ----------
    rate : xarray.DataArray
        The rain field with r as unit.
    cluttermap : ClutterMap object
        ClutterMap with n clutter
    plot_path : str
        This path is used to save the figure.
    title : str
        Title of this figure
    radar : str
        Radar name
    grid_extent : dict or None
        Extent of the map subplot
    """
    grid_slices = dict(
        header=[slice(None, 2, None), slice(None, 14, None)],
        map=[slice(2, 14, None), slice(None, 10, None)],
        rain_cmap=[slice(2, 14, None), slice(10, 12, None)],
        clutter_cmap=[slice(2, 14, None), slice(12, 14, None)]
    )
    plotter = Plotter(grid_size=(14, 14), grid_slices=grid_slices,
                      figsize=(13, 9), backend_name='agg')
    map_subplot = plotter.subplots.get('map')
    rotated_pole = ccrs.RotatedPole(-170.415, 36.063)
    map_subplot.projection = rotated_pole
    map_subplot.auto_extent = True
    if isinstance(grid_extent, dict):
        map_subplot.extent = grid_extent
    else:
        map_subplot.extent = dict(
            lon_min=9.6,
            lon_max=10.3,
            lat_min=53.35,
            lat_max=53.76
        )

    # rain layer
    grid = get_verified_grid(rate)
    rain_layer = RadarFieldLayer(radar_field=rate, grid=grid)
    rain_layer['cmap'] = available_cmaps['rain']
    rain_layer['cmap'].set_bad((0, 0, 0, 0))
    rain_layer['norm'] = colors.LogNorm(vmin=0.1, vmax=200)

    # clutter layer
    clutter_layer = RadarFieldLayer(radar_field=cluttermap.array.mean(axis=0),
                                    grid=grid)
    clutter_layer['cmap'] = cm.get_cmap("Greys", len(cluttermap.layers) + 1)
    clutter_layer['cmap'].set_bad((0, 0, 0, 0))

    rain_colorbar = ColorbarLayer(rain_layer)
    clutter_colorbar = ColorbarLayer(clutter_layer)

    hl = LawrHeaderLayer()

    plot_date = rate.indexes['time'][-1].tz_localize('UTC')
    current_date = plot_date.tz_convert('Europe/Berlin').strftime('%Y-%m-%d')
    current_time = plot_date.tz_convert('Europe/Berlin').strftime('%H:%M:%S')
    hl.left = OrderedDict(Datum=current_date, Zeit=current_time)
    hl.title = title
    hl.right = OrderedDict(Radar=radar, Parameter='Regenrate [mm/h]')

    plotter.add_layer('map', clutter_layer)
    plotter.add_layer('map', rain_layer)
    plotter.add_layer('rain_cmap', rain_colorbar)
    plotter.add_layer('clutter_cmap', clutter_colorbar)
    plotter.add_layer('header', hl)

    plotter.plot()

    rain_colorbar.colorbar.set_ticks([0.1, 0.5, 1, 2, 5, 10, 100])
    rain_colorbar.colorbar.ax.set_yticklabels(
        ['0.1', '0.5', '1', '2', '5', '10', '>100']
    )

    clutter_colorbar.colorbar.set_ticks([0., 0.5, 1.])

    plotter.savefig(plot_path)
    return plotter


@log_decorator(logger)
def plot_leaflet(rate, plot_path, grid_extent=None,
                 add_circle=(9.97451, 53.56796, 20 * 1e3),
                 dpi=100):
    """
    This function can be used to plot a rain rate in a leaflet-conform way. The
    figure will be transparent, and only the rain rate and a possible radar
    circle is shown on the figure.

    Parameters
    ----------
    rate : xarray.DataArray
        The rain field with r as unit.
    plot_path : str
        The created plot will be saved to this plot path.
    grid_extent : dict
        The minimum extent of the map subplot.
    add_circle : tuple or None
        The circle specifications: `longitude_center`, `latitude_center`,
        `radius`,  if `None` there will be no circle.
    dpi : int
        The pixels per inch resolution of the resulting png.

    Returns
    -------
    plotter : :py:class:`pylawr.plot.plotter.Plotter`
        This plotter instance was used to create the plot.
    """
    grid_slices = dict(
        map=[slice(None, None, None), slice(None, None, None)],
    )
    plotter = Plotter(grid_size=(1, 1), grid_slices=grid_slices,
                      figsize=(13, 9), backend_name='agg')
    plotter.gridspec_settings = dict(
        hspace=0, wspace=0, bottom=0, top=1, left=0, right=1
    )

    map_subplot = plotter.subplots.get('map')
    rotated_pole = ccrs.RotatedPole(-170.415, 36.063)
    map_subplot.projection = rotated_pole
    map_subplot.auto_extent = True
    if isinstance(grid_extent, dict):
        map_subplot.extent = grid_extent
    else:
        map_subplot.extent = dict(
            lon_min=9.6,
            lon_max=10.3,
            lat_min=53.35,
            lat_max=53.76
        )

    rain_layer = RadarFieldLayer(radar_field=rate, zorder=1)
    rain_layer['cmap'] = available_cmaps['rain']
    rain_layer['cmap'].set_bad((0, 0, 0, 0))
    rain_layer['norm'] = colors.LogNorm(vmin=0.1, vmax=200)

    plotter.add_layer('map', rain_layer)
    plotter.plot()

    if add_circle is not None:
        geod = geodesic.Geodesic()
        circle = geod.circle(add_circle[0], add_circle[1], add_circle[2])
        geom = [sgeom.Polygon(circle)]
        feature = ShapelyFeature(geom, rain_layer._transform)
        map_subplot.ax.add_feature(feature, facecolor='none',
                                   edgecolor='black',
                                   linestyle=':')

    map_subplot.ax.outline_patch.set_visible(False)
    map_subplot.ax.background_patch.set_visible(False)
    map_extent = map_subplot.ax.get_extent(crs=ccrs.PlateCarree())
    logger.debug('####### Map extent: {0}'.format(map_extent))
    plotter.savefig(plot_path, transparent=True, format='png', dpi=dpi)
    return plotter
