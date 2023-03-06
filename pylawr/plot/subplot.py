#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 5/31/18
#
# Created for pattern
#
#
#    Copyright (C) {2018}
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# System modules
import logging
import collections.abc

# External modules
import cartopy.crs as ccrs
from cartopy.mpl import geoaxes

import matplotlib.axes._axes as mpl_subplots

# Internal modules
from pylawr.plot.layer.base import BaseLayer


logger = logging.getLogger(__name__)


default_tick_params = {
    'axis': 'both',
    'which': 'both',
    'bottom': False,
    'top': False,
    'left': False,
    'right': False,
    'labelbottom': False,
    'labeltop': False,
    'labelleft': False,
    'labelright': False,
}


class Subplot(object):
    """
    A Subplot is a part of a plotting figure. Basically, it is a wrapper around
    :py:class:`matplotlib.axes.Axes`. A subplot has multiple layers, which are
    ordered in a list. During plotting a subplot creates a
    :py:class:`~matplotlib.axes.Axes`, on which the layers are plotted. The
    actual plotting logic is hided in the layers. The axes settings can be set
    in a dict-like manner or via manipulation of `ax_settings` dict.

    Parameters
    ----------
    layers_list : list(child of :py:class:`pylawr.plot.layer.base.BaseLayer`) or
        None, optional
        This list of layers are the basic layers of this subplot. Additional
        layers can be added with
        :py:meth:`~pylawr.plot.subplot.Subplot.add_layer` and layers can be
        deleted with :py:meth:`~pylawr.plot.subplot.Subplot.del_layer`. If the
        given list of layers is None, an empty list will be initialized. If the
        list of layers cannot be casted into a list type a TypeError is raised.
        Default is None.
    projection : child of :py:class:`cartopy.crs.Projection` or None
        The projection of this subplot. This cartopy projection is used to
        project given geo-referenced data into another coordinate system.
        If the projection is None, the data is not reprojected. Default is None.
    **ax_settings
        Variable keyword arguments dict, which is passed during axes creation
        to :py:class:`~matplotlib.axes.Axes`.

    Attributes
    ----------
    ax : :py:class:`~matplotlib.axes.Axes`,
        :py:class:`~cartopy.mpl.geoaxes.GeoAxes` or None
        The axes of this subplot. If this None, the subplot was not plotted
        yet.
    plotted : boolean
        If the subplot was already plotted once.
    layers : list(child of :py:class:`pylawr.plot.layer.base.BaseLayer`)
        The layers of this subplot. Plotting logic of these layers is called if
        :py:meth:`~pylawr.plot.subplot.Subplot.plot` is called.
    projection : child of :py:class:`cartopy.crs.Projection` or None
        The projection of this subplot. The projection is used during
        initialisation of :py:class:`~matplotlib.axes.Axes`. If a projection is
        set, the axes will be a :py:class:`~cartopy.mpl.geoaxes.GeoAxes` with
        given projection.
    ax_settings : dict
        Variable keyword arguments dict and is passed during axes creation to
        ``__init__`` of :py:class:`~matplotlib.axes.Axes`.
    extent_settings : dict
        Settings of the axes extent. This is used if the axes is a
        :py:class:`~cartopy.mpl.geoaxes.GeoAxes`, because cartopy disturbs the
        aspect ratio of the axes. The extent settings has `auto`, `lon_min`,
        `lon_max`, `lat_min`, `lat_max` and `projection` as keys.
    extent : dict(str, float)
        The extent of this axes in degrees, which sets the geographic border of
        this subplot. If ``auto_extent`` is set to True, this extent is modified
        for plotting purpose, but the original extent is returned. Extent is a
        dictionary with `lon_min`, `lon_max`, `lat_min` and `lat_max` as
        entries.
    auto_extent : bool
        Indicates if the axes extent should be extended automatically. If an
        extent is set, ``auto_extent`` will modify this extent so that the
        subplot has an aspect ratio as defined within grid slices. Default is
        True.
    """
    def __init__(self, layers_list=None, projection=None, **ax_settings):
        self._ax = None
        self._projection = None
        self._layers_list = None
        self._layers = layers_list
        self._extent_keys = ['lon_min', 'lon_max', 'lat_min', 'lat_max']
        self._extent_settings = dict(
            auto=True,
            lon_min=9.5,
            lon_max=10.5,
            lat_min=53,
            lat_max=54,
            projection=ccrs.PlateCarree()
        )
        self.projection = projection
        self.ax_settings = ax_settings

    def __getattr__(self, item):
        if self._ax is None:
            raise AttributeError('The given attribute is not available or this '
                                 'subplot was not plotted yet.')
        return getattr(self._ax, item)

    def __getitem__(self, item):
        return self.ax_settings[item]

    def __setitem__(self, key, value):
        self.ax_settings[key] = value

    def __delitem__(self, key):
        del self.ax_settings[key]

    def update(self, *args, **kwargs):
        return self.ax_settings.update(*args, **kwargs)

    @property
    def plotted(self):
        """
        Checks if plot method was already called once.

        Returns
        -------
        plotted : boolean
            If the subplot was already plotted once.
        """
        return isinstance(self._ax, mpl_subplots.Axes)

    @property
    def ax(self):
        """
        Get the axes for this subplot. This ax is created if this subplot was
        plotted once.

        Returns
        -------
        ax : :py:class:`~matplotlib.axes.Axes`,
        :py:class:`~cartopy.mpl.geoaxes.GeoAxes` or None
            The axes of this subplot. If this None, the subplot was not plotted
            yet.
        """
        return self._ax

    @property
    def _layers(self):
        return self._layers_list

    @_layers.setter
    def _layers(self, new_layers):
        if new_layers is None:
            self._layers_list = []
        elif not isinstance(new_layers, str) and \
                isinstance(new_layers, collections.abc.Iterable):
            self._layers_list = list(new_layers)
        else:
            raise TypeError('Given layers need to be an iterable, which can be '
                            'recasted to a list.')

    @property
    def layers(self):
        """
        Layers of this subplot. These layer will be plotted on this subplot.

        Returns
        -------
        layers : list(child of :py:class:`pylawr.plot.layer.base.BaseLayer`)
            The layers of this subplot. Plotting logic of these layers is called
            if :py:meth:`~pylawr.plot.subplot.Subplot.plot` is called.
        """
        return self._layers

    @property
    def extent_settings(self):
        """
        Get the current settings for the geographic extent of this subplot.

        Returns
        -------
        extent_settings : dict
            Settings of the axes extent. This is used if the axes is a
            :py:class:`~cartopy.mpl.geoaxes.GeoAxes`, because cartopy disturbs
            the aspect ratio of axes. The extent settings has `auto`,
            `lon_min`, `lon_max`, `lat_min`, `lat_max` and `projection` as keys.
        """
        return self._extent_settings

    @property
    def auto_extent(self):
        """
        Get boolean if the geographic extent of this subplot is automatically
        extended.

        Returns
        -------
        auto_extent : bool
            Indicates if the axes extent should be extended automatically. If an
            extent is set, ``auto_extent`` will modify this extent so that the
            subplot has an aspect ratio as defined within grid slices. Default
            is True.
        """
        return self._extent_settings['auto']

    @auto_extent.setter
    def auto_extent(self, new_auto):
        """
        Set the auto extent of this subplot.

        Parameters
        ----------
        new_auto : bool
            The new auto extent of this subplot.

        Warnings
        --------
        If a new auto_extent is set and the subplot was already plotted,
        :py:meth:``~pylawr.plot.subplot.Subplot.update_extent`` has to be
        called.

        """
        if not isinstance(new_auto, bool):
            raise TypeError('The given auto extent has to be a boolean')
        self._extent_settings['auto'] = new_auto

    @property
    def extent(self):
        """
        Get the geographic latitude and longitude extent for this subplot.

        Returns
        -------
        extent : dict(str, float)
            The extent of this axes in degrees, which sets the geographic border
            of this subplot. If ``auto_extent`` is set to True, this extent is
            modified for plotting purpose, but the original extent is returned.
            Extent is a dictionary with `lon_min`, `lon_max`, `lat_min` and
            `lat_max` as entries.
        """
        extent = {k: self._extent_settings[k] for k in self._extent_keys}
        return extent

    @extent.setter
    def extent(self, new_extent):
        """
        Update the extent of this subplot. All other keys than `lon_min`,
        `lon_max`, `lat_min` and `lat_max` are filtered out. The extent is
        always in ``longitude`` and ``latitude`` coordinates.

        Parameters
        ----------
        new_extent : dict(str, float)
            The new extent of this subplot in degrees. This dictionary is used
            to update the extent of this subplot.

        Warnings
        --------
        If a new extent is set and the subplot was already plotted,
        :py:meth:``~pylawr.plot.subplot.Subplot.update_extent`` has to be
        called.

        """
        if not isinstance(new_extent, dict):
            raise TypeError('The extent update is not a valid dictionary')
        filtered_extent = {k: ext for k, ext in new_extent.items()
                           if k in self._extent_keys}
        self._extent_settings.update(filtered_extent)

    @property
    def projection(self):
        """
        Get the projection of this subplot. This cartopy projection is used to
        project given geo-referenced data into another coordinate system. The
        projection is used during intialization of
        :py:class:`~matplotlib.axes.Axes`. If the projection is None, the data
        is not reprojected.

        Returns
        -------
        projection : child of :py:class:`cartopy.crs.Projection` or None
            The projection of this subplot. The projection is used during
            initialisation of :py:class:`~matplotlib.axes.Axes`. If a projection
            is set, the axes will be a :py:class:`~cartopy.mpl.geoaxes.GeoAxes`
            with given projection.
        """
        return self._projection

    @projection.setter
    def projection(self, new_proj):
        """
        Set the projection of this subplot. This cartopy projection is used to
        project given geo-referenced data into another coordinate system. The
        projection is used during intialization of
        :py:class:`~matplotlib.axes.Axes`. If the projection is None, the data
        is not reprojected.

        Parameters
        ----------
        new_proj : child of :py:class:`cartopy.crs.Projection` or None
            The new projection of this subplot. If the projection is not a valid
            cartopy projection or not None a TypeError is raised.

        """
        if not isinstance(new_proj, ccrs.Projection) and new_proj is not None:
            raise TypeError('Given projection is not a valid cartopy '
                            'projection and not None')
        self._projection = new_proj

    def add_layer(self, layer):
        """
        Add a given layer to the layers of this subplot.

        Parameters
        ----------
        layer : child of :py:class:`pylawr.plot.layer.base.BaseLayer`
            This layer will be added to the layers list of this subplot.

        """
        if not isinstance(layer, BaseLayer):
            raise TypeError('Given layer {0:s} is not a valid pylawr plotting '
                            'layer'.format(str(layer)))
        self._layers.append(layer)
        if self.plotted:
            self.plot_layer_on_ax(layer=layer)

    def swap_layer(self, new_layer, old_layer):
        """
        Swap an old layer for a new layer. This removes the old layer from this
        subplot and inserts the new layer at its place (including the zorder, if
        set). If this subplot was already plotted, the new layer will be plotted
        automatically on this subplot.

        Parameters
        ----------
        new_layer : child of :py:class:`pylawr.plot.layer.base.BaseLayer`
            This new layer will replace ``old_layer``.
        old_layer : child of :py:class:`pylawr.plot.layer.base.BaseLayer`
            The old layer will be replaced by ``new_layer``

        """
        if not isinstance(new_layer, BaseLayer) or \
                not isinstance(old_layer, BaseLayer):
            raise TypeError('Given ``new_layer`` and/or ``old_layer`` is no '
                            'valid pylawr layer')
        try:
            layer_index = self._layers.index(old_layer)
        except ValueError:
            raise KeyError(
                'Given ``old_layer`` was not found within the layers of this '
                'subplot. You can add ``new_layer`` to this subplot with '
                '``add_layer``.'
            )
        old_layer.remove()
        if old_layer.zorder is not None:
            new_layer.zorder = old_layer.zorder
        self._layers[layer_index] = new_layer
        if self.plotted:
            self.plot_layer_on_ax(new_layer)

    def _get_subplot_aspect(self, spec):
        gs_rows, gs_cols, start_num, stop_num = spec.get_geometry()
        grid_aspect = gs_cols / gs_rows
        y_min, x_min = divmod(start_num, gs_cols)
        y_max, x_max = divmod(stop_num, gs_cols)
        slice_aspect = (x_max-x_min+1) / (y_max-y_min+1)
        fig_aspect = self._ax.figure.get_figwidth() / \
                     self._ax.figure.get_figheight()
        subplot_aspect = slice_aspect / grid_aspect * fig_aspect
        return subplot_aspect

    def _calc_auto_extent(self, spec):
        """
        This method automatically calculate the extent based on current subplot
        position and size and current extent.

        Returns
        -------
        extent_tuple : tuple(float)
            The automatically calculated extent as tuple
            (`lon_min`, `lon_max`, `lat_min` and `lat_max`).
        """
        subplot_aspect = self._get_subplot_aspect(spec)
        extent = list(self._ax.get_extent())
        crs_width = extent[1]-extent[0]
        crs_height = extent[3]-extent[2]
        crs_aspect = crs_width / crs_height
        window_crs_ratio = subplot_aspect / crs_aspect
        if window_crs_ratio > 1:
            deg_ges = crs_width
            corr_deg = deg_ges * (window_crs_ratio - 1)
            extent[0] = extent[0] - corr_deg / 2
            extent[1] = extent[1] + corr_deg / 2
        else:
            corr = 1 / window_crs_ratio
            deg_ges = crs_height
            corr_deg = deg_ges * (corr - 1)
            extent[2] = extent[2] - corr_deg / 2
            extent[3] = extent[3] + corr_deg / 2
        return extent

    def update_extent(self, spec=None):
        """
        Update the extent of this subplot, if this has a valid
        :py:class:`~cartopy.mpl.geoaxes.GeoAxes`. The extent is updated based
        on :py:attr:`pylawr.plot.subplot.Subplot.extent_settings`.

        Parameters
        ----------
        spec : :py:class:`~matplotlib.gridspec.SubplotSpec` or None
            This spec is needed to adjust the extent automatically. Based on
            this extent, the aspect of this subplot is calculated.

        Raises
        ------
        TypeError
            A TypeError is raised if the axes is not a valid
            :py:class:`~cartopy.mpl.geoaxes.GeoAxes` or the subplot was not
            plotted yet.

        """
        if not self.plotted:
            raise TypeError('This subplot is not plotted yet and there is no '
                            'axes to update.')
        if not isinstance(self._ax, geoaxes.GeoAxes):
            raise TypeError('This subplot has no projection set and the axes '
                            'of this subplot is no GeoAxes.')
        try:
            extent_tuple = tuple(self.extent[k] for k in self._extent_keys)
            self._ax.set_extent(extent_tuple,
                                crs=self.extent_settings['projection'])
        except KeyError:
            pass
        if self.auto_extent:
            extent_auto_tuple = self._calc_auto_extent(spec)
            self._ax.set_extent(extent_auto_tuple, crs=self._projection)

    def new_axes(self, fig, spec=None, **ax_settings):
        """
        Create a new axes for this subplot.

        Parameters
        ----------
        fig : :py:class:`matplotlib.figure.Figure`
            The axes will be created on this figure. The figure needs a valid
            canvas to be drawn.
        spec : any, optional
            This argument determines the location of the subplot on given
            figure. Default is None.
        **ax_settings
            Variable keyword arguments dict, which is passed during
            axes creation to :py:class:`~matplotlib.axes.Axes`.

        """
        self._ax = fig.add_subplot(spec, projection=self._projection,
                                   **ax_settings)
        self._ax.tick_params(**default_tick_params)

    def plot_layer_on_ax(self, layer):
        """
        Plot given layer on this subplot. If the subplot was not plotted yet a
        ValueError will be raised. This method can be used to switch a layer of
        this subplot.

        Parameters
        ----------
        layer : child of :py:class:`pylawr.plot.layer.base.BaseLayer`
            This layer will be plotted on this subplot. This layer does not need
            to be a layer within this subplot. If the layer is not a valid
            pylawr plotting layer a TypeError will be raised.

        Warnings
        --------
        To call this method plot has to be called beforehand.
        """
        if not isinstance(layer, BaseLayer):
            raise TypeError('Given layer {0:s} is not a valid pylawr plotting '
                            'layer'.format(str(layer)))
        if not self.plotted:
            raise ValueError('This subplot is not plotted yet on a figure, '
                             'please call first ``plot``')
        layer.plot(ax=self.ax)

    def plot(self, fig, spec=None):
        """
        Plot this subplot. A new axes is created on given figure with given
        subplot spec. All layers of this subplot are plot on this axes.
        :py:attr:`~pylawr.plot.subplot.Subplot.ax_settings` are passed to
        :py:meth:`~matplotlib.figure.Figure.add_subplot` as additional
        arguments. The extent isa djusted according to ``extent_settings`` of
        this subplot.

        Parameters
        ----------
        fig : :py:class:`matplotlib.figure.Figure`
            The axes is created on this figure.
        spec : any, optional
            This argument determines the location of the subplot on given
            figure. Default is None.

        """
        self.new_axes(fig=fig, spec=spec, **self.ax_settings)
        for layer in self.layers:
            self.plot_layer_on_ax(layer=layer)
        if isinstance(self._ax, geoaxes.GeoAxes):
            self.update_extent(spec=spec)
