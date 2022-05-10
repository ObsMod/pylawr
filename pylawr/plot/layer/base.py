#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 04.09.17
#
# Created for pattern
#
#
#    Copyright (C) {2017}
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
import abc

# External modules
import cartopy.crs as ccrs

# Internal modules

logger = logging.getLogger(__name__)


colorbar = {
    'alpha': 1
}

pcolormesh = {
    'edgecolor': None,
    'alpha': 1,
    'transform': ccrs.PlateCarree(),
}

header_lines = {
    'lw': .5,
}

default_settings = {
    'pcolormesh': pcolormesh,
    'colorbar': colorbar,
    'header_lines': header_lines,
    'text': {},
    'add_osm': {},
    'osm_resolution': 12,
    'header_spacing': .05,
    'aspect': 10,
    'colorbar_h_pad': .35,
    'colorbar_v_pad': .05,
}


class BaseLayer(object):
    """
    The BaseLayer is a class for all layer.
    It contains the most basic layer functions and parameter.

    Parameters
    ----------
    zorder : int
        The z-order defines the drawing order of the layers. If no z-order
        is set, the z-order will be set during plotting based on layer order
        in subplot. Default is 0.
    **settings
        Variable keyword arguments dict, which is passed during plotting to
        the plotting logic and which sets the plotting behaviour of this
        layer.

    Attributes
    ----------
    zorder : int
        The drawing order of this layer. If not set, it will be
        automatically determined during plotting by subplot.
    settings : dict
        The settings for this layer. These settings can be accessed directly
        and via a dict like interface of this layer.
    """
    def __init__(self, zorder=0, **settings):
        self.zorder = zorder
        self.settings = settings
        self._collection = []

    def __getitem__(self, item):
        return self.settings[item]

    def __setitem__(self, key, value):
        self.settings[key] = value

    def __delitem__(self, key):
        del self.settings[key]

    def update(self, *args, **kwargs):
        return self.settings.update(*args, **kwargs)

    @property
    def collection(self):
        """
        Get all elements which are plotted on given axes. This
        collection is populated after plot was called once. It is used by
        :py:meth:`pylawr.plot.layer.base.BaseLayer.remove` to remove this
        layer from axes.

        Returns
        -------
        collection : list
            A list with all elements which are plotted. If the list is empty,
            plot was not called yet.
        """
        return self._collection

    @abc.abstractmethod
    def plot(self, ax):
        """"
        This method contains the main plot logic of this layer. For plot layer
        prototyping this plot method has to be overwritten.

        Parameters
        ----------
        ax : :py:class:`matplotlib.axes.Axes`
            This layer will be plotted on this given axes.
        """
        pass

    def remove(self):
        """
        All plotted elements of this layer are removed. Further, the plot
        collection is emptied to an empty list.
        """
        for c in self.collection:
            c.remove()
        self._collection = []
