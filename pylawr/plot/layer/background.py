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

# External modules
from cartopy.io.img_tiles import Stamen

# Internal modules
from .base import BaseLayer

logger = logging.getLogger(__name__)


class BackgroundLayer(BaseLayer):
    """
    This can be used to plot a cartopy image tile on given axes. As default
    open street map data is used but it is possible to change the image
    tile.


    Parameters
    ----------
    zorder : int
        The z-order defines the drawing order of the layers. If no z-order
        is set, the z-order will be set during plotting based on layer order
        in subplot. Default is 0.
    resolution : int
        The zoom level for given image tile. Default is 10.
    **settings
        Variable keyword arguments dict, which is passed during plotting to
        the plotting logic and which sets the plotting behaviour of this
        layer.

    Attributes
    ----------
    zorder : int
        The drawing order of this layer. If not set, it will be
        automatically determined during plotting by subplot.
    resolution : int
        The zoom level for given image tile.
    settings : dict
        The settings for this layer. It is passed to
        :py:meth:`cartopy.mpl.geoaxes.GeoAxes.add_image`. These settings can
        be accessed directly and via a dict like interface of this layer.
    img_tile
        This image tile is plotted to given axes during
        :py:meth:`~pylawr.plot.layer.background.BackgroundLayer.plot`. An
        :py:class`~cartopy.io.img_tiles.OSM` object is already initialized
        as default.
    """
    def __init__(self, zorder=0, resolution=10, **settings):
        super().__init__(zorder, **settings)
        self.resolution = resolution
        self.img_tile = Stamen(style='toner-lite')

    def plot(self, ax):
        """
        The background image is plotted on this GeoAxes with set zorder.
        :py:meth:`cartopy.mpl.geoaxes.GeoAxes.add_image` is called with
        ``img_tile`` as factory for plotting.

        Parameters
        ----------
        ax : :py:class:`cartopy.mpl.geoaxes.GeoAxes`
            Used to determine the axis to plot on.
        """
        ax.add_image(self.img_tile, self.resolution, zorder=self.zorder,
                     **self.settings)

        ax.annotate('Map tiles by Stamen Design, '
                    'under CC BY 3.0. Data by '
                    'OpenStreetMap, '
                    'under ODbL.',
                    xy=(5., 5.),
                    xycoords='axes pixels',
                    bbox=dict(boxstyle='square,pad=0',
                              fc='white', ec="none", alpha=.8),
                    zorder=1000)

        self._collection = ax.get_images()
