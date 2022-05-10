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
import os

# External modules
import matplotlib.colorbar as mpl_colorbar

# Internal modules
from .base import BaseLayer
from .radarfield import RadarFieldLayer

logger = logging.getLogger(__name__)


class ColorbarLayer(BaseLayer):
    """
    The colorbar layer is used as additional colorbar to given radar field
    layer. This colorbar is plotted on given axes and can there be on
    another axes than the radar field.

    Parameters
    ----------
    layer : :py:class:`~pylawr.plot.layer.radarfield.RadarFieldLayer`
        The colorbar will be created for this radar field layer.
        :py:attr:`pylawr.plot.layer.radarfield.RadarFieldLayer.plot_store`
        is used to get the colormap from this layer.
    zorder : int
        The z-order defines the drawing order of the layers. If no z-order
        is set, the z-order will be set during plotting based on layer order
        in subplot. Default is 0.
    **settings
        Variable keyword arguments dict, which is passed during plotting of
        :py:meth:`~matplotlib.figure.Figure.colorbar`. All possible kwargs
        of :py:meth:`~matplotlib.figure.Figure.colorbar` can used as keyword
        argument.

    Attributes
    ----------
    layer : :py:class:`~pylawr.plot.layer.radarfield.RadarFieldLayer`
        The colorbar will be created for this radar field layer during
        plotting.
        :py:attr:`pylawr.plot.layer.radarfield.RadarFieldLayer.plot_store`
        is used to get the colormap from this layer.
    zorder : int
        The drawing order of this layer. If not set, it will be
        automatically determined during plotting by subplot.
    settings : dict
        The settings for this layer. It is passed to
        :py:meth:`~matplotlib.figure.Figure.colorbar`. These settings can be
        accessed directly and via a dict like interface of this layer. All
        possible kwargs of :py:meth:`~matplotlib.figure.Figure.colorbar` can
        set to this dictionary.
    h_pad : float
        The horizontal padding of the colorbar to given axes in percentage
        [0, 1] of the axes width. The horizontal padding is the distance to
        the left and right border of the axes and the colorbar width
        therefore depends on this padding. Default is 0.35.
    v_pad : float
        The vertical padding of the colorbar to given axes in percentage
        [0, 1] of the axes height. The vertical padding is the distance to
        the top and and border of the axes and the colorbar height therefore
        depends on this padding. Default is 0.05
    colorbar : :py:class:`matplotlib.colorbar.Colorbar` or None
        The plotted colorbar accessible with this attribute. This attribute
        can be used to manipulate the colorbar directly. This layer is not
        plotted yet, if this attribute is None.
    """
    def __init__(self, layer, zorder=0, **settings):
        super().__init__(zorder, **settings)
        self._layer = None
        self.layer = layer
        self.colorbar = None
        self.h_pad = 0.35
        self.v_pad = 0.05

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layer):
        if not isinstance(layer, RadarFieldLayer):
            raise TypeError(
                'The given layer is not a valid pylawr.plot radar field layer. '
                'A colorbar can only be created for a radar field layer.'
            )
        self._layer = layer

    def plot(self, ax):
        """
        Adds a colorbar to given axes. The colorbar is sized and moved so that
        it has :py:attr:`~pylawr.plot.layer.colorbar.Colorbar.h_pad` and
        :py:attr:`~pylawr.plot.layer.colorbar.Colorbar.v_pad` as paddings. All
        other colorbar settings are set based on
        :py:attr:`~pylawr.plot.layer.colorbar.Colorbar.settings`. The set
        :py:attr:`~pylawr.plot.layer.colorbar.Colorbar.zorder` is used as zorder
        for the colorbar axes.

        Parameters
        ----------
        ax : :py:class:`matplotlib.axes.Axes`
            Parent axes for the colorbar. The colorbar will be plotted on this
            axes but will have its own colorbar axes. The position and zorder of
            this axes is not changed during plotting.
        """
        box = ax.get_position()
        h_pad_pix = self.h_pad * box.width
        v_pad_pix = self.v_pad * box.height
        colorbar = ax.figure.colorbar(
            mappable=self.layer.plot_store, ax=ax, **self.settings
        )
        colorbar.ax.set_position([
            box.x0 + h_pad_pix, box.y0 + v_pad_pix,
            box.width - 2 * h_pad_pix, box.height - 2 * v_pad_pix
        ])
        colorbar.ax.zorder = self.zorder
        ax.set_position(box)
        self.colorbar = colorbar
        self.collection.append(colorbar)
