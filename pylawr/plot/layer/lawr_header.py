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
from copy import deepcopy

# External modules
import numpy as np

# Internal modules
from .base import BaseLayer

logger = logging.getLogger(__name__)


class LawrHeaderLayer(BaseLayer):
    """
    A LawrHeaderLayer adds a header layer. This header layer has three
    specific area, which can be used (title, left, right):

    .. code::

        ---------------------------------
        ------------- TITLE -------------
        ---------------------------------
        +++++++++++++++++++++++++++++++++
        ----------------+----------------
        ---- LEFT ------+---- RIGHT -----
        ----------------+----------------

    The different areas are splitted by lines. ``left`` and ``right`` are a
    dictionary where the entries are display as `key`: `value`. The title can be
    set as str.

    Parameters
    ----------
    zorder : int
        The z-order defines the drawing order of the layers. If no z-order
        is set, the z-order will be set during plotting based on layer order
        in subplot. Default is 0.
    **settings
        Variable keyword arguments dict, which is passed during plotting to
        every :py:meth:`~matplotlib.axes.Axes.text` call. Title and info box
        specific settings can be set with ``title_settings`` and
        ``info_settings``.

    Attributes
    ----------
    zorder : int
        The drawing order of this layer. If not set, it will be
        automatically determined during plotting by subplot.
    settings : dict
        The settings for this layer. These settings can be accessed directly
        and via a dict like interface of this layer.
    title : str
        This title is used as title for this header layer.
    left : dict(str, str)
        This dictionary is plotted to the left information box within this
        header. If the order should be preserved, you can also pass an
        :py:class:`~collections.OrderedDict`.
    right : dict(str, str)
        This dictionary is plotted to the right information box within this
        header. If the order should be preserved, you can also pass an
        :py:class:`~collections.OrderedDict`.
    text_padding : float
        This padding is used to generate space to the left position of the
        text. This is only applied to the information box texts. A positive
        value will move the text to the right, while a negative value to the
        left.
    line_settings : dict
        These specific settings are passed to the creation of the lines to
        create the three different visually separated boxes. All valid
        kwargs of :py:class:`matplotlib.lines.Line2D` except `transform` and
        `zorder` can be used.
    title_settings : dict
        These specific settings are used during creation of the title and
        is passed to :py:meth:`~matplotlib.axes.Axes.text`. This settings
        dict has a higher priority than the normal settings dict and is used
        to update the normal settings dict for the title. All valid kwargs
        of :py:class:`matplotlib.text.Text` except `zorder` can be used.
    info_settings : dict
        These specific settings are used during creation of the left and
        right information box and is passed to
        :py:meth:`~matplotlib.axes.Axes.text`. These settings have a
        higher priority than the normal settings dict and is used to update
        the normal settings dict for the title. All valid kwargs of
        :py:class:`matplotlib.text.Text` except `zorder` can be used.
    """
    def __init__(self, zorder=0, **settings):
        super().__init__(zorder, **settings)
        self.title = 'title missing'
        self.left = {
            'left_info': 'every entry will create a new line',
            'key_l': 'value_l'
        }
        self.right = {
            'right_info': 'every entry will create a new line',
            'key_r': 'value_r'
        }
        self.text_padding = 0.05
        self.line_settings = dict(
            color='k', lw=0.5
        )
        self.title_settings = dict(
            va='center', ha='center',
        )
        self.info_settings = dict(
            va='center', ha='left'
        )

    def _plot_lines_on_ax(self, ax):
        line_horizontal = ax.axhline(y=2/3, xmin=0, xmax=1, zorder=self.zorder,
                                     **self.line_settings)
        line_vertical = ax.axvline(x=0.5, ymin=0, ymax=2/3, zorder=self.zorder,
                                   **self.line_settings)
        return line_horizontal, line_vertical

    def _plot_text_on_ax(self, ax, x, y, text, **specific_settings):
        ax_coords = ax.transAxes
        text_settings = deepcopy(self.settings)
        text_settings.update(specific_settings)
        plotted_text = ax.text(x=x, y=y, s=text, zorder=self.zorder,
                               transform=ax_coords, **text_settings)
        return plotted_text

    def _plot_title_on_ax(self, ax):
        text_title = self._plot_text_on_ax(ax=ax, x=0.5, y=5/6, text=self.title,
                                           **self.title_settings)
        return text_title

    def _plot_info_box_on_ax(self, ax, info_dict, x_pos,):
        y_positions = np.linspace(2/3, 0, len(info_dict)+2)[1:-1]
        text_list = []
        for text_num, key in enumerate(info_dict.keys()):
            text_to_plot = '{0:<5}: {1:<40}'.format(key, info_dict[key])
            y_pos = y_positions[text_num]
            plotted_text = self._plot_text_on_ax(
                ax=ax, x=x_pos, y=y_pos, text=text_to_plot, **self.info_settings
            )
            text_list.append(plotted_text)
        return text_list

    def plot(self, ax):
        """
        This header layer is plotted on given axes. The lines are plotted
        firstly, while texts are plotted afterwards. All are plotted with
        set zorder on axes.

        Parameters
        ----------
        ax : :py:class:`matplotlib.axes.Axes`
            The header is plotted with set zorder on this subplot.
        """
        line_h, line_v = self._plot_lines_on_ax(ax=ax)
        text_title = self._plot_title_on_ax(ax=ax)
        text_left_list = self._plot_info_box_on_ax(ax=ax, info_dict=self.left,
                                                   x_pos=0+self.text_padding)
        text_right_list = self._plot_info_box_on_ax(ax=ax, info_dict=self.right,
                                                    x_pos=0.5+self.text_padding)
        self._collection.append(line_h)
        self._collection.append(line_v)
        self._collection.append(text_title)
        self._collection.extend(text_left_list)
        self._collection.extend(text_right_list)
