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
import importlib
import tempfile
import os

# External modules
import matplotlib.figure as mpl_figure
import matplotlib.gridspec as mpl_gridspec

# Internal modules
from pylawr.plot.layer.base import BaseLayer
from pylawr.plot.backend import BackendLoader
from pylawr.plot.subplot import Subplot
from pylawr.utilities.decorators import tuplesetter


logger = logging.getLogger(__name__)


default_grid_slices = dict(
    map=(slice(None, None), slice(None, None))
)


class NotPlottedError(Exception):
    pass


class Plotter(object):
    """
    Plotter is used as an easy access for the plotting interface. The plotter
    creates a figure and canvas during plotting- These two are holding the
    subplots and the layers, where the plotting logic is hided. The figure
    settings can be set in a dict-like manner or via manipulation of
    `fig_settings` dict. After the plotter is plotted, all attributes from
    :py:class:`matplotlib.figure.Figure` can be accessed as if the plotter is a
    figure.

    Parameters
    ----------
    backend_name : str, optional
        This backend will be loaded and used for plotting. The backend is loaded
        dynamically and will create a canvas which is used to plot the figure.
        Default is `agg`.
    grid_size : tuple, optional
        Figure's grid size. This grid size is used to create a gridspec for
        given figure.
    grid_slices : dict(str, tuple(slice, slice)) or None, optional
        For every grid slice a subplot is created. During creation, the key is
        used as name and the slices (row slice, column slice) indicates,
        where the subplot is located at the plot. For every key a new subplot is
        created.
    **fig_settings
        Variable keyword arguments dict, which is passed during plotting
        figure creation to :py:class:`~matplotlib.figure.Figure`.
    """
    def __init__(self, backend_name='agg', grid_size=(14, 14), grid_slices=None,
                 **fig_settings):
        self._backend = BackendLoader(name=backend_name)
        self._figure = None
        self._canvas = None
        self._grid_slices = {}
        self._grid_size = None
        self._subplots = {}
        self.gridspec_settings = dict(
            hspace=0, wspace=0, bottom=0.02, top=0.98, left=0.02, right=0.98
        )
        self.grid_size = grid_size
        self.fig_settings = fig_settings
        self._add_grid_slices_to_subplot(grid_slices)

    def __getattr__(self, key):
        if self._figure is None:
            raise AttributeError('The given attribute is not available or this '
                                 'plotter was not plotted yet.')
        return getattr(self._figure, key)

    def __getitem__(self, item):
        return self.fig_settings[item]

    def __setitem__(self, key, value):
        self.fig_settings[key] = value

    def __delitem__(self, key):
        del self.fig_settings[key]

    def update(self, *args, **kwargs):
        return self.fig_settings.update(*args, **kwargs)

    @property
    def backend_name(self):
        """
        Get the name of the used backend.

        Returns
        -------
        backend_name : str
            This backend is used for plotting purpose.
        """
        return self._backend.name

    @backend_name.setter
    def backend_name(self, new_name):
        """
        Set the name of the backend. This sets a new backend, which is then used
        for plotting.

        Parameters
        ----------
        new_name : str
            The name of the new backend, which will be used.

        Warnings
        --------
        If a new backend is set after plotting,
        :py:meth:`~pylawr.plot.plotter.Plotter.plot` has to be called again.
        """
        self._backend.name = new_name

    @property
    def figure(self):
        """
        Get the currently opened figure.

        Returns
        -------
        figure : :py:class:`matplotlib.figure.Figure`
            The currently opened figure.
        """
        return self._figure

    @property
    def grid_size(self):
        """
        Return the size of figure's grid.

        Returns
        -------
        grid_size : tuple(int)
            The grid size as tuple (n_rows, n_cols)
        """
        return self._grid_size

    @tuplesetter(grid_size, valid_types=(int, ))
    def grid_size(self, new_grid_size):
        """
        Set the new grid size of figure's grid.

        Parameters
        ----------
        new_grid_size : int or tuple(int)
            The new grid size as int or tuple. A tuple will be shortened to two
            entries (n_rows, n_cols). An integer will be converted to
            a tuple with the same number of rows and columns.
        """
        return new_grid_size

    @property
    def gridspec(self):
        """
        Get an initialized :py:class:`~matplotlib.gridspec.Gridspec` with set
        grid size. ``gridspec_settings`` attribute is passed as kwargs to
        :py:class:`~matplotlib.gridspec.Gridspec`.

        Returns
        -------
        gridspec : :py:class:`matplotlib.gridspec.Gridspec`
            Initialized gridspec with ``grid_size`` as geometry.
        """
        gridspec = mpl_gridspec.GridSpec(
            *self.grid_size, **self.gridspec_settings
        )
        return gridspec

    @property
    def grid_slices(self):
        return self._grid_slices

    def _add_grid_slices_to_subplot(self, grid_slices):
        if grid_slices is None:
            grid_slices = default_grid_slices
        if isinstance(grid_slices, dict):
            for name, grid_slice in grid_slices.items():
                self.add_subplot(name, grid_slice)
        else:
            raise TypeError('The given grid slices are not a dictionary')

    @property
    def plotted(self):
        """
        Check if plot was already called.

        Returns
        -------
        plotted : bool
            If plot was called at least once.
        """
        return isinstance(self._figure, mpl_figure.Figure)

    @property
    def subplots(self):
        return self._subplots

    def add_subplot(self, name,
                    grid_slice=(slice(None, None), slice(None, None)),
                    **ax_settings):
        """
        Adds a subplot to the plotter. This subplot is added to the subplots
        dict. Also, given slice is added to grid slice dict with given name as
        key.

        Parameters
        ----------
        name : str
            The name of the new subplot. If there is already a subplot with this
            name within the subplot dict, a KeyError will be raised.
        grid_slice : tuple(slice, slice)
            This grid slice specifies where the subplot is located on the
            figure. It has to be a tuple with two slices (row position,
            column position). This slice is added to the `grid_slices` dict.
        **ax_settings
            Variable keyword arguments dict, which is passed to the subplot.
        """
        if name in self.subplots.keys():
            raise KeyError('A subplot with given name already exists!')
        self._grid_slices[name] = grid_slice
        self.subplots[name] = Subplot(**ax_settings)
        if self.plotted:
            self._plot_subplot(name)
        logger.debug('Added {0:s} to subplots with {1} as grid slices'.format(
            name, grid_slice
        ))

    def del_subplot(self, name):
        """
        Remove given subplot out of the plotter. The subplot will be removed
        from subplots and grid_slices dictionary.

        Parameters
        ----------
        name : str
            Subplot which should be removed.
        """
        if self.plotted:
            self._subplots[name].ax.remove()
        self._subplots.pop(name)
        self._grid_slices.pop(name)
        logger.debug('Removed {0:s} from subplots'.format(name))

    def change_grid_slice(self, name, new_grid_slice):
        """
        Change the grid slice of a given subplot. The grid slice within grid
        slices dict is changed for given subplot name as key. If the plotter
        was already plotted, the subplot is replotted.

        Parameters
        ----------
        name : str
            For this subplot the grid slice is changed. If this subplot name
            cannot be found within the grid slices dict a KeyError is raised.
        new_grid_slice : tuple(slice, slice)
            This grid slice is set as new grid slice of given subplot. It has to
            be a tuple with two slices (row position, column position).
        """
        if name not in self._grid_slices.keys():
            raise KeyError(
                'Given subplot {0:s} is not available, available subplots are: '
                '{1:s}'.format(name, ','.join(self._grid_slices.keys()))
            )
        self._grid_slices[name] = new_grid_slice
        if self.plotted:
            self._subplots[name].ax.remove()
            self._plot_subplot(name)
        logger.debug('Changed the grid slice for {0:s} from {1} to {2}'.format(
            name, self._grid_slices[name], new_grid_slice)
        )

    def add_layer(self, subplot_name, layer):
        """
        Adds a layer to given subplot name. This layer will be added to the
        layer list of the subplot. The subplot has to exist beforehand.

        Parameters
        ----------
        subplot_name : str
            The layer is added to the layer list of the subplot corresponding to
            this subplot name. If the subplot does not exist yet, a KeyError
            will be raised.
        layer : child of :py:class:`pylawr.plot.layer.base.BaseLayer`
            This layer is added to given subplot name.
        """
        if subplot_name not in self.subplots.keys():
            err_msg = 'The given subplot {0:s} does not exist yet, available ' \
                      'subplots are: {1:s}'.format(
                        subplot_name, ','.join(self.subplots.keys())
                      )
            raise KeyError(err_msg)
        self.subplots[subplot_name].add_layer(layer)

    def _get_subplot_name_from_layer(self, layer):
        """
        Get the corresponding subplot name from given layer.

        Parameters
        ----------
        layer : :py:class:`pylawr.plot.layer.base.BaseLayer`
            The subplot name to this layer is searched.

        Returns
        -------
        subplot_name : str
            The found subplot name to given layer.

        Raises
        ------
        KeyError
            If no corresponding subplot was found to given layer.
        """
        if not isinstance(layer, BaseLayer):
            raise TypeError('The given layer is not a valid pylawr layer')
        subplot_names = [name for name, subplot in self._subplots.items()
                         if layer in subplot.layers]
        try:
            subplot_name = subplot_names[0]
        except IndexError:
            raise KeyError('No corresponding subplot was found to given layer')
        return subplot_name

    def swap_layer(self, new_layer, old_layer=None, layer_num=None,
                   zorder=None):
        """
        An already set layer will be swapped with a given new_layer. The layer
        to replace has to be specified with one of three different arguments
        (in descending priority):

        * `old_layer` : replace given old layer with given new layer
        * `layer_num` : replace given old layer to the corresponding layer
            number with given new layer
        * `zorder` : replace given old layer to the corresponding z-order with
            given new layer

        Parameters
        ----------
        new_layer : child of :py:class:`pylawr.plot.layer.base.BaseLayer`
            This layer will replace the specified old layer. If the plotter was
            already plotted this layer will be also plotted.
        old_layer : child of :py:class:`pylawr.plot.layer.base.BaseLayer` or
            None
            If this old layer is given, it will be searched within the subplots
            and replaced by the new layer. If it is not found within the
            subplots, a ValueError will be raised. If it is None, either
            `layer_num` or `zorder` has to be specified. Default is None.
        layer_num : tuple(str, int) or None
            The i-th layer within specified subplot is replaced. The first entry
            of the tuple identifies the subplot, while the integer specifies the
            layer number within given subplot. A KeyError will be raised, if the
            layer number cannot be found within specified subplot. If this is
            None, either `old_layer` or `zorder` has to be specified. Default is
            None.
        zorder : tuple(str, int) or None
            The layer with given z-order within specified subplot is replaced.
            The first entry of the tuple identifies the subplot, while the
            integer specified the z-order within given subplot. A KeyError will
            be raised, if the z-order cannot be found within specified subplot.
            If this is None, either `old_layer` or `layer_num` has to be
            specified. The z-order has to be Default is None.

        """
        if old_layer is not None:
            subplot_name = self._get_subplot_name_from_layer(old_layer)
        elif isinstance(layer_num, tuple):
            subplot_name = layer_num[0]
            old_layer = self._subplots[subplot_name].layers[layer_num[1]]
        elif isinstance(zorder, tuple):
            subplot_name = zorder[0]
            subplot_layers = self._subplots[subplot_name].layers
            zorder_layers = {l.zorder: l for l in subplot_layers}
            old_layer = zorder_layers[zorder[1]]
        else:
            raise TypeError('No old_layer, layer number or zorder in right '
                            'type was given')
        self._subplots[subplot_name].swap_layer(new_layer=new_layer,
                                                old_layer=old_layer)

    def new_figure(self, **fig_settings):
        """
        Creates a new figure and canvas for this figure based on currently
        selected backend.

        Parameters
        ----------
        **fig_settings
            Variable keyword arguments dict, which is passed during
            figure creation to :py:class:`~matplotlib.figure.Figure`.
        """
        self._figure = mpl_figure.Figure(**fig_settings)
        self._canvas = self._backend.canvas(self._figure)
        logger.debug('Created a new figure and canvas')

    def _plot_subplot(self, subplot_name):
        if subplot_name not in self._subplots.keys():
            err_msg = 'The given subplot {0:s} does not exist yet, available ' \
                      'subplots are: {1:s}'.format(
                        subplot_name, ','.join(self._subplots.keys())
                      )
            raise KeyError(err_msg)
        if not self.plotted:
            raise NotPlottedError('This plotter is not plotted yet, '
                                  'please call ``plot`` firstly.')
        subplot_slices = self._grid_slices[subplot_name]
        subplot_spec = self.gridspec[subplot_slices[0], subplot_slices[1]]
        self._subplots[subplot_name].plot(fig=self._figure, spec=subplot_spec)

    def plot(self):
        """
        Plot a new figure and all subplots and layers on this figure. A new
        figure and canvas are created. The figure is passed to all subplots such
        that all subplots can use this figure to plot their layers on this
        figure. All set figure settings are passed to the new figure.

        Warnings
        --------
        This method creates a new figure such that the old figure and canvas are
        deleted.
        """
        self.new_figure(**self.fig_settings)
        _ = [self._plot_subplot(name) for name in self.subplots.keys()]
        logger.debug('Plotted with {0:s} as subplots'.format(
            ','.join(self.subplots.keys()))
        )

    # pragma: no cover
    def show(self):
        """
        This method is used to show this plotter as window. Our plotter is based
        on the object-oriented interface of matplotlib without
        :py:mod:`~matplotlib.pyplot`. We therefore need to save this plotter as
        png image to a temporary file. This is then showed with
        :py:mod:`~matplotlib.pyplot`.

        Warnings
        --------
        This method can be a little bit slow because this plotter is saved and
        replotted. Further, :py:mod:`~matplotlib.pyplot` is imported, which can
        be a problem for non-interactive shells and as a consequence, this
        method is not tested with unittests.
        """
        if not self.plotted:
            raise NotPlottedError('This plotter is not plotted yet, '
                                  'please call ``plot`` firstly.')
        plt = importlib.import_module('matplotlib.pyplot')
        temp_file_name = '{0:s}.png'.format(
            next(tempfile._get_candidate_names())
        )
        self.savefig(temp_file_name)
        array = plt.imread(temp_file_name)
        plt.figure(**self.fig_settings)
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.imshow(array)
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.autoscale(tight=True)
        plt.show()
        os.remove(temp_file_name)
