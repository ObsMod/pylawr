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

# External modules
import matplotlib as mpl

# Internal modules


logger = logging.getLogger(__name__)


class BackendLoader(object):
    """
    This load is used to load dynamically a backend for the plotter. The figure
    will be plotted on the canvas of the loaded backend.

    Parameters
    ----------
    name : str
        The corresponding backend to this name will be loaded.
    """
    def __init__(self, name='agg'):
        self._backend_path_template = 'matplotlib.backends.backend'
        self._backend = None
        self._name = None
        self.name = name

    def _load_backend(self):
        module_name = '{0:s}_{1:s}'.format(
            self._backend_path_template, self.name.lower()
        )
        self._backend = importlib.import_module(module_name)

    @property
    def name(self):
        """
        The name of the backend in it corrected form.

        Returns
        -------
        name : str
            Backend's name.
        """
        return self._name

    @name.setter
    def name(self, new_name):
        corrected_name = mpl.rcsetup.validate_backend(new_name)
        self._name = corrected_name
        self._load_backend()

    @property
    def canvas(self):
        """
        The canvas of the loaded backend.

        Returns
        -------
        canvas : child of :py:class:`matplotlib.backend_bases.FigureCanvasBase`
            This canvas can be used as figure holder for object oriented
            matplotlib programming.
        """
        canvas = getattr(self._backend, 'FigureCanvas')
        return canvas
