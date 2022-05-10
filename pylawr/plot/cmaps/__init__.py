#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 31.01.2018
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

# External modules
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Internal modules
from . import color_list


available_cmaps = {}


def prepare_cmap(name, cdict):
    """
    Function to prepare a color map with given color dict.

    Parameters
    ----------
    name : str
        The name of the created color map.
    cdict : dict
        The colormap will be based on this dictionary with color values from
        0 to 255.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        The prepared color map with the given name and the given normalized
        color values.
    """
    normalized_cdict = {k: np.asarray(v) / 255 for k, v in cdict.items()}
    cmap = LinearSegmentedColormap(name, normalized_cdict)
    return cmap


color_vars = [var for var in color_list.__dict__.keys() if var[:2] != '__']
for var in color_vars:
    color_dict = getattr(color_list, var)
    color_map = prepare_cmap(var, color_dict)
    locals()[var] = color_map
    available_cmaps[var] = color_map

__all__ = list(available_cmaps.keys())
