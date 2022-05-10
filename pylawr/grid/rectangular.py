#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import abc

# External modules
import numpy as np

# Internal modules
from .base import BaseGrid
from pylawr.utilities.decorators import tuplesetter


logger = logging.getLogger(__name__)


class RectangularGrid(BaseGrid):
    """
    The RectangularGrid can be used for easier composite calculations of
    radar fields. The calculated coordinates are valid for the centroids
    of the grid points.

    Parameters
    ----------
    resolution : tuple(float or int), float or int, optional
        The resolution of the RectangularGrid. If the
        parameter is float or int, then the resolution will be set to equal
        values for both coordinates. Only the first two entries of a tuple
        will be used. Default is 100.
    start : tuple(float or int), float or int, optional
        The start position of the lower left grid point centroid within the
        RectangularGrid.If the parameter is float or int, then the
        resolution will be
        set to equal values for booth coordinates. Only the first two
        entries of a tuple will be used. Default is -5000.
    nr_points : tuple(int) or int, optional
        The number of points for both directions. If the tuple is int, then
        the resolution will be set to equal values for both coordinates.
        Only the first two entries of a tuple will be used. Default is 500.
    center : tuple(float), optional
        The center of the grid. The center should have two or three entries
        within the tuple. The first entry is the latitude position of the
        grid center in degree. The second entry is the longitude position of
        the grid center in degree. The optional third entry is the height of
        the center above mean sea level in meters. If no third entry is
        given the center height is set to zero. The center is
        used to transform the coordinates into latitude and longitude
        coordinates. Default is the position on  the top of the
        "Geomatikum": (53.56796, 9.97451, 95).
    beam_ele : float, optional
        The beam elevation in degrees. The beam elevation is used to
        calculate the altitude map. The beam elevation is set constant if
        it is a float. The value of the beam elevation are normalized for
        angle between -90 and 90 degrees such that the sign is preserved.
        This beam elevation is used to calculate the elevation for a grid
        point. The calculated elevation is valid for the whole grid box.
    """
    def __init__(self, resolution=100, start=-5000, nr_points=500,
                 center=(53.56796, 9.97451, 95), beam_ele=3):
        super().__init__(center, beam_ele)
        self._start = None
        self._resolution = None
        self._nr_points = None
        self.resolution = resolution
        self.start = start
        self.nr_points = nr_points

    @property
    def resolution(self):
        return self._resolution

    @tuplesetter(resolution, valid_types=(int, float))
    def resolution(self, new_resolution):
        return new_resolution

    @property
    def _data_shape(self):
        return self._nr_points

    @_data_shape.setter
    def _data_shape(self, new_shape):
        self._nr_points = new_shape

    @property
    def nr_points(self):
        return self._nr_points

    @tuplesetter(nr_points, len_tuple=2, valid_types=int)
    def nr_points(self, nr_points):
        return nr_points

    @property
    def start(self):
        return self._start

    @tuplesetter(start, len_tuple=2, valid_types=(int, float))
    def start(self, start):
        return start

    def _calc_coordinates(self):
        coords = []
        for i in range(2):
            start = self.start[i]
            res = self.resolution[i]
            forward_steps = res*self.nr_points[i]
            curr_coord = np.linspace(start, start+forward_steps,
                                     self.nr_points[i], endpoint=False)
            coords.append(curr_coord)
        return tuple(coords)

    def _coords_bounds(self):
        coords = []
        for i in range(2):
            res = self.resolution[i]
            start = self.start[i] - res/2
            forward_steps = res*(self.nr_points[i]+1)
            curr_coord = np.linspace(start, start+forward_steps,
                                     self.nr_points[i]+1, endpoint=False)
            coords.append(curr_coord)
        return tuple(coords)

    @property
    @abc.abstractmethod
    def center_distance(self):
        pass

    @abc.abstractmethod
    def _coords2latlon(self, *args):
        """
        Calculate the latitude and longitude coordinates for this grid based on
        given coordinates.

        Returns
        -------
        latitude : numpy.ndarray
            The calculated latitude coordinate as numpy.ndarray with the shape
            of the coordinates.
        longitude : numpy.ndarray
            The calculated longitude coordinate as numpy.ndarray with the shape
            of the coordinates.
        """
        pass
