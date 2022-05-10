#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import numpy as np
import pandas as pd

# Internal modules
from .base import BaseGrid


logger = logging.getLogger(__name__)


class UnstructuredGrid(BaseGrid):
    """
    An unstructured grid is a grid, where for every grid point latitude and
    longitude values are given, but they don't need to be in their own
    structured grid. This grid can be used to read in icosahedral model
    data.
    Further it is used for hole interpolation within this pylawr package.
    Compared to other grids this grid is only an one-dimensional grid with a
    coordinate called ``grid_cell``.

    Parameters
    ----------
    in_coords : :py:class:``numpy.ndarray``
        The input coordinates in degrees. These input coordinates are used
        as coordinates. The array need to have two axes. The first dimension
        is also used as grid dimension. The second dimension indicates if
        only latitude and longitude (2) is given or if an additional
        altitude in metre (3) is passed. If no altitude is given,
        the altitude is calculated based on the distance to the center.
    center : tuple(float), optional
        The center of the grid. The centre should have two or three entries
        within the tuple. The first entry is the latitude position of the
        grid center in degree. The second entry is the longitude position of
        the grid centre in degree. The optional third entry is the height of
        the center above mean sea level in meters. If no third entry is
        given the center height is set to zero. The centre is
        used to transform the polar coordinates into latitude and longitude
        coordinates. It is also used to calculate the altitude map together
        with the beam elevation. Default is the position on  the top of the
        "Geomatikum": (53.56796, 9.97451, 95).
    beam_ele : float, optional
        The beam elevation in degrees. The beam elevation is used to
        calculate the altitude map. The beam elevation is set constant if
        it is a float. The value of the beam elevation are normalized for
        angle between -90 and 90 degrees such that the sign is preserved.
        This beam elevation is used to calculate the elevation for a grid
        point. The calculated elevation is valid for the whole grid box.
    """
    def __init__(self, in_coords, center=(53.56796, 9.97451, 95), beam_ele=3.):
        super().__init__(center, beam_ele)
        self.in_coords = in_coords
        self._coord_names = ('grid_cell', )

    @property
    def _data_shape(self):
        return (self.in_coords.shape[0],)

    @_data_shape.setter
    def _data_shape(self, new_shape):
        """
        This setter is needed because in base_grid the data shape is set.
        """
        pass

    @property
    def center_distance(self):
        """
        Get the distance to the center in meters.

        Returns
        -------
        distance : numpy.ndarray
            The distance to the center in meters.
        """
        distance = self._haversine_distance(self.lat_lon, self.center[:2])
        return distance

    def _calc_coordinates(self):
        return self.in_coords[:, :2],

    def _coords2latlon(self, *args):
        coord_array = args[0]
        return coord_array[:, 0], coord_array[:, 1]

    def _calc_altitude(self, *args):
        """
        Calculate the altitude map for this grid based on the given altitude
        of the `in_coords` or, if no altitude is given, on the distance to
        the radar site, the centre position height and the beam elevation.
        For the altitude calculation refer to the super class
        :py:class:`pylawr.grid.BaseGrid`.

        Parameters
        ----------
        args
            Optional parameters, need to be provided if the altitude
            calculation of :py:class:`pylawr.grid.BaseGrid` is applied.

        Returns
        -------
        altitude : numpy.ndarray
            The altitude.
        """
        try:
            altitude = self.in_coords[:, 2]
        except IndexError:
            altitude = super()._calc_altitude(*args)
        return altitude

    @staticmethod
    def _calc_coord_fields(*coords):
        return (coords[0], )

    def get_coordinates(self):
        """
        Get the coordinates for this grid in a xarray-conform structure.

        Returns
        -------
        coords : dict(str, (str, numpy.ndarray))
            The coordinates for this grid in a xarray-conform structure. The
            key is the coordinate name. The coordinates have as value a tuple
            with their own name, indicating that the they are self-describing,
            and the coordinate values as numpy array.
        """
        coords_data = self.coords
        coord_index = pd.MultiIndex.from_arrays(
            coords_data[0].T, names=('grid_lat', 'grid_lon')
        )
        coords = {
            'grid_cell': (('grid_cell', ), coord_index)
        }
        return coords

    def get_multiindex(self):
        multiindex = self.get_coordinates()['grid_cell'][1]
        return multiindex

    def _haversine_formula(self, p1, p2, dlat, dlon):
        """
        Calculate the great circle distance between two points
        on the earth. The formula is based on the haversine formula
        :cite:`mendoza1795`.

        Parameters
        ----------
        p1 : tuple (array_like, array_like)
            The coordinates (latitude, longitude) of the first point in
            radians.
        p2 : tuple (array_like, array_like)
            The coordinates (latitude, longitude) of the second point in
            radians.
        dlat : array_like
            The distance between the first and the second latitude in radians.
        dlon : array_like
            The distance between the first and the second longitude in radians.

        Returns
        -------
        distance : array_like
            The calculated haversine distance in meters.

        Notes
        -----
        The haversine distance is calculated based on the earth radius, given
        within the initialization.
        """
        a = np.sin(dlat / 2.0) ** 2
        a += np.cos(p1[0]) * np.cos(p2[0]) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = self.earth_radius * c
        return distance

    def _haversine_distance(self, p1, p2):
        """
        Calculate the great circle distance between two points
        on the earth. The formula is based on the haversine formula
        :cite:`mendoza1795`.

        Parameters
        ----------
        p1 : tuple (array_like, array_like)
            The coordinates (latitude, longitude) of the first point in
            degrees.
        p2 : tuple (array_like, array_like)
            The coordinates (latitude, longitude) of the second point in
            degrees.

        Returns
        -------
        distance : array_like
            The calculated haversine distance in meters.
        """
        p1_rad = np.deg2rad(p1)
        p2_rad = np.deg2rad(p2)
        dlat = p2_rad[0] - p1_rad[0]
        dlon = p2_rad[1] - p1_rad[1]
        distance = self._haversine_formula(p1_rad, p2_rad, dlat, dlon)
        return distance
