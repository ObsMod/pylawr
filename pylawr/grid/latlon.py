#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import numpy as np

# Internal modules
from .rectangular import RectangularGrid


logger = logging.getLogger(__name__)


class LatLonGrid(RectangularGrid):
    """
    The LatLonGrid can be used to translate RadarGrids into NWP weather
    model grids. The calculated coordinates are absolute values and valid
    for the centroids of the grid points.

    Parameters
    ----------
    resolution : tuple(float or int), float or int, optional
        The resolution of the LatLonGrid in degrees. The first value is
        valid for `lat`, while the second one is valid for `lon`. If the
        parameter is float or int, then the resolution will be set to equal
        values for `lat` and `lon`. Only the first two entries of a tuple
        will be used. Default is (0.0009, 0.001515).
    start : tuple(float or int), float or int, optional
        The start position of the lower left grid point centroid within the
        LatLonGrid in degrees. This position is relative to the center.
        The first value is valid for `lat`, while the second one is valid
        for `lon`. If the parameter is float or int, then the resolution
        will be set to equal values for `lat` and `lon`. Only the first two
        entries of a tuple will be used. Default is (53.387, 9.673).
    nr_points : tuple(int) or int, optional
        The number of points for both directions. The first value is
        valid for `lat`, while the second one is valid for `lon`. If the
        tuple is int, then the resolution will be set to equal values for
        `lat` and `lon`. Only the first two entries of a tuple will be used.
        Default is 401.
    center : tuple(float), optional
        The center of the grid. The center should have two or three entries
        within the tuple. The first entry is the latitude position of the
        grid center in degree. The second entry is the longitude position of
        the grid center in degree. The optional third entry is the height of
        the center above mean sea level in degrees. If no third entry is
        given the center height is set to zero. The center is
        used to transform the polar coordinates into latitude and longitude
        coordinates. It is also used to calculate the altitude map together
        with the beam elevation. Default is the position on  the top of the
        "Geomatikum": (53.56796, 9.97451, 95). The center is also the center
        of the equator. Thus, the north pole for the rotated pole
        coordinates will be calculated based on this parameter. The altitude
        will be set to constant, based on the defined third value.
    beam_ele : float, optional
        The beam elevation in degrees. The beam elevation is used to
        calculate the altitude map. The beam elevation is set constant if
        it is a float. The value of the beam elevation are normalized for
        angle between -90 and 90 degrees such that the sign is preserved.
        This beam elevation is used to calculate the elevation for a grid
        point. The calculated elevation is valid for the whole grid box.
    """
    def __init__(self, resolution=(0.0009, 0.001515), start=(53.387, 9.673),
                 nr_points=401, center=(53.56796, 9.97451, 95), beam_ele=3.):
        super().__init__(resolution, start, nr_points, center, beam_ele)
        self._coord_names = ('latitude', 'longitude')

    @staticmethod
    def _latlon2rad(latitude, longitude):
        """
        Convert given latitude and longitude pair into radians.

        Parameters
        ----------
        latitude : array_like
            The latitude in degrees which should be converted into radians.
        longitude : array_like
            The longitude in degrees which should be converted into radians.

        Returns
        -------
        rad_lat : array_like
            The converted latitude in radians.
        rad_lon : array_like
            The converted longitude in radians.
        """
        return np.deg2rad(latitude), np.deg2rad(longitude)

    @staticmethod
    def _get_distance(p1, p2):
        """
        Calculate the distance between two points.

        Parameters
        ----------
        p1 : tuple(array_like, array_like)
            The latitude and longitude of the first point in radians.
        p2 : tuple(array_like, array_like)
            The latitude and longitude of the second point in radians.

        Returns
        -------
        dlat : array_like
            The difference in the latitude coordinates between second point and
            first point in radians.
        dlon : array_like
            The difference in the longitude coordinates between second point
            and first point in radians.
        """
        dlat = p2[0] - p1[0]
        dlon = p2[1] - p1[1]
        return dlat, dlon

    def _haversine_formula(self, p1, p2, dlat, dlon):
        """
        Calculate the great circle distance between two points
        on the earth. The formula is based on the haversine formula
        :cite:`mendoza1795`.

        Parameters
        ----------
        p1 : tuple (array_like, array_like)
            The coordinates (latitude, longitude) of the first point in radians.
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
            The coordinates (latitude, longitude) of the first point in radians.
        p2 : tuple (array_like, array_like)
            The coordinates (latitude, longitude) of the second point in
            radians.

        Returns
        -------
        distance : array_like
            The calculated haversine distance in meters.
        """
        p1_rad = self._latlon2rad(*p1)
        p2_rad = self._latlon2rad(*p2)
        dlat, dlon = self._get_distance(p1_rad, p2_rad)
        distance = self._haversine_formula(p1_rad, p2_rad, dlat, dlon)
        return distance

    @property
    def center_distance(self):
        distance = self._haversine_distance(self.coord_fields, self.center[:2])
        return distance

    def _coords2latlon(self, *coords):
        """
        Returns the given coordinates because they are already latitude and
        longitude.

        Returns
        -------
        latitude : numpy.ndarray
            The calculated latitude coordinate as numpy.ndarray with the shape
            of the coordinates.
        longitude : numpy.ndarray
            The calculated longitude coordinate as numpy.ndarray with the shape
            of the coordinates.
        """
        return coords[0], coords[1]
