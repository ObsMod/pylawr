#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import numpy as np
import cartopy.crs as ccrs

# Internal modules
from .rectangular import RectangularGrid


logger = logging.getLogger(__name__)


class CartesianGrid(RectangularGrid):
    """
    The CartesianGrid can be used for easier composite calculations of
    radar fields. The calculated coordinates are relative to the given
    center and valid for the centroids of the grid points.

    Parameters
    ----------
    resolution : tuple(float or int), float or int, optional
        The resolution of the CartesianGrid in meters. The first value is
        valid for `y`, while the second one is valid for `x`. If the
        parameter is float or int, then the resolution will be set to equal
        values for `y` and `x`. Only the first two entries of a tuple will
        be used. Default is 100.
    start : tuple(float or int), float or int, optional
        The start position of the lower left grid point centroid within the
        CartesianGrid in meters. This position is relative to the center.
        The first value is valid for `y`, while the second one is valid for
        `x`. If the parameter is float or int, then the resolution will be
        set to equal values for `y` and `x`. Only the first two entries of a
        tuple will be used. Default is -20000.
    nr_points : tuple(int) or int, optional
        The number of points for both directions. The first value is
        valid for `y`, while the second one is valid for `x`. If the tuple
        is int, then the resolution will be set to equal values for
        `y` and `x`. Only the first two entries of a tuple will be used.
        Default is 401.
    center : tuple(float), optional
        The center of the grid. The center should have two or three entries
        within the tuple. The first entry is the latitude position of the
        grid center in degree. The second entry is the longitude position of
        the grid center in degree. The optional third entry is the height of
        the center above mean sea level in meters. If no third entry is
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
    def __init__(self, resolution=100, start=-20000, nr_points=401,
                 center=(53.56796, 9.97451, 95), beam_ele=3.):
        super().__init__(resolution, start, nr_points, center, beam_ele)
        self._coord_names = ('y', 'x')

    @property
    def meters2deg(self):
        earth_perimeter = self.earth_radius * 2 * np.pi
        deg_per_m = 360 / earth_perimeter
        return deg_per_m

    @property
    def center_distance(self):
        distance_pow = self.coord_fields[0] ** 2 + self.coord_fields[1] ** 2
        distance = np.sqrt(distance_pow)
        return distance

    @property
    def north_pole(self):
        """
        Get the north pole for given center. At the moment, only center
        coordinates in the first quadrant (latitude and longitude are positive)
        are used to calculate the north pole.

        Returns
        -------
        lat : float
            The latitude of the north pole.
        lon : float
            The longitude of the north pole.

        Raises
        ------
        ValueError
            A ValueError is raised if the center coordinates are not within the
            first quadrant.
        """
        if self.center[0] < 0 or self.center[1] < 0:
            raise ValueError(
                'The given center coordinates are not within the first '
                'quadrant, this is not supported at the moment!'
            )
        lat = 90 - self.center[0]
        lon = self.center[1] - 180
        return lat, lon

    def _coords2latlon(self, *coords):
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
        orig_y, orig_x = coords
        orig_lat = orig_y * self.meters2deg
        orig_lon = orig_x * self.meters2deg
        pole_lat, pole_lon = self.north_pole
        origin_proj = ccrs.RotatedPole(pole_longitude=pole_lon,
                                       pole_latitude=pole_lat)
        target_proj = ccrs.PlateCarree()
        new_coords = target_proj.transform_points(
            src_crs=origin_proj,
            x=orig_lon,
            y=orig_lat
        )
        longitude = new_coords[..., 0]
        latitude = new_coords[..., 1]
        return latitude, longitude
